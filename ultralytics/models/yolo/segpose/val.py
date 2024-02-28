# Ultralytics YOLO 🚀, AGPL-3.0 license

from multiprocessing.pool import ThreadPool
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils import LOGGER, NUM_THREADS, ops
from ultralytics.utils.checks import check_requirements
from ultralytics.utils.metrics import OKS_SIGMA, SegPoseMetrics, box_iou, kpt_iou, mask_iou
from ultralytics.utils.plotting import output_to_target, plot_images


class SegPoseValidator(DetectionValidator):
    """
    A class for validation based on a Pose and Segmentation models.

    Example:
        ```python
        from ultralytics.models.yolo.segpose import SegPoseValidator

        args = dict(model='yolov8n-segpose.pt', data='coco8-segpose.yaml')
        validator = SegPoseValidator(args=args)
        validator()
        ```
    """

    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None):
        """Initialize SegPoseValidator with custom parameters and assigned attributes."""
        super().__init__(dataloader, save_dir, pbar, args, _callbacks)
        self.plot_masks = None
        self.process = None
        self.sigma = None
        self.kpt_shape = None
        self.args.task = "segpose"
        self.metrics = SegPoseMetrics(save_dir=self.save_dir, on_plot=self.on_plot)

    def preprocess(self, batch):
        """Preprocesses the batch by converting masks and keypoints to float and sending to device."""
        batch = super().preprocess(batch)
        batch["masks"] = batch["masks"].to(self.device).float()
        batch["keypoints"] = batch["keypoints"].to(self.device).float()
        return batch

    def init_metrics(self, model):
        """Initialize metrics and select mask processing function based on save_json flag."""
        super().init_metrics(model)
        self.plot_masks = []
        if self.args.save_json:
            check_requirements("pycocotools>=2.0.6")
            self.process = ops.process_mask_upsample  # more accurate
        else:
            self.process = ops.process_mask  # faster
        self.kpt_shape = self.data["kpt_shape"]
        is_pose = self.kpt_shape == [17, 3]
        nkpt = self.kpt_shape[0]
        self.sigma = OKS_SIGMA if is_pose else np.ones(nkpt) / nkpt
        self.stats = dict(tp_p=[], target_cls_p=[], tp_m=[], tp=[], conf=[], pred_cls=[], target_cls=[])
        self.seg_start_index = 6
        num_masks = model.nm
        self.kpt_start_index = self.seg_start_index + num_masks

    def get_desc(self):
        """Return a formatted description of evaluation metrics."""
        return ("%22s" + "%11s" * 14) % (
            "Class",
            "Images",
            "Instances",
            "Box(P",
            "R",
            "mAP50",
            "mAP50-95)",
            "Mask(P",
            "R",
            "mAP50",
            "mAP50-95)",
            "Pose(P",
            "R",
            "mAP50",
            "mAP50-95)",
        )

    def postprocess(self, preds):
        """Post-processes YOLO predictions and returns output detections with proto."""
        p = ops.non_max_suppression(
            preds[0],
            self.args.conf,
            self.args.iou,
            labels=self.lb,
            multi_label=True,
            agnostic=self.args.single_cls,
            max_det=self.args.max_det,
            nc=self.nc,
        )
        proto = preds[1][2] if len(preds[1]) == 4 else preds[1]  # second output is len 4 if pt, but only 1 if exported
        return p, proto

    def _prepare_batch(self, si, batch):
        """Prepares a batch for training or inference by processing images and targets."""
        prepared_batch = super()._prepare_batch(si, batch)
        midx = [si] if self.args.overlap_mask else batch["batch_idx"] == si
        prepared_batch["masks"] = batch["masks"][midx]

        kpts = batch["keypoints"][batch["batch_idx"] == si]
        h, w = prepared_batch["imgsz"]
        kpts = kpts.clone()
        kpts[..., 0] *= w
        kpts[..., 1] *= h
        kpts = ops.scale_coords(prepared_batch["imgsz"], kpts, prepared_batch["ori_shape"],
                                ratio_pad=prepared_batch["ratio_pad"])
        prepared_batch["kpts"] = kpts
        return prepared_batch

    def _prepare_pred(self, pred, pbatch, proto):
        """Prepares a batch for training or inference by processing images and targets."""
        predn = super()._prepare_pred(pred, pbatch)
        pred_masks = self.process(proto, pred[:, self.seg_start_index:self.kpt_start_index],
                                  pred[:, :4], shape=pbatch["imgsz"])
        nk = pbatch["kpts"].shape[1]
        pred_kpts = predn[:, self.kpt_start_index:].view(len(predn), nk, -1)
        ops.scale_coords(pbatch["imgsz"], pred_kpts, pbatch["ori_shape"], ratio_pad=pbatch["ratio_pad"])
        return predn, pred_masks, pred_kpts

    def update_metrics(self, preds, batch):
        """Metrics."""
        for si, (pred, proto) in enumerate(zip(preds[0], preds[1])):
            self.seen += 1
            npr = len(pred)
            stat = dict(
                conf=torch.zeros(0, device=self.device),
                pred_cls=torch.zeros(0, device=self.device),
                tp=torch.zeros(npr, self.niou, dtype=torch.bool, device=self.device),
                tp_m=torch.zeros(npr, self.niou, dtype=torch.bool, device=self.device),
                tp_p=torch.zeros(npr, self.niou, dtype=torch.bool, device=self.device)
            )
            pbatch = self._prepare_batch(si, batch)
            cls, bbox = pbatch.pop("cls"), pbatch.pop("bbox")
            gt_kpts = pbatch["kpts"]

            # Filter out labels that don't have keypoints to avoid overcounting the total number of Pose labels
            # in ap_per_class.
            kpts_exist = gt_kpts[:, :, 2].sum(dim=1) != 0
            stat["target_cls_p"] = cls[kpts_exist]
            assert all(cls[kpts_exist] == 0)

            nl = len(cls)
            stat["target_cls"] = cls
            if npr == 0:
                if nl:
                    for k in self.stats.keys():
                        self.stats[k].append(stat[k])
                    if self.args.plots:
                        self.confusion_matrix.process_batch(detections=None, gt_bboxes=bbox, gt_cls=cls)
                continue

            # Masks
            gt_masks = pbatch.pop("masks")
            # Predictions
            if self.args.single_cls:
                pred[:, 5] = 0
            predn, pred_masks, pred_kpts = self._prepare_pred(pred, pbatch, proto)
            stat["conf"] = predn[:, 4]
            stat["pred_cls"] = predn[:, 5]

            # Evaluate
            if nl:
                stat["tp"] = self._process_batch(predn, bbox, cls)
                stat["tp_m"] = self._process_batch(
                    predn, bbox, cls, pred_masks=pred_masks, gt_masks=gt_masks,
                    overlap=self.args.overlap_mask, masks=True
                )
                stat["tp_p"] = self._process_batch(predn, bbox, cls, pred_kpts=pred_kpts, gt_kpts=gt_kpts)
                if self.args.plots:
                    self.confusion_matrix.process_batch(predn, bbox, cls)

            for k in self.stats.keys():
                self.stats[k].append(stat[k])

            pred_masks = torch.as_tensor(pred_masks, dtype=torch.uint8)
            if self.args.plots and self.batch_i < 3:
                self.plot_masks.append(pred_masks[:15].cpu())  # filter top 15 to plot

            # Save
            if self.args.save_json:
                pred_masks = ops.scale_image(
                    pred_masks.permute(1, 2, 0).contiguous().cpu().numpy(),
                    pbatch["ori_shape"],
                    ratio_pad=batch["ratio_pad"][si],
                )
                self.pred_to_json(predn, batch["im_file"][si], pred_masks)
            # if self.args.save_txt:
            #    save_one_txt(predn, save_conf, shape, file=save_dir / 'labels' / f'{path.stem}.txt')

    def finalize_metrics(self, *args, **kwargs):
        """Sets speed and confusion matrix for evaluation metrics."""
        self.metrics.speed = self.speed
        self.metrics.confusion_matrix = self.confusion_matrix

    # FIXME: Update the docstring.
    def _process_batch(self, detections, gt_bboxes, gt_cls, pred_masks=None, gt_masks=None, overlap=False, masks=False,
                       pred_kpts=None, gt_kpts=None):
        """
        Return correct prediction matrix.

        Args:
            detections (array[N, 6]), x1, y1, x2, y2, conf, class
            labels (array[M, 5]), class, x1, y1, x2, y2
            pred_kpts (torch.Tensor, optional): Tensor of shape [N, 51] representing predicted keypoints.
                51 corresponds to 17 keypoints each with 3 values.
            gt_kpts (torch.Tensor, optional): Tensor of shape [N, 51] representing ground truth keypoints.

        Returns:
            correct (array[N, 10]), for 10 IoU levels
        """
        assert not masks or pred_kpts is None, "masks and keypoints cannot be processed together"
        if masks:
            if overlap:
                nl = len(gt_cls)
                index = torch.arange(nl, device=gt_masks.device).view(nl, 1, 1) + 1
                gt_masks = gt_masks.repeat(nl, 1, 1)  # shape(1,640,640) -> (n,640,640)
                gt_masks = torch.where(gt_masks == index, 1.0, 0.0)
            if gt_masks.shape[1:] != pred_masks.shape[1:]:
                gt_masks = F.interpolate(gt_masks[None], pred_masks.shape[1:], mode="bilinear", align_corners=False)[0]
                gt_masks = gt_masks.gt_(0.5)
            iou = mask_iou(gt_masks.view(gt_masks.shape[0], -1), pred_masks.view(pred_masks.shape[0], -1))
        elif pred_kpts is not None and gt_kpts is not None:
            # FIXME: This may not be necessary. The single class case works without this and in the multi-class case
            # this filtering by itself does not seem to help. Something else needs to be fixed first.
            # Filter out labels that don't have keypoints to avoid overcounting the total number of Pose labels.
            kpts_exist = gt_kpts[:, :, 2].sum(dim=1) != 0
            gt_kpts = gt_kpts[kpts_exist]
            gt_cls = gt_cls[kpts_exist]
            gt_bboxes = gt_bboxes[kpts_exist]
            pose_detections = detections[:, 5] == 0
            detections = detections[pose_detections]
            pred_kpts = pred_kpts[pose_detections]
            # `0.53` is from https://github.com/jin-s13/xtcocoapi/blob/master/xtcocotools/cocoeval.py#L384
            area = ops.xyxy2xywh(gt_bboxes)[:, 2:].prod(1) * 0.53
            iou = kpt_iou(gt_kpts, pred_kpts, sigma=self.sigma, area=area)
        else:  # boxes
            iou = box_iou(gt_bboxes, detections[:, :4])

        return self.match_predictions(detections[:, 5], gt_cls, iou)

    def plot_val_samples(self, batch, ni):
        """Plots validation samples with bounding box labels."""
        plot_images(
            batch["img"],
            batch["batch_idx"],
            batch["cls"].squeeze(-1),
            batch["bboxes"],
            masks=batch["masks"],
            kpts=batch["keypoints"],
            paths=batch["im_file"],
            fname=self.save_dir / f"val_batch{ni}_labels.jpg",
            names=self.names,
            on_plot=self.on_plot,
        )

    def plot_predictions(self, batch, preds, ni):
        """Plots batch predictions with masks and bounding boxes."""
        max_det = 15 # not set to self.args.max_det due to slow plotting speed
        pred_kpts = torch.cat([p[:max_det, self.kpt_start_index:].view(-1, *self.kpt_shape) for p in preds[0]], 0)
        plot_images(
            batch["img"],
            *output_to_target(preds[0], max_det=max_det),
            masks=torch.cat(self.plot_masks, dim=0) if len(self.plot_masks) else self.plot_masks,
            kpts=pred_kpts,
            paths=batch["im_file"],
            fname=self.save_dir / f"val_batch{ni}_pred.jpg",
            names=self.names,
            on_plot=self.on_plot,
        )  # pred
        self.plot_masks.clear()

    def pred_to_json(self, predn, filename, pred_masks):
        """
        Save one JSON result.

        Examples:
             >>> result = {"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}
        """
        from pycocotools.mask import encode  # noqa

        def single_encode(x):
            """Encode predicted masks as RLE and append results to jdict."""
            rle = encode(np.asarray(x[:, :, None], order="F", dtype="uint8"))[0]
            rle["counts"] = rle["counts"].decode("utf-8")
            return rle

        stem = Path(filename).stem
        image_id = int(stem) if stem.isnumeric() else stem
        box = ops.xyxy2xywh(predn[:, :4])  # xywh
        box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
        pred_masks = np.transpose(pred_masks, (2, 0, 1))
        with ThreadPool(NUM_THREADS) as pool:
            rles = pool.map(single_encode, pred_masks)
        for i, (p, b) in enumerate(zip(predn.tolist(), box.tolist())):
            self.jdict.append(
                {
                    "image_id": image_id,
                    "category_id": self.class_map[int(p[5])],
                    "bbox": [round(x, 3) for x in b],
                    "keypoints": p[self.kpt_start_index:],
                    "score": round(p[4], 5),
                    "segmentation": rles[i],
                }
            )

    def eval_json(self, stats):
        """Return COCO-style object detection evaluation metrics."""
        if self.args.save_json and self.is_coco and len(self.jdict):
            seg_anno_json = self.data["path"] / "annotations/instances_val2017.json"  # annotations
            pose_anno_json = self.data["path"] / "annotations/person_keypoints_val2017.json"  # annotations
            pred_json = self.save_dir / "predictions.json"  # predictions
            LOGGER.info(f"\nEvaluating pycocotools mAP using {pred_json}, {seg_anno_json}, and {pose_anno_json}...")
            try:  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
                check_requirements("pycocotools>=2.0.6")
                from pycocotools.coco import COCO  # noqa
                from pycocotools.cocoeval import COCOeval  # noqa

                for x in seg_anno_json, pose_anno_json, pred_json:
                    assert x.is_file(), f"{x} file not found"
                seg_anno = COCO(str(seg_anno_json))  # init annotations api
                pose_anno = COCO(str(pose_anno_json))  # init annotations api
                # FIXME: Do we need separate seg_pred and pose_pred?
                seg_pred = seg_anno.loadRes(str(pred_json))  # init predictions api (must pass string, not Path)
                pose_pred = pose_anno.loadRes(str(pred_json))  # init predictions api (must pass string, not Path)
                for i, eval in enumerate([COCOeval(seg_anno, seg_pred, "bbox"),
                                          COCOeval(seg_anno, seg_pred, "segm"),
                                          COCOeval(pose_anno, pose_pred, "keypoints"),
                                          COCOeval(pose_anno, pose_pred, "bbox"),]):
                    eval.params.imgIds = [int(Path(x).stem) for x in self.dataloader.dataset.im_files]  # im to eval
                    eval.params.catIds = [self.class_map[c] for c in range(self.nc)]
                    eval.evaluate()
                    eval.accumulate()
                    eval.summarize()
                    idx = i * 4 + 2
                    if idx < len(self.metrics.keys):
                        # update mAP50-95 and mAP50
                        stats[self.metrics.keys[idx + 1]], stats[self.metrics.keys[idx]] = eval.stats[:2]
            except Exception as e:
                LOGGER.warning(f"pycocotools unable to run: {e}")
        return stats
