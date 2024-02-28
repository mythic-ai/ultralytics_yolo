# Ultralytics YOLO 🚀, AGPL-3.0 license

from ultralytics.engine.results import Results
from ultralytics.models.yolo.detect.predict import DetectionPredictor
from ultralytics.utils import DEFAULT_CFG, LOGGER, ops


class SegPosePredictor(DetectionPredictor):
    """
    A class extending the DetectionPredictor class for prediction based on a SegPose model.

    Example:
        ```python
        from ultralytics.utils import ASSETS
        from ultralytics.models.yolo.segpose import SegPose!Predictor

        args = dict(model='yolov8n-seg.pt', source=ASSETS)
        predictor = SegPosePredictor(overrides=args)
        predictor.predict_cli()
        ```
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """Initializes the SegPosePredictor with the provided configuration, overrides, and callbacks."""
        super().__init__(cfg, overrides, _callbacks)
        self.args.task = "segose"
        if isinstance(self.args.device, str) and self.args.device.lower() == "mps":
            LOGGER.warning(
                "WARNING ⚠️ Apple MPS known Pose bug. Recommend 'device=cpu' for Pose models. "
                "See https://github.com/ultralytics/ultralytics/issues/4031."
            )

    def postprocess(self, preds, img, orig_imgs):
        """Applies non-max suppression and processes detections for each image in an input batch."""
        p = ops.non_max_suppression(
            preds[0],
            self.args.conf,
            self.args.iou,
            agnostic=self.args.agnostic_nms,
            max_det=self.args.max_det,
            nc=len(self.model.names),
            classes=self.args.classes,
        )

        if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        seg_start_index = 6
        num_masks = self.model.nm
        kpt_start_index = seg_start_index + num_masks

        results = []
        proto = preds[1][2] if isinstance(preds[1], tuple) else preds[1]  # tuple if PyTorch model or array if exported
        for i, pred in enumerate(p):
            orig_img = orig_imgs[i]
            pred_kpts = (pred[:, kpt_start_index:].view(len(pred), *self.model.kpt_shape)
                         if len(pred) else pred[:, kpt_start_index:])
            pred_kpts = ops.scale_coords(img.shape[2:], pred_kpts, orig_img.shape)
            img_path = self.batch[0][i]
            if not len(pred):  # save empty boxes
                masks = None
            elif self.args.retina_masks:
                pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
                masks = ops.process_mask_native(proto[i], pred[:, seg_start_index:kpt_start_index],
                                                pred[:, :4], orig_img.shape[:2])  # HWC
            else:
                masks = ops.process_mask(proto[i], pred[:, seg_start_index:kpt_start_index], pred[:, :4], img.shape[2:],
                                         upsample=True)  # HWC
                pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
            results.append(Results(orig_img, path=img_path, names=self.model.names, boxes=pred[:, :seg_start_index],
                                   masks=masks, keypoints=pred_kpts))
        return results
