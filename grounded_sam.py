from groundingdino.util.inference import Model as GDModel
from segment_anything import sam_model_registry, SamPredictor
import cv2
import numpy as np
import supervision as sv
import os
import torch
import torchvision
from collections import defaultdict as dd

torch.cuda.set_device(0)  # set the GPU device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

GD_HOME = "../GroundingDINO/"
SAM_HOME = "../segment-anything-main/"

# GroundingDINO config and checkpoint
GROUNDING_DINO_CONFIG_PATH = os.path.join(GD_HOME, "groundingdino/config/GroundingDINO_SwinT_OGC.py")
GROUNDING_DINO_CHECKPOINT_PATH = os.path.join(GD_HOME, "groundingdino_swint_ogc.pth")

# Segment-Anything checkpoint
SAM_ENCODER_VERSION = "vit_h"
SAM_CHECKPOINT_PATH = os.path.join(SAM_HOME, "sam_vit_h_4b8939.pth")


class GSAM:
    def __init__(self):
        self.gd_model = GDModel(model_config_path=GROUNDING_DINO_CONFIG_PATH,
                                model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH)
        sam = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH)
        sam.to(device=DEVICE)
        self.sam_predictor = SamPredictor(sam)

        self.BOX_THRESHOLD = 0.25
        self.TEXT_THRESHOLD = 0.25
        self.NMS_THRESHOLD = 0.8

        self.box_annotator = sv.BoxAnnotator()
        self.mask_annotator = sv.MaskAnnotator()

    def segment(self, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
        self.sam_predictor.set_image(image)
        result_masks = []
        for box in xyxy:
            masks, scores, logits = self.sam_predictor.predict(
                box=box,
                multimask_output=True
            )
            index = np.argmax(scores)
            result_masks.append(masks[index])
        return np.array(result_masks)

    def run_single_image(self,
                         image_path,
                         classes,
                         dsize=(512, 512),
                         de_duplicated=True,
                         save_name=None,
                         save_box_path=None,
                         save_mask_path=None,
                         visible=False):
        image = cv2.imread(image_path)
        image = cv2.resize(image, dsize)
        detections = self.gd_model.predict_with_classes(
            image=image,
            classes=classes,
            box_threshold=self.BOX_THRESHOLD,
            text_threshold=self.BOX_THRESHOLD
        )
        if visible:
            print(f"Before NMS: {len(detections.xyxy)} boxes")
        nms_idx = torchvision.ops.nms(
            torch.from_numpy(detections.xyxy),
            torch.from_numpy(detections.confidence),
            self.NMS_THRESHOLD
        ).numpy().tolist()
        detections.xyxy = detections.xyxy[nms_idx]
        detections.confidence = detections.confidence[nms_idx]
        detections.class_id = detections.class_id[nms_idx]
        if visible:
            print(f"After NMS: {len(detections.xyxy)} boxes")

        if len(detections.class_id) > len(classes) and de_duplicated:
            sdict = dict()
            for idx, (cls_id, conf) in enumerate(zip(detections.class_id, detections.confidence)):
                if cls_id not in sdict:
                    sdict[cls_id] = (idx, conf)
                else:
                    if sdict[cls_id][1] < conf:
                        sdict[cls_id] = (idx, conf)
            keep_idx = [v[0] for k, v in sdict.items()]
            detections.xyxy = detections.xyxy[keep_idx]
            detections.confidence = detections.confidence[keep_idx]
            detections.class_id = detections.class_id[keep_idx]
            if visible:
                print(f"After De-duplicating: {len(detections.xyxy)} boxes")

        # sort the detections according to the given classes
        sorted_idx = np.argsort(detections.class_id)
        detections.xyxy = detections.xyxy[sorted_idx]
        detections.confidence = detections.confidence[sorted_idx]
        detections.class_id = detections.class_id[sorted_idx]

        labels = [
            f"{classes[class_id]} {confidence:0.2f}"
            for _, _, confidence, class_id, _
            in detections]

        annotated_frame = self.box_annotator.annotate(scene=image.copy(), detections=detections, labels=labels)

        if save_box_path is not None:
            cv2.imwrite(os.path.join(save_box_path, "{}_gd_image.jpg".format(save_name)), annotated_frame)

        # convert detections to masks
        detections.mask = self.segment(
            image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
            xyxy=detections.xyxy
        )
        annotated_image = self.mask_annotator.annotate(scene=image.copy(), detections=detections)
        annotated_image = self.box_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)

        if save_mask_path is not None:
            cv2.imwrite(os.path.join(save_mask_path, "{}_gsam_image.jpg".format(save_name)), annotated_image)

        return detections
