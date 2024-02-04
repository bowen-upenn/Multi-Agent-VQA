# This file is modified from the official grounded_sam_simple_demo.py file at 
# https://github.com/IDEA-Research/Grounded-Segment-Anything/blob/main/grounded_sam_simple_demo.py
import cv2
import numpy as np
import supervision as sv

import torch
import torchvision

import sys
sys.path.append('./Grounded-Segment-Anything')

from segment_anything import sam_model_registry, SamPredictor
from groundingdino.util.inference import load_model, load_image, predict, annotate


def query_grounding_dino(device, args, model, image_path, text_prompt="bear."):
    # # Example code to init the GroundingDINO model
    # model = load_model("GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py", "./groundingdino_swint_ogc.pth")
    # If you want to detect multiple objects in one sentence with Grounding DINO, we suggest separating each name with . . 
    # An example: cat . dog . chair .

    image_source, image = load_image(image_path)

    boxes, logits, phrases = predict(
        model=model,
        image=image,
        caption=text_prompt,
        box_threshold=args['dino']['BOX_THRESHOLD'],
        text_threshold=args['dino']['TEXT_THRESHOLD'],
        device=device
    )

    # annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
    # cv2.imwrite("annotated_image.jpg", annotated_frame)
    return boxes, logits, phrases


def query_grounded_sam(rank, args, sam_predictor, image, class_prompt=["The running dog"]):
    # # Example code to init the GroundingSAM model
    # sam = sam_model_registry[args['sam']['SAM_ENCODER_VERSION']](checkpoint=args['sam']['SAM_CHECKPOINT_PATH'])
    # sam.to(rank)
    # sam_predictor = SamPredictor(sam)

    # detect objects
    detections = grounding_dino_model.predict_with_classes(
        image=image,
        classes=class_prompt,
        box_threshold=args['sam']['BOX_THRESHOLD'],
        text_threshold=args['sam']['TEXT_THRESHOLD']
    )

    # annotate image with detections
    box_annotator = sv.BoxAnnotator()
    labels = [
        f"{class_prompt[class_id]} {confidence:0.2f}" 
        for _, _, confidence, class_id, _ 
        in detections]
    annotated_frame = box_annotator.annotate(scene=image.copy(), detections=detections, labels=labels)

    # save the annotated grounding dino image
    # cv2.imwrite("groundingdino_annotated_image.jpg", annotated_frame)

    # NMS post process
    print(f"Before NMS: {len(detections.xyxy)} boxes")
    nms_idx = torchvision.ops.nms(
        torch.from_numpy(detections.xyxy), 
        torch.from_numpy(detections.confidence), 
        args['sam']['NMS_THRESHOLD']
    ).numpy().tolist()

    detections.xyxy = detections.xyxy[nms_idx]
    detections.confidence = detections.confidence[nms_idx]
    detections.class_id = detections.class_id[nms_idx]

    print(f"After NMS: {len(detections.xyxy)} boxes")

    # convert detections to masks
    detections.mask = segment(
        sam_predictor=sam_predictor,
        image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
        xyxy=detections.xyxy
    )

    # annotate image with detections
    box_annotator = sv.BoxAnnotator()
    mask_annotator = sv.MaskAnnotator()
    labels = [
        f"{CLASSES[class_id]} {confidence:0.2f}" 
        for _, _, confidence, class_id, _ 
        in detections]
    annotated_image = mask_annotator.annotate(scene=image.copy(), detections=detections)
    annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)

    # save the annotated grounded-sam image
    # cv2.imwrite("grounded_sam_annotated_image.jpg", annotated_image)
    return annotated_image
