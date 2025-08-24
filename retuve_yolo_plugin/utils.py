# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import os
import time

import cv2
import numpy as np
import torch
from retuve.classes.seg import SegFrameObjects, SegObject
from retuve.keyphrases.config import Config
from retuve.logs import ulogger

FILEDIR = os.path.abspath(f"{os.path.dirname(os.path.abspath(__file__))}/")


def get_mask(points, shape, color=(255, 255, 255)):
    contours = np.array([points], dtype=np.int32)
    mask = np.zeros(shape, np.uint8)
    mask = cv2.drawContours(mask, contours, -1, color, -1)

    return mask


def predict(
    images,
    weights=None,
    imgsz=512,
    conf=0.8,
    device=None,
    model=None,
    stream=False,
    chunk_size=1,  # Default chunk size
):
    """
    Predict the DICOM using a YOLO model with chunking support.
    """

    # Weights and YOLO are mutually exclusive
    if weights is None and model is None:
        raise ValueError("Either weights or model must be specified")

    if weights is not None and model is not None:
        raise ValueError("Either weights or model must be specified, not both")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    start = time.time()

    # Function to process images in chunks
    def process_in_chunks(images, chunk_size):
        for i in range(0, len(images), chunk_size):
            yield images[i : i + chunk_size]

    all_results = []

    for chunk in process_in_chunks(images, chunk_size):
        results = model.predict(
            chunk,
            imgsz=imgsz,
            conf=conf,
            verbose=False,
            stream=stream,
        )
        all_results.extend(results)  # Combine results from all chunks

    ulogger.info(f"YOLO Segmentation model time: {time.time() - start:.2f}s")

    return all_results


def yolo_predict_pose(
    images,
    keyphrase,
    default_weights,
    model=None,
    config=None,
    imgsz=800,
    conf=0.6,
    stream=False,
):
    if not config:
        config = Config.get_config(keyphrase)

    landmark_results = []

    if not model:
        from ultralytics import YOLO

        model = YOLO(default_weights)
        if "onnx" not in default_weights:
            model.to(config.device)

    results = predict(
        images=images,
        model=model,
        imgsz=imgsz,
        device=config.device,
        conf=conf,
        stream=stream,
    )

    # Prepare storage for best detections
    best_detections = {
        "left": {
            0: {"keypoints": [None, None]},
            1: {"keypoints": [None, None]},
        },
        "right": {
            0: {"keypoints": [None, None]},
            1: {"keypoints": [None, None]},
        },
    }

    for result in results:
        boxes = result.boxes.cpu().numpy()
        keypoints_list = result.keypoints.xy.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        orig_shape = result.boxes.orig_shape  # (height, width)
        img_center_x = orig_shape[1] / 2

        for box, keypoints, clss, conf in zip(boxes, keypoints_list, classes, confs):
            # Get box center x
            box = box.xyxy[0]
            x1, y1, x2, y2, *_ = box
            box_center_x = (x1 + x2) / 2

            # Decide side
            side = "left" if box_center_x < img_center_x else "right"

            # For each class, keep only the highest confidence detection per side
            if conf > best_detections[side][clss].get("conf", False):
                best_detections[side][int(clss)] = {
                    "box": box,
                    "keypoints": keypoints,
                    "conf": conf,
                }

        frame_landmarks = [
            best_detections["left"][0]["keypoints"],
            best_detections["left"][1]["keypoints"],
            best_detections["right"][0]["keypoints"],
            best_detections["right"][1]["keypoints"],
        ]

        frame_landmarks = [item for sublist in frame_landmarks for item in sublist]

        landmark_results.append(frame_landmarks)

    landmark_names = [
        "pel_l_o",
        "pel_l_i",
        "h_point_l",
        "fem_l",
        "pel_r_o",
        "pel_r_i",
        "h_point_r",
        "fem_r",
    ]

    return landmark_results, landmark_names


def shared_yolo_predict(
    images,
    keyphrase,
    default_weights,
    model=None,
    config=None,
    imgsz=512,
    conf=0.8,
    stream=False,
):

    if not config:
        config = Config.get_config(keyphrase)

    seg_results = []

    if not model:
        from ultralytics import YOLO

        model = YOLO(default_weights, task="segment")
        if "onnx" not in default_weights:
            model.to(config.device)

    attempts = 0
    while attempts < 10:
        try:
            results = predict(
                images=images,
                model=model,
                imgsz=imgsz,
                device=config.device,
                conf=conf,
                stream=stream,
            )
            break
        except torch.cuda.OutOfMemoryError:
            # wipe process GPU memory
            print("Out of memory. Retrying...")
            time.sleep(15)
            attempts += 1
            torch.cuda.empty_cache()

    timings = []

    for result in results:
        start = time.time()

        img = result.orig_img

        seg_frame_objects = SegFrameObjects(img=img)
        try:
            data = zip(result.masks, result.boxes)

        except TypeError:
            seg_results.append(SegFrameObjects.empty(img))
            continue

        for mask, box in data:
            box = box.cpu().numpy()
            # Don't apply mask.cpu
            # https://github.com/ultralytics/ultralytics/issues/8732

            if len(box.cls) > 1:
                recorded_error += "Too much detected. UNEXPECTED "

            clss = int(box.cls[0])

            points = mask.xy[0]
            confidence = box.conf[0]
            box = box.xyxy[0]

            mask = get_mask(points, img.shape)

            seg_obj = SegObject(points, clss, mask, box=box, conf=confidence)
            seg_frame_objects.append(seg_obj)

        timings.append(time.time() - start)

        seg_results.append(seg_frame_objects)

    return seg_results, timings
