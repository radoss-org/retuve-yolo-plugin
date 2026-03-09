# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import os
import time

import cv2
import numpy as np
import requests
import torch
from dotenv import load_dotenv
from retuve.classes.seg import SegFrameObjects, SegObject
from retuve.keyphrases.config import Config
from retuve.logs import ulogger

FILEDIR = os.path.abspath(f"{os.path.dirname(os.path.abspath(__file__))}/")
WEIGHTS_PATH = f"{FILEDIR}/weights/v1.0/"


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
    reduce_memory=False,
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

    all_results = []

    for image in images:

        results = model.predict(
            [image],
            imgsz=imgsz,
            conf=conf,
            verbose=False,
            stream=stream,
            retina_masks=True,
            device=device,
        )

        if len(all_results) == 0:
            all_results.extend(results)
            continue

        # If reduce_memory is True, only keep detections with at least one box
        if reduce_memory:
            filtered = [
                r for r in results if len(r.boxes) > 0
            ]  # Each result has .boxes tensor
            all_results.extend(filtered)
        else:
            all_results.extend(results)

        del results

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

    if model == None:
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

    print(results[0].keypoints)

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

        frame_landmarks = [
            list(item)
            for sublist in frame_landmarks
            if sublist is not None
            for item in sublist
            if item is not None
        ]

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

    print(landmark_results)

    return landmark_results, landmark_names


import cv2
import numpy as np


def remove_bridges_keep_main(mask, thr=0.5):
    """Remove 1px bridges and keep largest component. mask: (H,W) float or binary"""
    # Convert to single channel if needed
    if len(mask.shape) == 3:
        mask = mask[:, :, 0]  # Take first channel

    # Binarize
    if mask.dtype != np.uint8:
        mask = (mask >= thr).astype(np.uint8) * 255

    # Remove 1px bridges with opening
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Keep largest component
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        opened, connectivity=8
    )
    if num_labels <= 1:
        result = opened
    else:
        # Find largest (skip background at index 0)
        areas = stats[1:, cv2.CC_STAT_AREA]
        largest_idx = 1 + np.argmax(areas)
        result = np.zeros_like(opened)
        result[labels == largest_idx] = 255

    # Convert back to 3-channel RGB
    result = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)
    return result


def shared_yolo_predict(
    images,
    keyphrase,
    default_weights,
    model=None,
    config=None,
    imgsz=512,
    conf=0.8,
    stream=False,
    reduce_memory=False,
):

    if not config:
        config = Config.get_config(keyphrase)

    seg_results = []

    if model == None:
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
                reduce_memory=reduce_memory,
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

            # TODO: Make this functional without lowering ICC
            mask = remove_bridges_keep_main(mask)

            seg_obj = SegObject(points, clss, mask, box=box, conf=confidence)
            seg_frame_objects.append(seg_obj)

        timings.append(time.time() - start)

        seg_results.append(seg_frame_objects)

    return seg_results, timings


def _download_weights(api_url: str) -> bool:
    """Download weights from GitHub repository or public URL."""
    ext = api_url.split(".")[-1]
    file_path = WEIGHTS_PATH + f"hip-yolo-us.{ext}"
    os.makedirs(WEIGHTS_PATH, exist_ok=True)

    try:
        print(f"Downloading {api_url}...")
        download_url = _convert_github_url(api_url)

        if "github.com" in download_url and "/raw/" in download_url:
            headers = {"User-Agent": "Python-GitHub-File-Downloader"}
        elif "api.github.com" in download_url:
            load_dotenv()
            token = os.getenv("GITHUB_PAT")
            if not token:
                print(
                    "Warning: GITHUB_PAT not found. This may fail for private repositories..."
                )
                headers = {"User-Agent": "Python-GitHub-File-Downloader"}
            else:
                headers = {
                    "Authorization": f"token {token}",
                    "Accept": "application/vnd.github.v3.raw",
                    "User-Agent": "Python-GitHub-File-Downloader",
                }
        else:
            headers = {"User-Agent": "Python-GitHub-File-Downloader"}

        response = requests.get(download_url, headers=headers, stream=True)
        response.raise_for_status()

        with open(file_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        file_size = os.path.getsize(file_path)
        print(f"Successfully downloaded {file_path} ({file_size:,} bytes)")
        return True

    except Exception as e:
        print(f"Download failed: {e}")
        if os.path.exists(file_path):
            os.remove(file_path)
        return False


def _convert_github_url(url: str) -> str:
    """Convert various GitHub URL formats to downloadable URLs."""
    if "github.com" not in url:
        return url

    if "/blob/" in url:
        parts = url.replace("https://github.com/", "").split("/")
        if len(parts) >= 4:
            owner, repo = parts[0], parts[1]
            branch_and_path = "/".join(parts[3:])
            path_parts = branch_and_path.split("/")
            branch, file_path = path_parts[0], "/".join(path_parts[1:])

            load_dotenv()
            token = os.getenv("GITHUB_PAT")

            if token:
                return (
                    f"https://api.github.com/repos/{owner}/{repo}/contents/{file_path}"
                )
            else:
                return f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{file_path}"

    if "raw.githubusercontent.com" in url or "/raw/" in url or "api.github.com" in url:
        return url

    return url


def get_yolo_model(
    config, default_weights, weights_path=None, download_if_missing=True
):
    """Load YOLO model for ultrasound with automatic weight downloading."""
    from ultralytics import YOLO

    target_weights = weights_path if weights_path is not None else default_weights

    # If target_weights is a URL, handle custom weight downloading
    if target_weights.startswith("http") and download_if_missing:
        print(f"Custom weights URL provided: {target_weights}")
        if not _download_weights(target_weights):
            raise FileNotFoundError(f"Failed to download weights from {target_weights}")
        # After download, use the local file path
        ext = target_weights.split(".")[-1]
        target_weights = WEIGHTS_PATH + f"hip-yolo-us.{ext}"
    elif not os.path.exists(target_weights) and download_if_missing:
        print(f"Weights file {target_weights} not found. Downloading...")
        if not _download_weights(target_weights):
            raise FileNotFoundError(f"Failed to download weights to {target_weights}")

    if not os.path.exists(target_weights):
        raise FileNotFoundError(f"Weights file {target_weights} does not exist")

    print(f"Loading YOLO model from: {target_weights}")
    model = YOLO(target_weights, task="segment")

    if "onnx" not in target_weights.lower():
        model.to(config.device)

    return model
