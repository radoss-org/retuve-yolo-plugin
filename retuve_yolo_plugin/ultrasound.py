# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import os
from pathlib import Path

import requests
from dotenv import load_dotenv
from radstract.data.dicom import convert_dicom_to_images
from retuve.hip_us.classes.enums import HipLabelsUS
from retuve.keyphrases.config import Config
from retuve.logs import log_timings

from .utils import FILEDIR, get_yolo_model, shared_yolo_predict

WEIGHTS = f"{FILEDIR}/weights/v1.0/hip-yolo-us.onnx"
WEIGHTS_PATH = f"{FILEDIR}/weights/v1.0/"


# check weights file exists
if not os.path.exists(WEIGHTS):
    raise FileNotFoundError(f"Weight file not found: {WEIGHTS}")


def get_yolo_model_us(config, weights_path=None, download_if_missing=True):
    """Load YOLO model for ultrasound with automatic weight downloading."""
    model = get_yolo_model(
        config,
        WEIGHTS,
        weights_path=weights_path,
        download_if_missing=download_if_missing,
    )

    return model


def yolo_predict_dcm_us(
    dcm,
    keyphrase,
    model=None,
    custom_weights=None,
    imgsz=512,
    conf=0.6,
    reduce_memory=False,
):
    """Predict on DICOM data for ultrasound."""
    config = Config.get_config(keyphrase)
    dicom_images = convert_dicom_to_images(
        dcm,
        dicom_type=config.dicom_type,
    )

    return yolo_predict_us(
        dicom_images,
        keyphrase,
        model,
        custom_weights,
        imgsz,
        conf,
        reduce_memory,
    )


def yolo_predict_us(
    images,
    keyphrase,
    model=None,
    custom_weights=None,
    imgsz=512,
    conf=0.6,
    reduce_memory=False,
):
    """Predict on images for ultrasound."""
    config = Config.get_config(keyphrase)

    # Use the cached model if no specific model instance is provided
    if model == None:
        model = get_yolo_model_us(config, custom_weights)

    weights_path = custom_weights if custom_weights is not None else WEIGHTS

    seg_results, timings = shared_yolo_predict(
        images,
        keyphrase,
        weights_path,
        model,
        config,
        conf=conf,
        imgsz=imgsz,
        reduce_memory=reduce_memory,
    )

    for seg_result in seg_results:
        for seg_obj in seg_result:
            if seg_obj.empty:
                continue
            seg_obj.cls = HipLabelsUS(seg_obj.cls)

    log_timings(timings, title="Segmentation Processing:")
    return seg_results


# Check weights exist on import (optional - can be removed if too strict)
if not os.path.exists(WEIGHTS):
    print(
        f"Warning: {WEIGHTS} does not exist. " "Will attempt to download when needed."
    )
