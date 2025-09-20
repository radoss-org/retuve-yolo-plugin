# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import os
from pathlib import Path


from radstract.data.dicom import convert_dicom_to_images
from retuve.hip_us.classes.enums import HipLabelsUS
from retuve.keyphrases.config import Config
from retuve.logs import log_timings

from .utils import get_yolo_model

from .utils import FILEDIR, shared_yolo_predict

WEIGHTS = f"{FILEDIR}/weights/v1.0/hip-yolo-us.onnx"


def get_yolo_model_us(config, weights_path=None, download_if_missing=True):
    return get_yolo_model(config, WEIGHTS, weights_path, download_if_missing)


def yolo_predict_dcm_us(
    dcm, keyphrase, model=None, custom_weights=None, imgsz=512, conf=0.6
):
    """Predict on DICOM data for ultrasound."""
    config = Config.get_config(keyphrase)
    dicom_images = convert_dicom_to_images(
        dcm,
        crop_coordinates=config.crop_coordinates,
        dicom_type=config.dicom_type,
    )
    return yolo_predict_us(dicom_images, keyphrase, model, custom_weights, imgsz, conf)


def yolo_predict_us(
    images, keyphrase, model=None, custom_weights=None, imgsz=512, conf=0.6
):
    """Predict on images for ultrasound."""
    config = Config.get_config(keyphrase)

    if model is None:
        model = get_yolo_model_us(config, custom_weights)

    weights_path = custom_weights if custom_weights is not None else WEIGHTS

    seg_results, timings = shared_yolo_predict(
        images, keyphrase, weights_path, model, config, conf=conf, imgsz=imgsz
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
    print(f"Warning: {WEIGHTS} does not exist. Will attempt to download when needed.")
