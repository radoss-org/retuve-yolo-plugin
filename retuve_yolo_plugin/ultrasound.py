# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import os
import sys

from radstract.data.dicom import convert_dicom_to_images
from retuve.hip_us.classes.enums import HipLabelsUS
from retuve.keyphrases.config import Config
from retuve.logs import log_timings

from .utils import FILEDIR, shared_yolo_predict

WEIGHTS = f"{FILEDIR}/weights/hip-yolo-us.pt"
# check weights file exists
if not os.path.exists(WEIGHTS):
    sys.exit(f"Error: {WEIGHTS} does not exist")


def get_yolo_model_us(config):
    from ultralytics import YOLO

    model = YOLO(WEIGHTS)
    model.to(config.device)

    return model


def yolo_predict_dcm_us(dcm, keyphrase, model=None):
    config = Config.get_config(keyphrase)

    dicom_images = convert_dicom_to_images(
        dcm,
        crop_coordinates=config.crop_coordinates,
        dicom_type=config.dicom_type,
    )

    return yolo_predict_us(dicom_images, keyphrase, model)


def yolo_predict_us(images, keyphrase, model=None):
    config = Config.get_config(keyphrase)

    seg_results, timings = shared_yolo_predict(
        images, keyphrase, WEIGHTS, model, config, conf=0.5
    )

    for seg_result in seg_results:
        for seg_obj in seg_result:
            if seg_obj.empty:
                continue
            seg_obj.cls = HipLabelsUS(seg_obj.cls)

    log_timings(timings, title="Segmentation Processing:")

    return seg_results
