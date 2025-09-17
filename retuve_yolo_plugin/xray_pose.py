# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import os
import sys
import time

import numpy as np
from radstract.data.dicom import convert_dicom_to_images
from retuve.classes.seg import SegFrameObjects, SegObject
from retuve.hip_xray.classes import HipLabelsXray, LandmarksXRay
from retuve.keyphrases.config import Config
from retuve.logs import log_timings

from .utils import FILEDIR, yolo_predict_pose
from .xray_utils import fit_triangle_to_mask

WEIGHTS = f"{FILEDIR}/weights/v1.0/hip-yolo-xray-pose.pt"
# check weights file exists
if not os.path.exists(WEIGHTS):
    raise FileNotFoundError(f"Weight file not found: {WEIGHTS}")


def get_yolo_model_xray(config):
    from ultralytics import YOLO

    model = YOLO(WEIGHTS)
    model.to(config.device)

    return model


def yolo_predict_dcm_xray(dcm, keyphrase, model=None):
    config = Config.get_config(keyphrase)

    dicom_images = convert_dicom_to_images(
        dcm,
        crop_coordinates=config.crop_coordinates,
        dicom_type=config.dicom_type,
    )

    return yolo_predict_xray(dicom_images, keyphrase, model, config)


def yolo_predict_xray(images, keyphrase, model=None, stream=False):
    config = Config.get_config(keyphrase)

    landmark_results = []
    seg_results = []

    landmarks_list, landmark_names = yolo_predict_pose(
        images,
        keyphrase,
        WEIGHTS,
        model,
        config,
        imgsz=800,
        conf=0.5,
        stream=stream,
    )

    for i, landmarks in enumerate(landmarks_list):
        landmarks_obj = LandmarksXRay()

        (
            landmarks_obj.pel_l_o,
            landmarks_obj.pel_l_i,
            landmarks_obj.fem_l,
            landmarks_obj.h_point_l,
            landmarks_obj.pel_r_o,
            landmarks_obj.pel_r_i,
            landmarks_obj.fem_r,
            landmarks_obj.h_point_r,
        ) = landmarks
        landmark_results.append(landmarks_obj)

        seg_results.append(
            SegFrameObjects(
                img=np.array(images[i]),
                seg_objects=None,
            )
        )

    return landmark_results, seg_results
