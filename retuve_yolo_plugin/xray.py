# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import os
import sys
import time

from radstract.data.dicom import convert_dicom_to_images
from retuve.hip_xray.classes import HipLabelsXray, LandmarksXRay
from retuve.keyphrases.config import Config
from retuve.logs import log_timings

from .utils import FILEDIR, shared_yolo_predict, get_yolo_model
from .xray_utils import fit_triangle_to_mask

WEIGHTS = f"{FILEDIR}/weights/v1.0/hip-yolo-xray-seg.pt"


def get_yolo_model_xray(config, weights_path=None, download_if_missing=True):
    return get_yolo_model(config, WEIGHTS, weights_path, download_if_missing)


def yolo_predict_dcm_xray(dcm, keyphrase, model=None):
    config = Config.get_config(keyphrase)

    dicom_images = convert_dicom_to_images(
        dcm,
        crop_coordinates=config.crop_coordinates,
        dicom_type=config.dicom_type,
    )

    return yolo_predict_xray(dicom_images, keyphrase, model, config)


def yolo_predict_xray(images, keyphrase, model=None, stream=False, imgsz=800, conf=0.5):
    config = Config.get_config(keyphrase)

    landmark_results = []

    seg_results, timings = shared_yolo_predict(
        images,
        keyphrase,
        WEIGHTS,
        model,
        config,
        imgsz=imgsz,
        conf=conf,
        stream=stream,
    )

    for seg_result in seg_results:
        for seg_obj in seg_result:
            if seg_obj.empty:
                continue
            seg_obj.cls = HipLabelsXray(seg_obj.cls)

    if len(timings) == 0:
        timings.append(0)
    log_timings(timings, title="Segmentation Processing:")

    timings = []
    for seg_frame_objects in seg_results:
        start = time.time()
        landmarks = LandmarksXRay()

        if len(seg_frame_objects) != 2:
            landmark_results.append(landmarks)
            continue

        tri_1 = seg_frame_objects[0]
        tri_2 = seg_frame_objects[1]

        h_point_l, pel_l_o, pel_l_i, h_point_r, pel_r_o, pel_r_i = fit_triangle_to_mask(
            tri_1.points, tri_2.points
        )

        if h_point_l is None:
            landmark_results.append(landmarks)
            continue

        landmarks.h_point_l, landmarks.pel_l_o, landmarks.pel_l_i = (
            h_point_l,
            pel_l_o,
            pel_l_i,
        )
        landmarks.h_point_r, landmarks.pel_r_o, landmarks.pel_r_i = (
            h_point_r,
            pel_r_o,
            pel_r_i,
        )

        timings.append(time.time() - start)
        landmark_results.append(landmarks)

    if len(timings) == 0:
        timings.append(0)
    log_timings(timings, title="Landmark Processing:")

    return landmark_results, seg_results
