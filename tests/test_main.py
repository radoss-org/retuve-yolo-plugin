# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import pydicom
from PIL import Image
from retuve.defaults.hip_configs import default_US, default_xray
from retuve.funcs import analyse_hip_3DUS, analyse_hip_xray_2D
from retuve.testdata import Cases, download_case

from retuve_yolo_plugin.ultrasound import (
    get_yolo_model_us,
    yolo_predict_dcm_us,
)
from retuve_yolo_plugin.xray import yolo_predict_xray
from retuve_yolo_plugin.xray_pose import (
    yolo_predict_xray as yolo_predict_xray_pose,
)


def test_ultrasound():
    dcm_file = download_case(Cases.ULTRASOUND_DICOM)[0]

    default_US.device = "cpu"

    dcm = pydicom.dcmread(dcm_file)

    hip_datas, video, *_ = analyse_hip_3DUS(
        dcm,
        keyphrase=default_US,
        modes_func=yolo_predict_dcm_us,
        modes_func_kwargs_dict={},
    )

    assert hip_datas.grafs_hip.metrics[0].value > 0


def test_ultrasound_call_model_first():
    dcm_file = download_case(Cases.ULTRASOUND_DICOM)[0]

    default_US.device = "cpu"

    dcm = pydicom.dcmread(dcm_file)

    model = get_yolo_model_us(default_US)

    hip_datas, video, *_ = analyse_hip_3DUS(
        dcm,
        keyphrase=default_US,
        modes_func=yolo_predict_dcm_us,
        modes_func_kwargs_dict={"model": model},
    )

    assert hip_datas.grafs_hip.metrics[0].value > 0


def test_ultrasound_custom():
    dcm_file = download_case(Cases.ULTRASOUND_DICOM)[0]

    default_US.device = "cpu"

    dcm = pydicom.dcmread(dcm_file)

    model = get_yolo_model_us(
        default_US,
        weights_path="https://github.com/radoss-org/retuve-yolo-plugin/raw/refs/heads/main/retuve_yolo_plugin/weights/v1.0/hip-yolo-us.pt",
    )

    hip_datas, video, *_ = analyse_hip_3DUS(
        dcm,
        keyphrase=default_US,
        modes_func=yolo_predict_dcm_us,
        modes_func_kwargs_dict={"model": model, "imgsz": 512, "conf": 0.6},
    )

    assert hip_datas.grafs_hip.metrics[0].value > 0

    video.write_videofile("test_ultrasound_custom.mp4")


def test_xray():

    jpg_file = download_case(Cases.XRAY_JPG)[0]

    default_xray.device = "cpu"

    img = Image.open(jpg_file)

    hip, *_ = analyse_hip_xray_2D(
        img,
        keyphrase=default_xray,
        modes_func=yolo_predict_xray,
        modes_func_kwargs_dict={},
    )

    assert hip.metrics[0].value > 0


def test_xray_pose():

    jpg_file = download_case(Cases.XRAY_JPG)[0]

    default_xray.device = "cpu"

    img = Image.open(jpg_file)

    hip, *_ = analyse_hip_xray_2D(
        img,
        keyphrase=default_xray,
        modes_func=yolo_predict_xray_pose,
        modes_func_kwargs_dict={},
    )

    assert hip.metrics[0].value > 0


if __name__ == "__main__":
    test_ultrasound_call_model_first()
