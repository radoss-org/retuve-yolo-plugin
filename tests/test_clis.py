# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import json

from retuve.batch import run_single
from retuve.defaults.hip_configs import default_US
from retuve.keyphrases.enums import HipMode
from retuve.testdata import Cases, download_case

from retuve_yolo_plugin.ultrasound import (
    get_yolo_model_us,
    yolo_predict_dcm_us,
)

default_US.batch.hip_mode = HipMode.US3D
default_US.batch.mode_func = yolo_predict_dcm_us
default_US.device = "cpu"
default_US.batch.mode_func_args = {"model": get_yolo_model_us(default_US)}
default_US.api.api_token = "password"


def test_single():
    dcm_file = download_case(Cases.ULTRASOUND_DICOM)[0]

    run_single(
        default_US,
        dcm_file,
    )
