# Retuve YOLO Segmentation AI Plugin

![tests](https://github.com/radoss-org/retuve-yolo-plugin/actions/workflows/test.yml/badge.svg)

__For more information on Retuve, see https://github.com/radoss-org/retuve__

This codebase has the AI Plugin for Retuve, which uses Radiopedia data from [The Open Hip Dataset](https://github.com/radoss-org/open-hip-dysplasia) to train.

The model weights are strictly under the **combined terms of the CC BY-NC-SA 3.0 license, and the AGPL Licence**. This is because the model is trained on Radiopedia Data, which is under the CC BY-NC-SA 3.0 license, and the [YOLO ultralytics](https://www.ultralytics.com/) codebase is under the AGPL Licence.

This means that you cannot use this codebase for any commercial purposes, you must attribute Radiopedia for the data used to train the model, and you must obide by the terms of the AGPL Licence.

The codes dual licences are in the [LICENSE](LICENSE) file and the [LICENSE2](LICENSE2) file.


## Initial Results

For a more detailed look at the validation results, see:
- [Ultrasound Validation Results](docs/ultrasound_val.md)
- [X-ray Validation Results](docs/xray_val.md)

### Detailed Performance Metrics

#### Ultrasound Validation Results (Limited Cases)

| Metric | Results |
|--------|-----------|
| Alpha Angle ICC | 0.86 (95% CI -0.07-0.81) |
| Coverage ICC | 0.92 (95% CI 0.84-0.96) |

*Ultrasound validation results for alpha angle and coverage measurements*


#### X-ray Validation Results

| Metric | Left Side | Right Side |
|--------|-----------|------------|
| ICC Acetabular Index | 0.860 (95% CI 0.830-0.880) | 0.845 (95% CI 0.810-0.870) |
| ICC Wilberg Index | 0.891 (95% CI 0.860-0.910) | 0.902 (95% CI 0.860-0.930) |

*X-ray validation results for acetabular index and Wilberg angle measurements*

#### Classification Performance

For X-ray DDH classification, Retuve demonstrated strong performance in distinguishing Grade 1 IHDI from Grades 2, 3, and 4:

| Classification Task | F1 Score | Recall | Precision |
|---------------------|----------|--------|-----------|
| Grade 1 vs. Grades 2-4 IHDI | 0.940 | 0.914 | 0.967 |
| Per-Class (All Grades) | 0.593 | 0.570 | 0.637 |

*Classification performance for DDH grading on X-ray images*

**Note:** For the Grade 1 vs. Grades 2-4 classification analysis, cases where Retuve returned a result of "0" were logically classified as IHDI Grade 2 or higher, as a "0" result represents a Retuve processing error and indicates the system's inability to confidently classify the case as normal (Grade 1).

### F1 vs Confidence Score Analysis

The F1 vs confidence score plots shown in the detailed validation results help users select optimal confidence thresholds for their specific use case. The F1 score balances precision (accuracy of positive predictions) and recall (ability to find all positive cases), providing a single metric to evaluate model performance across different confidence levels.

**Why this matters for inference:**
- **Screening applications**: Lower thresholds (0.2-0.4) maximize recall to avoid missing cases
- **Diagnostic applications**: Higher thresholds (0.5-0.7) maximize precision for confident diagnoses
- **Research applications**: Mid-range thresholds (0.3-0.5) provide balanced performance

Users should select confidence thresholds based on their clinical priorities: whether it's more important to catch all potential cases (high recall) or to minimize false positives (high precision).

### Training Parameters Summary

| Parameter | Ultrasound | X-ray Segmentation | X-ray Pose |
|-----------|------------|-------------------|------------|
| **Model Architecture** | YOLOv11n-seg | YOLOv11n-seg | YOLOv11n-pose |
| **Task** | Segmentation | Segmentation | Pose Estimation |
| **Epochs** | 100 | 200 | 100 |
| **Batch Size** | 4 | 16 | 16 |
| **Image Size** | 500px | 800px | 800px |
| **Learning Rate** | 0.01 | 0.01 | 0.01 |
| **Data Augmentation** | Minimal | Extensive | Moderate |
| **Key Augmentations** | HSV (0.1), Rotation (5°) | HSV (0.7), Mosaic, AutoAug | HSV (0.7), AutoAug, Flip |
| **Device** | cuda:4 | cuda:1 | Auto |

*Training configurations optimized for each modality's specific requirements and dataset characteristics*

## Installation

### From PyPI

To install the plugin from PyPI:

```bash
uv pip install retuve-yolo-plugin
```

### From Source

To install the plugin from source:

```bash
pip install git+https://github.com/radoss-org/retuve-yolo-plugin.git
```

### Development Installation

For development, clone the repository and install with uv:

```bash
git clone https://github.com/radoss-org/retuve-yolo-plugin.git
cd retuve-yolo-plugin
uv sync --dev
```

### Running Tests and Formatting

```bash
# Run tests
uv run pytest -vv -n 4 ./tests

# Format code
uv run black .

# Check formatting
uv run black --check .
```

## Example Usage

Please see https://github.com/radoss-org/retuve/tree/main/examples for more examples. This is purely meant to illustrate how to use the plugin.

```python
import pydicom
from retuve.defaults.hip_configs import default_US
from retuve.funcs import analyse_hip_3DUS
from retuve.testdata import Cases, download_case

from retuve_yolo_plugin.ultrasound import yolo_predict_dcm_us

# Get an example case
dcm_file = download_case(Cases.ULTRASOUND_DICOM)[0]

default_US.device = "cpu"

dcm = pydicom.dcmread(dcm_file)

hip_datas, *_ = analyse_hip_3DUS(
    dcm,
    keyphrase=default_US,
    modes_func=yolo_predict_dcm_us,
    modes_func_kwargs_dict={},
)

print(hip_datas)
```

## Attribution

We give full attribution to the authors that made this effort possible on Radiopedia. The list of these authors can be found [here](https://github.com/radoss-org/open-hip-dysplasia/tree/main/radiopedia_ultrasound_2d#attribution).

## License

The codes dual licences are in the [LICENSE](LICENSE) file and the [LICENSE2](LICENSE2) file.

If you are interested in a less-restritive licence, the first step is to [contact Radiopedia](https://radiopaedia.org/licence?lang=gb#obtaining_a_license) for a special licence to use all the data this model is trained on. That list can be found [here](https://github.com/radoss-org/open-hip-dysplasia/tree/main/radiopedia_ultrasound_2d#attribution).

The 2nd step is to contact YOLO Ultralytics for a commercial licence for their codebase. That process is described [here](https://github.com/ultralytics/ultralytics?tab=readme-ov-file#-license).

RadOSS will then consider providing you a commercial licence for this plugin at no charge. Please contact us at info@radoss.org when you have obtained the licence from Radiopedia and YOLO Ultralytics.