# Retuve YOLO Segmentation AI Plugin

![tests](https://github.com/radoss-org/retuve-yolo-plugin/actions/workflows/test.yml/badge.svg)

__For more information on Retuve, see https://github.com/radoss-org/retuve__

This codebase has the AI Plugin for Retuve, which uses Radiopedia data from [The Open Hip Dataset](https://github.com/radoss-org/open-hip-dysplasia) to train.

The model weights are strictly under the **combined terms of the CC BY-NC-SA 3.0 license, and the AGPL Licence**. This is because the model is trained on Radiopedia Data, which is under the CC BY-NC-SA 3.0 license, and the [YOLO ultralytics](https://www.ultralytics.com/) codebase is under the AGPL Licence.

This means that you cannot use this codebase for any commercial purposes, you must attribute Radiopedia for the data used to train the model, and you must obide by the terms of the AGPL Licence.

The codes dual licences are in the [LICENSE](LICENSE) file and the [LICENSE2](LICENSE2) file.


## Initial Results

![Combined](docs/combined_all_plots.png)

### Detailed Performance Metrics

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

## UPDATE - X-Ray Version 2 - Landmark Detection

We have added a new version of the x-ray model, which is trained on the MTDDH dataset (https://www.nature.com/articles/s41597-025-05146-x). We suggest reading this datasets description as it is very diverse and of mixed quality.

This model is available in the `retuve_yolo_plugin.xray_v2` module.

The model is trained on the MTDDH dataset, which is a dataset of 1000's of x-rays of the hip.

We show initial results with a 50/50 train/val split with an F1 Score of `0.951` for seperating IHDI Grade 1 from 2, 3 and 4. We also show a mean error in the acetabular angle of `3` degrees, and median of `2.4` degrees.

![MTDDH](docs/combined_all_plots.png)

It is expected that with a different non-pose model, better results can be achieved. Although the ICC is lower than v1, the F1-Score for separating IHDI Grade 1 from 2, 3 and 4 is higher - therefore this model should be preferred for grading IHDI.

## Installation

To install the plugin, you can use the following command:

```bash
pip install git+https://github.com/radoss-org/retuve-yolo-plugin.git
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