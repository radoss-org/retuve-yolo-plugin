# Retuve YOLO Segmentation AI Plugin

![tests](https://github.com/radoss-org/retuve-yolo-plugin/actions/workflows/test.yml/badge.svg)

__For more information on Retuve, see https://github.com/radoss-org/retuve__

This codebase has the AI Plugin for Retuve, which uses Radiopedia data from [The Open Hip Dataset](https://github.com/radoss-org/open-hip-dysplasia) to train.

The model weights are strictly under the **combined terms of the CC BY-NC-SA 3.0 license, and the AGPL Licence**. This is because the model is trained on Radiopedia Data, which is under the CC BY-NC-SA 3.0 license, and the [YOLO ultralytics](https://www.ultralytics.com/) codebase is under the AGPL Licence.

This means that you cannot use this codebase for any commercial purposes, you must attribute Radiopedia for the data used to train the model, and you must obide by the terms of the AGPL Licence.

The codes dual licences are in the [LICENSE](LICENSE) file and the [LICENSE2](LICENSE2) file.

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