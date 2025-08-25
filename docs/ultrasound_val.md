# Ultrasound Validation Results

## Overview

This document presents the validation results for the ultrasound hip dysplasia detection model. The model uses YOLO11n-seg architecture for semantic segmentation of anatomical landmarks in ultrasound images, enabling automated measurement of alpha angles and coverage percentages.

## Training Configuration

The ultrasound model was trained with the following key parameters:
- **Model**: YOLOv11n-seg (nano segmentation variant)
- **Epochs**: 100
- **Batch Size**: 4
- **Image Size**: 500px
- **Learning Rate**: 0.01
- **Data Augmentation**: Minimal (HSV variations, small rotation/translation)

## Validation Performance

### Segmentation Accuracy

![Ultrasound Validation Results](images/ultrasound_val.png)

The validation results show the model's performance across different metrics:
- **Precision**: Measures the accuracy of positive predictions
- **Recall**: Measures the model's ability to find all positive instances
- **mAP50**: Mean Average Precision at IoU threshold of 0.5
- **mAP50-95**: Mean Average Precision averaged across IoU thresholds from 0.5 to 0.95

### Clinical Measurement Accuracy

| Metric | ICC Score | 95% Confidence Interval |
|--------|-----------|-------------------------|
| Alpha Angle | 0.86 | -0.07 to 0.81 |
| Coverage | 0.92 | 0.84 to 0.96 |

*ICC: Intraclass Correlation Coefficient - measures agreement between automated and manual measurements*

### F1 Score vs Confidence Threshold [Training Time]

![Ultrasound F1 vs Confidence](images/training/ultrasound_f1.png)

This plot shows how the F1 score varies with different confidence thresholds. The F1 score balances precision and recall, helping users choose an optimal confidence threshold for inference:

### Limitations

- Training data size is limited compared to X-ray datasets