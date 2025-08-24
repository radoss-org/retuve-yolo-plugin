# X-ray Validation Results

## Overview

This document presents the validation results for the X-ray hip dysplasia detection models. The system employs two complementary YOLO11 models:
1. **Segmentation Model**: For anatomical landmark segmentation
2. **Pose Estimation Model**: For keypoint detection and angle measurements

## Training Configurations

### Segmentation Model
- **Model**: YOLOv11n-seg (nano segmentation variant)
- **Epochs**: 200
- **Batch Size**: 16
- **Image Size**: 800px
- **Data Augmentation**: Extensive (HSV, mosaic, auto-augment, erasing)

### Pose Estimation Model
- **Model**: YOLOv11n-pose (nano pose variant)
- **Epochs**: 100
- **Batch Size**: 16
- **Image Size**: 800px
- **Data Augmentation**: Moderate (HSV, auto-augment, horizontal flip)

## Segmentation Validation Results

### Overall Performance

![X-ray Segmentation Validation](images/xray_val_seg.png)

### F1 Score Analysis [Training Time] (Segmentation)

![X-ray Segmentation F1 vs Confidence](images/training/xray_f1_seg.png)

## Pose Estimation Validation Results

### Overall Performance

![X-ray Pose Validation](images/xray_val.png)

### F1 Score Analysis [Training Time] (Pose)

![X-ray Pose F1 vs Confidence](images/training/xray_f1.png)

## Clinical Measurement Accuracy

### Angle Measurements

| Metric | Left Side ICC | Left Side CI | Right Side ICC | Right Side CI |
|--------|---------------|--------------|----------------|---------------|
| Acetabular Index | 0.860 | 0.830-0.880 | 0.845 | 0.810-0.870 |
| Wilberg Index | 0.891 | 0.860-0.910 | 0.902 | 0.860-0.930 |

*ICC: Intraclass Correlation Coefficient - measures agreement between automated and manual measurements*

### Classification Performance

The system demonstrates strong diagnostic capabilities:

| Classification Task | F1 Score | Recall | Precision |
|---------------------|----------|--------|-----------|
| Grade 1 vs. Grades 2-4 IHDI | 0.940 | 0.914 | 0.967 |
| Per-Class (All Grades) | 0.593 | 0.570 | 0.637 |

## F1 vs Confidence Threshold Analysis

The F1 vs confidence plots help users select optimal thresholds for different clinical scenarios:
