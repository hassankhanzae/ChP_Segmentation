# ChP_Segmentation
Pipeline for Ventricular-Specific Lateral and Third ChP Segmentation using Swin- Transformer U-Net with Convolutional Block Attention Module (CBAM)

### Input/Output-Format
The input NiFTi files should be copied in **input_data** folder path.
1. Single Nifti File(.nii or .nii.gz)
2. Multiple Nifti Files
### Output Prediction
Output path folder: **prediction**
```text
prediction/
├── original_file/
├── third_ventricle_mask/
├── third_chp_mask/
├── lat_ventricle_mask/
├── lat_chp_mask/
├── combined_ventricles/
└── combined_chp/
