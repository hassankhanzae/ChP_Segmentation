# ChP_Segmentation
Pipeline for Ventricular-Specific Lateral and Third ChP Segmentation using Swin- Transformer U-Net with Convolutional Block Attention Module (CBAM)

### Input/Output-Format
The input NiFTi files should be copied in **input_data** folder path.
1. Single Nifti File(.nii or .nii.gz)
2. Multiple Nifti Files
### Output
Output path folder: **prediction**

project/
├── src/
│   ├── components/
│   ├── pages/
│   └── utils/
├── public/
│   └── images/
├── tests/
└── README.md

├── prediction/  # save the all segmentation result.
│   ├── original_file/ # Save orignal input file.
│   ├── third_ventricle_mask/ # Segmented Third Ventricle Mask.
│   ├── third_chp_mask/ # Segmented Third Ventricle ChP Mask.
│   ├── lat_ventricle_mask/ # Segmented Lateral Ventricle Mask. 
│   ├── lat_chp_mask/ # Segmented Lateral Ventricle ChP Mask.
│   ├── combined_ventricles/ # Combined Lateral and Third Ventricle Masks.
│   ├── combined_chp/ # Combined Lateral and Third ChP Masks. 
