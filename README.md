# ChP_Segmentation
Pipeline for Ventricular-Specific Lateral and Third ChP Segmentation using Swin- Transformer U-Net with Convolutional Block Attention Module (CBAM)

### Input/Output-Format
The input NiFTi files should be copied in **input_data** folder path.
1. Single Nifti File(.nii or .nii.gz)
2. Multiple Nifti Files
### Output
Output path folder: **prediction**
prediction/
├── original_file/                 # Saves original input files
│
├── third_ventricle_mask/         # Segmented third ventricle mask
├── third_chp_mask/               # Segmented third ventricle choroid plexus (ChP) mask
│
├── lat_ventricle_mask/           # Segmented lateral ventricle mask
├── lat_chp_mask/                 # Segmented lateral ventricle choroid plexus (ChP) mask
│
├── combined_ventricles/          # Combined lateral + third ventricle masks
└── combined_chp/                 # Combined lateral + third ChP masks
