# ChP_Segmentation
Pipeline for Ventricular-Specific Lateral and Third ChP Segmentation using Swin- Transformer U-Net with Convolutional Block Attention Module (CBAM)

## Python Code Envoirment Setup

Install the required packages listed in **requirements.txt** file. The code run smoothly with Python 3.8 version and pytorch 12.0 version.
Clone the repository and navigate to repository.

```bash
git clone https://github.com/hassankhanzae/ChP_Segmentation.git
cd ChP_Segmentation

### Input/Output-Format
The input NiFTi files should be copied in **input_data** folder path.
1. Single Nifti File(.nii or .nii.gz)
2. Multiple Nifti Files
### Output Prediction
Output path folder: **prediction**
```text
prediction/ 
├── original_file/ # Saves original input files
├── third_ventricle_mask/ # Segmented third ventricle mask
├── third_chp_mask/ # Segmented third ventricle ChP mask
├── lat_ventricle_mask/ # Segmented lateral ventricle mask
├── lat_chp_mask/ # Segmented lateral ventricle ChP mask
├── combined_ventricles/ # Combined lateral + third ventricle masks
└── combined_chp/ # Combined lateral + third ChP masks
