# ChP_Segmentation
Pipeline for Ventricular-Specific Lateral and Third ChP Segmentation using Swin- Transformer U-Net with Convolutional Block Attention Module (CBAM)

## Python Code Envoirment Setup

Install the required packages listed in **requirements.txt** file. The code run smoothly with Python 3.8 version.
Clone the repository and navigate to repository.

```bash
git clone https://github.com/hassankhanzae/ChP_Segmentation.git
cd ChP_Segmentation
python pipeline.py
```
### Download Train Model Weights
Download the weights for the trained model from [Google drive]([https://example.com](https://drive.google.com/drive/folders/18zV-amDe2JP_jiHTXkAdm0k5JZgIXFPK?usp=share_link) and copy the downloaded weights in the **weights** folder.

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
```

### Required Pipeline Structure
```text
project/
│
├── pipeline.py
├── weights/
│   ├── lventricle_weight.pth
│   ├── lchp_weight.pth
│   ├── 3ventricle_weight.pth
│   └── 3chp_best_weight.pth
│
├── input_data/
│   ├── case1.nii.gz
│   ├── case2.nii.gz
│
└── prediction/   (#Predicted Segmentations Auto-Created Folder)
