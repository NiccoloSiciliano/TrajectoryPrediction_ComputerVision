# White Blood Cells Classification and Segmentation

## Project Overview
This project focuses on the **classification** and **segmentation** of **White Blood Cells (WBCs)** using deep learning models. The goal is to first segment WBCs from images and then classify them based on their type using different CNN-based architectures.

### Models:
- **Segmentation**:  
  1. **ReducedUnet**: A lightweight version of the well-known U-Net architecture for WBC segmentation.
  2. **DoubleDecArch**: A custom CNN model with an encoder-decoder architecture and a double decoder for enhanced segmentation.
  
- **Classification**:  
  1. **Arch1**: A simple CNN architecture for classifying WBCs from raw cell images.
  2. **Combined Image Classifier**: A variant of `Arch1` that uses combined input from the segmented mask and the original image.

## Table of Contents
- [Architecture Details](#architecture-details)
  - [Segmentation](#segmentation)
  - [Classification](#classification)
- [Usage](#usage)
  - [Training](#training)
  - [Testing](#testing)
## Architecture Details

### Segmentation
1. **ReducedUnet**:
   - A reduced version of the U-Net architecture designed to be computationally lighter while maintaining effective segmentation performance.
   - **Architecture**: 
     - Encoder and decoder paths.
     - Skip connections between the encoder and decoder layers for efficient segmentation.
     - Output: Segmentation masks for WBCs.

2. **DoubleDecArch**:
   - A CNN model based on an encoder-decoder architecture but with a **double decoder** to improve segmentation results.
   - **Architecture**: 
     - The encoder extracts features from the input image.
     - Two separate decoder branches process the features to refine the segmentation.
     - Output: Segmentation masks with enhanced detail through dual decoders.

### Classification
1. **Arch1**:
   - A simple CNN classifier that takes WBC images as input and classifies them into their respective categories.
   - **Input**: Raw WBC image.
   - **Output**: Cell type classification.

2. **Combined Image Classifier**:
   - This model is an alternative to `Arch1`, with the same architecture but a different input.
   - **Input**: A combined image that merges the segmented mask and the original image, providing additional features for classification.
   - **Output**: Cell type classification based on combined input.


## Usage

### Training
- **Train Segmentation Model**:
  ```bash
  python train_seg_main.py path_dtset
  ```
  Replace `path_dtset` with the path to your segmentation dataset.

- **Train Classification Model**:
  ```bash
  python train_classification_main.py path_dtset
  ```
  Replace `path_dtset` with the path to your classification dataset.

### Testing
- **Test Segmentation on Single Image**:
  ```bash
  python test_seg_image_main.py image_path mask_path model_path
  ```
  Replace:
  - `image_path`: Path to the input image.
  - `mask_path`: Path to the ground truth mask.
  - `model_path`: Path to the trained segmentation model.

- **Test Segmentation Model on Dataset**:
  ```bash
  python test_seg_model_main.py ALL test_path model_path
  ```
  Replace:
  - `test_path`: Path to the test dataset.
  - `model_path`: Path to the trained segmentation model.

- **Test Two-Model Segmentation on Single Image**:
  ```bash
  python test_twomodel_image.py ALL image_path mask_path wall_model_path nucleus_model_path
  ```
  Replace:
  - `image_path`: Path to the input image.
  - `mask_path`: Path to the ground truth mask.
  - `wall_model_path`: Path to the wall segmentation model.
  - `nucleus_model_path`: Path to the nucleus segmentation model.

- **Test Two-Model Segmentation on Dataset**:
  ```bash
  python test_twomodel_main.py ALL wall_model_path nucleus_model_path test_path
  ```
  Replace:
  - `wall_model_path`: Path to the wall segmentation model.
  - `nucleus_model_path`: Path to the nucleus segmentation model.
  - `test_path`: Path to the test dataset.

- **Test Classification on Single Image**:
  ```bash
  python test_classification_image_main.py var.py image_path model_path
  ```
  Replace:
  - `image_path`: Path to the input image.
  - `model_path`: Path to the trained classification model.

- **Test Classification on Dataset**:
  ```bash
  python test_classification_model_main.py model_path dataset_path
  ```
  Replace:
  - `dataset_path`: Path to the dataset.
  - `model_path`: Path to the trained classification model.
