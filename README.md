# PUG - OCR Deep Learning Pipeline

This project involves training and evaluating a deep learning pipeline for optical character recognition (OCR) on handwritten text. The pipeline integrates two main components:

- **CRAFT** for text detection.
- **TrOCR** for text recognition (fine-tuned).

---

## Table of Contents

- [Project Setup](#project-setup)
- [Training Infrastructure](#training-infrastructure)
- [Model Training](#model-training)
- [Sanity Check](#sanity-check)
- [Logging and Checkpointing](#logging-and-checkpointing)
- [Evaluation](#evaluation)

---

## Project Setup

The project environment has been set up with support for GPU acceleration. The implementation can run both locally and on remote platforms. To ensure reproducibility and simplicity, the following dependencies are required:

### Dependencies

- Python 3.8+
- PyTorch
- `transformers` library
- TensorBoard
- `wandb` (optional for experiment tracking)
- CRAFT-specific libraries (e.g., `craft-text-detector`)

### Steps to Set Up

1. Clone the repository and install dependencies:
   
    **Bash:**
    ```bash
    git clone <repository_url>
    cd <repository>
    pip install -r requirements_clean.txt
    ```

    **PowerShell:**
    ```powershell
    git clone <repository_url>
    cd <repository>
    pip install -r requirements.txt
    ```

3. Ensure GPU availability by checking:
    ```python
    import torch
    print(torch.cuda.is_available())
    ```

---

## Training Infrastructure

### Key Features

- **Results Logging**:  
  All training metrics, including loss and evaluation metrics (CER, WER, LER), are logged using TensorBoard.

- **Hyperparameter Selection**:  
  The script allows flexible configuration of hyperparameters such as batch size, learning rate, number of epochs, and maximum target sequence length.

- **Checkpointing**:  
  The model state and optimizer are saved after each epoch to enable resuming training from a checkpoint.

---

## Model Training

The training script consists of the following key components:

### Base Models

- **CRAFT**: Pre-trained text detection model for identifying text regions in images.
- **TrOCR**: Fine-tuned transformer model for recognizing text from detected regions.

### Dataset

- **OCRDataset**: For training and evaluating the TrOCR model on single-line text. The main goal is to load data consisting of cropped image regions along with a CSV file containing the transcriptions of those regions.
- **CraftDataset**: For evaluating the full pipeline with both CRAFT and TrOCR. The main goal is to load data consisting of full images along with a JSON file containing information about the regions and their transcriptions.

### Training Loop

- The TrOCR model is trained to minimize loss while logging the progress.
- A sanity check ensures the training loop is functioning correctly.

---

## Sanity Check

To validate the correctness of the training setup, the model is trained on a single batch to verify:

- The model can overfit to this batch.
- Loss decreases as expected.

Sanity check results are logged to TensorBoard under the `runs/training_logs` directory.

---

## Logging and Checkpointing

### Results Logging

TensorBoard tracks:
- Training loss (`TrOCR/Train_Loss`)
- Validation CER (`TrOCR/Validation_CER`)
- Hyperparameters and metrics

Sample log directory: `runs/training_logs/`.

### Hyperparameter Logging

Key hyperparameters (e.g., batch size, learning rate, number of epochs) and metrics (training loss, validation CER) are logged to TensorBoard using the `add_hparams` API.

### Checkpointing

The model, optimizer, and training state are saved after each epoch to the `checkpoints/` directory.  
Example checkpoint file: `checkpoints/checkpoint_<id>.pth`.

---

## Evaluation

The evaluation pipeline combines the following:

- **CRAFT** for detecting text regions in full-page images.
- **TrOCR** for recognizing text within each region.

### Metrics Computed During Evaluation

- **CER (Character Error Rate)**: Measures the average character-level differences between predictions and ground truth.
- **WER (Word Error Rate)**: Measures word-level accuracy.
- **LER (Label Error Rate)**: Checks exact matches between predicted and ground truth text.

The evaluation metrics are logged to TensorBoard under `Evaluation`.
