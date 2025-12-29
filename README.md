 # NeuroSegment-3D: Volumetric Brain Tumor Segmentation


[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Live%20Demo-blue)](https://huggingface.co/spaces/berataydemir/NeuroSegment_3D)


This project is a deep learning-based system designed for the automated detection and volumetric segmentation of brain tumors from Magnetic Resonance Imaging (MRI) data. Utilizing a 3D U-Net architecture, the model processes NIfTI sequences to preserve spatial depth information, offering a significant advantage over traditional 2D slice-based analysis.


## System Interface & Demonstration


The system is deployed and accessible for research testing via Hugging Face Spaces. It features a professional radiology-grade interface for visualizing segmentation masks over anatomical structures.


![Interface Preview](demo_preview.png)


## Project Overview


Medical image analysis, particularly in neuro-oncology, requires precise delineation of pathological tissue. This system leverages the MONAI (Medical Open Network for AI) framework to process volumetric data, identifying tumor regions with high spatial consistency.


### Key Technical Features

* **Architecture:** 3D U-Net (Encoder-Decoder with Residual Units).

* **Input Data:** Single-channel volumetric MRI (T1-weighted sequences).

* **Output:** Binary segmentation masks (Tumor vs. Background).

* **Processing:** Full volumetric inference (96x96x96 voxel resolution).


## Technology Stack


* **Core Framework:** PyTorch & MONAI

* **Data Handling:** Nibabel (NIfTI processing)

* **Visualization:** Matplotlib & NumPy

* **Deployment:** Gradio (Web Interface)


## Installation and Local Usage


To run the system locally, follow these steps:


1.  **Clone the Repository:**

    ```bash

    git clone [https://github.com/berataydemirr/NeuroSegment-3D.git](https://github.com/berataydemirr/NeuroSegment-3D.git)

    cd NeuroSegment-3D

    ```


2.  **Install Dependencies:**

    ```bash

    pip install -r requirements.txt

    ```


3.  **Launch the Application:**

    ```bash

    python app.py

    ```
# NeuroSegment-3D: Volumetric Brain Tumor Segmentation

[![Hugging Face Spaces](https://img.shields.io/badge/Hugging%20Face-Live%20Demo-blue)](https://huggingface.co/spaces/berataydemir/NeuroSegment_3D)
[![Python](https://img.shields.io/badge/Python-3.10-3776AB)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0-EE4C2C)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

This project implements a deep learning-based system for the automated detection and volumetric segmentation of brain tumors using Magnetic Resonance Imaging (MRI) data. Built upon a 3D U-Net architecture, the model processes NIfTI sequences to preserve spatial depth information, offering a significant diagnostic advantage over traditional 2D slice-based analysis methods.

## System Interface & Demonstration

The system is fully deployed and accessible for research validation via Hugging Face Spaces. It features a professional radiology-grade interface for visualizing segmentation masks over anatomical structures in real-time.

![Interface Preview](demo_preview.png)

## System Architecture

The following pipeline illustrates the end-to-end processing flow, from raw NIfTI data ingestion to the generation of the segmentation mask:

```mermaid
graph LR
    A[Input MRI .nii.gz] -->|Load & Normalize| B(Preprocessing)
    B -->|Resize to 96x96x96| C{3D U-Net Model}
    C -->|Feature Extraction| D[Encoder Path]
    D -->|Reconstruction| E[Decoder Path]
    E -->|Sigmoid Activation| F[Segmentation Mask]
    F -->|Post-Processing| G[3D Visualization]

## Disclaimer


This software is developed for **academic and research purposes only**. It is not a certified medical device and should not be used for clinical diagnosis, treatment planning, or surgical guidance. ek olarak ne ekleyebilirim daha havali durmasi adina yani insanlar bakinca wow olsun
