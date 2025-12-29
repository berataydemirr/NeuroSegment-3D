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
