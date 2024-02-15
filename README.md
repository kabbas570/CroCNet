# Cardiovascular Disease Diagnosis through Cardiac Image Segmentation

## Introduction

This repository contains the code and resources for the "Cardiovascular Disease Diagnosis through Cardiac Image Segmentation" paper accepted for ISBI 2024 – International Symposium on Biomedical Imaging. https://biomedicalimaging.org/2024/

## Abstract

Automated diagnosis of cardiovascular disease relies on cardiac image segmentation, a critical task in medical imaging. Our proposed strategy employs specialist networks focusing on individual anatomies (left ventricle, right ventricle, or myocardium) to achieve accurate segmentation. In the initial stage, a ternary segmentation is performed to identify these anatomical regions, followed by cropping the original image for focused processing. An attention mechanism, facilitated by an additive attention block (E-2A block), interlinks features from different anatomies, serving as a soft relative shape prior. Our approach aims for efficiency and effectiveness in cardiac image segmentation.

## Methodology

Overview of the proposed CroCNet pipeline. (a) E-2AUNet architecture for initial segmentation in the first stage. The cropped image and initial binary segmentations are fed into (b) specialist networks (LV-Net, RV-Net, and MYO-Net) to
refine the predictions for each cardiac region. We leverage (c) efficient additive attention, including (d) a cross-E2A block that implements cross-attention to intermingle features between the specialist networks.
![idea2](https://github.com/kabbas570/Medical-Image-Segmentation-From--Scratch-/assets/56618776/41b38184-ddd5-4ce7-b294-2b243d12fbd2)
## Results

Comparison of the results obtained from different methods. Methods indicated with the ∗ use of multi-view inputs.

![viz](https://github.com/kabbas570/Medical-Image-Segmentation-From--Scratch-/assets/56618776/07f6d2a3-be7e-4fc8-bcc6-79ecf67bc2a8)
![ab1](https://github.com/kabbas570/Medical-Image-Segmentation-From--Scratch-/assets/56618776/9ea765f9-bac9-4ff1-848c-2fe4a9a4fba1)
## Citation

Abbas Khan, Muhammad Asad, Martin Benning, Caroline Roney, Gregory Slabaugh, 'Crop and Couple: cardiac image segmentation using interlinked specialist networks', https://arxiv.org/abs/2402.09156


## Acknowledgments

This work was funded through the mini-Centre for Doctoral Training in AI-based Cardiac Image Computing provided through the Faculty of Science and Engineering, Queen Mary University of London. This paper utilized Queen Mary’s QMUL Research-IT Services support Andrena HPC facility. Caroline Roney acknowledges funding from a UKRI Future Leaders Fellowship (MR/W004720/1).

---

**Note: This paper has been accepted for ISBI 2024 – International Symposium on Biomedical Imaging.**
