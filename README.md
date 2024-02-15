# Cardiovascular Disease Diagnosis through Cardiac Image Segmentation

## Introduction

This repository contains the code and resources for the paper titled "Cardiovascular Disease Diagnosis through Cardiac Image Segmentation," accepted for ISBI 2024 – International Symposium on Biomedical Imaging. https://biomedicalimaging.org/2024/

## Abstract

Automated diagnosis of cardiovascular disease relies on cardiac image segmentation, a critical task in medical imaging. Our proposed strategy employs specialist networks focusing on individual anatomies (left ventricle, right ventricle, or myocardium) to achieve accurate segmentation. In the initial stage, a ternary segmentation is performed to identify these anatomical regions, followed by cropping the original image for focused processing. An attention mechanism, facilitated by an additive attention block (E-2A block), interlinks features from different anatomies, serving as a soft relative shape prior. Our approach aims for efficiency and effectiveness in cardiac image segmentation.

## Methodology

Overview of the proposed CroCNet pipeline. (a) E-2AUNet architecture for initial segmentation in the first stage. The cropped image and initial binary segmentations are fed into (b) specialist networks (LV-Net, RV-Net, and MYO-Net) to
refine the predictions for each cardiac region. We leverage (c) efficient additive attention, including (d) a cross-E2A block that implements cross-attention to intermingle features between the specialist networks.

## Results

Comparison of the results obtained from different methods. Methods indicated with the ∗ use of multi-view inputs.

## Citation

Abbas Khan, Muhammad Asad, Martin Benning, Caroline Roney, Gregory Slabaugh, 'Crop and Couple: cardiac image segmentation using interlinked specialist networks', https://arxiv.org/abs/2402.09156

[Include citation details once available.]

## Acknowledgments

[Include any acknowledgments or credits for third-party resources.]

## License

This project is licensed under the [License Name] - see the [LICENSE.md](LICENSE.md) file for details.

---

**Note: This paper has been accepted for ISBI 2024 – International Symposium on Biomedical Imaging.**
