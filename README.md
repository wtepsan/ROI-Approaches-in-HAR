# Comparative Analysis of Adaptive ROI Approaches in Human Action Recognition

This repository contains the code implemented as part of the research for the paper titled "Comparative Analysis of Adaptive ROI Approaches in Human Action Recognition," authored by Worawit Tepsan, Sitapa Watcharapinchai, Pitiwat Lueangwitchajaroen, and Sorn Sooksatra. The paper has been published in 2024 and presents an in-depth analysis of different Adaptive Regions of Interest (ROI) techniques in the field of Human Action Recognition (HAR).

## Abstract
 This study examines Adaptive Regions of Interest (ROI) methods in Human Action Recognition (HAR) by utilizing OpenPose keypoints for feature extraction from video data. Employing the NTU RGB+D 60 dataset and the EfficientNetB7 model for analysis, we assess ROIs ranging from full-body to specific joint segmentations. Our findings indicate that image inputs based on ROIs, particularly those focusing on 7-joint regions, substantially improve HAR accuracy. An ensemble method incorporating multiple image inputs further amplifies performance. Finally, by replacing the RGB channel in MMNet, the model achieves a 0.9718 accuracy score on the XSUB benchmark and a 0.9929 accuracy score on XVIEW, with the XSUB results establishing new standards in the field.

## Citation

If you use this code or our findings in your research, please cite our paper as follows:

```bibtex
@article{ComparativeROIHAR,
  author = {Tepsan, Worawit and Watcharapinchai, Sitapa and Lueangwitchajaroen, Pitiwat and Sooksatra, Sorn},
  doi = {00.0000/00000},
  journal = {Journal Title},
  month = sep,
  number = {1},
  pages = {1--6},
  title = {{Comparative Analysis of Adaptive ROI Approaches in Human Action Recognition}},
  volume = {1},
  year = {2024}
}
