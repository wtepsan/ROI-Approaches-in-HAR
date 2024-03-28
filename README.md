# Adaptive Body Part ROI with Triple Stream Approach

This repository contains the code implemented as part of the research for the paper titled "Adaptive Body Part ROI with Triple Stream Approach," authored by Worawit Tepsan, Sitapa Watcharapinchai, Pitiwat Lueangwitchajaroen, and Sorn Sooksatra. The paper has been published in 2024 and presents an in-depth analysis of different Adaptive Regions of Interest (ROI) techniques in the field of Human Action Recognition (HAR).

## Abstract
In this paper, we present experiments with different methods for extracting a representative image from a video using OpenPose keypoints. Based on the comparative results, the best among the methods is the 7 joints ROI. In addition, an ensemble of results from the 7 joints ROI and Full Body Segmentation can significantly increase the accuracy of action recognition. Moreover, when this is combined with another modality, such as in MMNets, which replaces the RGB channels, it shows a significant improvement in accuracy. The results can compete with other state-of-the-art approaches. 
As we can see, there are many factors that affect the efficacy of Human Action Recognition (HAR) models, such as frame selection. In this study, we have employed random frame selection, which is a straightforward approach to frame choice. However, there is potential for improving accuracy by exploring various frame selection methodologies. Therefore, investigating and enhancing frame selection techniques will be the focus of our future research

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
