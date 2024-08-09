# Adaptive Body Part ROI with Triple Stream Approach

This repository contains the code implemented as part of the research for the paper titled "Adaptive Body Part ROI with Triple Stream Approach for Human Action Recognition," authored by Worawit Tepsan, Sitapa Watcharapinchai, Pitiwat Lueangwitchajaroen, and Sorn Sooksatra. 

## Abstract
This study explores Adaptive Regions of Interest (ROI) methods in Human Action Recognition (HAR) through the utilization of OpenPose keypoints for ROI image generation from video data. Utilizing the NTU RGB+D 60 dataset and the EfficientNetB7 model, we examine ROIs ranging from full-body to specific joint segmentations. We propose a Triple Stream approach—where each stream employs a unique ROI image generation process. Our results demonstrate that the Triple Stream approach, combining Full Body Segmentation, 7 Joint ROI, and 6 Joint ROI, significantly enhances HAR accuracy for the XSUB benchmark. Similarly, for the XVIEW benchmark, a combination of Full Body Segmentation, 7 Joint ROI, and 3 Joint ROI significantly improves accuracy. Our proposed approach can also be adapted to enhance the performance of other models. Notably, by integrating the Triple Stream approach with alterations to the RGB channel in MMNet \cite{MMnet}, we achieve accuracies of 97.2\% on the XSUB benchmark and 99.3\% on XVIEW.

## Download Paper
LINK: https://doi.org/10.1109/JCSSE61278.2024.10613745

## System Requirements
- Python 3.9

## Code Implementation
To implement the code, you will need to download dataset, pretrained models and set up paths properly. I will futher add some details later. So sorry for an inconvenience. 

## Citation

If you use this code or our findings in your research, please cite our paper as follows:

```bibtex
@INPROCEEDINGS{10613745,
  author={Tepsan, Worawit and Sooksatra, Sorn and Lueangwitchajaroen, Pitiwat and Watcharapinchai, Sitapa},
  booktitle={2024 21st International Joint Conference on Computer Science and Software Engineering (JCSSE)}, 
  title={Adaptive Body Part ROI with Triple Stream Approach for Human Action Recognition}, 
  year={2024},
  volume={},
  number={},
  pages={79-85},
  keywords={Image segmentation;Adaptation models;Accuracy;Image synthesis;Training data;Streaming media;Benchmark testing;─Human Action Recognition;ROI},
  doi={10.1109/JCSSE61278.2024.10613745}
}

