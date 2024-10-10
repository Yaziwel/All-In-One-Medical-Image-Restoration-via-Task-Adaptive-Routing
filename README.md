# All-In-One Medical Image Restoration via Task-Adaptive Routing (AMIR)

PyTorch implementation for All-In-One Medical Image Restoration via Task-Adaptive Routing [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2405.19769) (MICCAI 2024).

## Network Architecture

![](README.assets/network_arch.png)

## Dataset

You can download the preprocessed datasets for MRI super-resolution, CT denoising, and PET synthesis from Baidu Netdisk [here](https://pan.baidu.com/s/1oBBG_Stcn7cfO8U49S146w?pwd=3x13 ).

The original dataset for MRI super-resolution and CT denoising are as follows:

- MRI super-resolution: [IXI dataset](http://brain-development.org/ixi-dataset/)

- CT denoising: [AAPM dataset](https://www.aapm.org/grandchallenge/lowdosect/)

## Visualization

You can use [AMIDE](https://amide.sourceforge.net/) to visualize the ".nii" file. Note that the color map for MRI and CT images is "black/white linear," while the color map for PET images is "white/black linear." Additionally, you need to rescale the PET image according to the voxel size specified in the paper.

![](README.assets/vis.png)

## Citation

If you find AMIR useful in your research, please consider citing:

```bibtex
@inproceedings{yang2024amir,
  title={All-In-One Medical Image Restoration via Task-Adaptive Routing},
  author={Yang, Zhiwen and Chen, Haowei and Qian, Ziniu and Yi, Yang and Zhang, Hui and Zhao, Dan and Wei, Bingzheng and Xu, Yan},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={67--77},
  year={2024},
  organization={Springer}
}
```

