# 3D ResNets for PPG measurements

Base ResNet3D files were cloned from:
https://github.com/kenshohara/3D-ResNets-PyTorch

## Pulse Measurements
This repository implements PPG measurements based on face videos using ResNet 3D. 

## ResNet 3D info 
Paper on arXiv.

[
Hirokatsu Kataoka, Tenga Wakamiya, Kensho Hara, and Yutaka Satoh,  
"Would Mega-scale Datasets Further Enhance Spatiotemporal 3D CNNs",  
arXiv preprint, arXiv:2004.04968, 2020.
](https://arxiv.org/abs/2004.04968)

PyTorch code for the following papers:

[
Hirokatsu Kataoka, Tenga Wakamiya, Kensho Hara, and Yutaka Satoh,  
"Would Mega-scale Datasets Further Enhance Spatiotemporal 3D CNNs",  
arXiv preprint, arXiv:2004.04968, 2020.
](https://arxiv.org/abs/2004.04968)

[
Kensho Hara, Hirokatsu Kataoka, and Yutaka Satoh,  
"Towards Good Practice for Action Recognition with Spatiotemporal 3D Convolutions",  
Proceedings of the International Conference on Pattern Recognition, pp. 2516-2521, 2018.
](https://ieeexplore.ieee.org/document/8546325)

[
Kensho Hara, Hirokatsu Kataoka, and Yutaka Satoh,  
"Can Spatiotemporal 3D CNNs Retrace the History of 2D CNNs and ImageNet?",  
Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 6546-6555, 2018.
](http://openaccess.thecvf.com/content_cvpr_2018/html/Hara_Can_Spatiotemporal_3D_CVPR_2018_paper.html)

[
Kensho Hara, Hirokatsu Kataoka, and Yutaka Satoh,  
"Learning Spatio-Temporal Features with 3D Residual Networks for Action Recognition",  
Proceedings of the ICCV Workshop on Action, Gesture, and Emotion Recognition, 2017.
](http://openaccess.thecvf.com/content_ICCV_2017_workshops/papers/w44/Hara_Learning_Spatio-Temporal_Features_ICCV_2017_paper.pdf)

This code includes training, fine-tuning and testing on Kinetics, Moments in Time, ActivityNet, UCF-101, and HMDB-51.

## Citation

If you use this code or pre-trained models, please cite the following:

```bibtex
@inproceedings{hara3dcnns,
  author={Kensho Hara and Hirokatsu Kataoka and Yutaka Satoh},
  title={Can Spatiotemporal 3D CNNs Retrace the History of 2D CNNs and ImageNet?},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages={6546--6555},
  year={2018},
}
```
