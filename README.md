# iWave
Wavelet-like Transform with CNN re-implementation.

This repo provides the official implementation of "[iWave: CNN-Based Wavelet-Like Transform for Image Compression](https://ieeexplore.ieee.org/abstract/document/8931632)".

Accepted by IEEE TMM.

Author: Haichuan Ma, Dong Liu, Ruiqin Xiong, Feng Wu

## **BibTeX**

@article{ma2019iwave,

  title={iWave: CNN-Based Wavelet-Like Transform for Image Compression},
  
  author={Ma, Haichuan and Liu, Dong and Xiong, Ruiqin and Wu, Feng},
  
  journal={IEEE Transactions on Multimedia},
  
  year={2019},
  
  publisher={IEEE}
  
}

## **Update Notes**

2020.10.29 Upload code and pre-trained models.

2020.10.29 Init this repo.

## **How To Test**

0. Dependencies. We test with MIT deepo docker image.

1. Transform an image with transform.py. PTAHs or image SIZE need to be modified.

2. Reconstruct an image from iWave cofficients with inver_transform.py. PTAHs or image SIZE need to be modified.

3. Have fun!


## **Results**

iWave++ outperforms [Joint](http://papers.nips.cc/paper/8275-joint-autoregressive-and-hierarchical-priors-for-learned-image-compression), [Variational](https://arxiv.org/abs/1802.01436), and [iWave](https://ieeexplore.ieee.org/abstract/document/8931632). For more information, please refer to the paper.

1. RGB PSNR on Kodak dataset.

![image](https://github.com/mahaichuan/Versatile-Image-Compression/blob/master/figs/Kodak.PNG)

2. RGB PSNR on Tecnick dataset.

![image](https://github.com/mahaichuan/Versatile-Image-Compression/blob/master/figs/Tecnick.PNG)
