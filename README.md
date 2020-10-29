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

1. Decompose image iWave, compared with CDF9/7. (top: CDF9/7, bottom: iWave)

![image](https://github.com/mahaichuan/iWave/blob/main/figs/decom.PNG)

2. Reconstructions (gray-scale image pathes from Kodak-05 and Kodak-08), compared with JPEG-2000 (Jasper), at the same bit rate. (top: CDF9/7, bottom: iWave)

![image](https://github.com/mahaichuan/iWave/blob/main/figs/patches-1.PNG)

![image](https://github.com/mahaichuan/iWave/blob/main/figs/patches-2.PNG)
