# ADASR: An Adversarial Auto-Augmentation Framework for Hyperspectral and Multispectral Data Fusion
---
This repo is the official implementation of "ADASR: An Adversarial Auto-Augmentation Framework for Hyperspectral and Multispectral Data Fusion"

Our paper is accpeted by IEEE Geoscience and Remote Sensing Letters (GRSL).

The early access version can be downloaded in [arxiv](https://arxiv.org/pdf/2310.07255.pdf).

<img src="./imgs/overall.png"/>
Fig.1. overview. (a) The adversarial auto-augmentation framewor (b) The design of our data augmentor G.


### checkpoints
This folder is used to store the training results and a folder named `houston18_5_S1=0.001_20000_10000_S2=0.001_30000_20000_S3=6e-05_15000_5000` is given as a example.

- `convolution_hr2msi.pth` is the trained result of SpeDnet, `PSF.pth` is the trained result of SpaDnet, and `spectral_upsample.pth` is the trained result of SpeUnet.

- `opt.txt` is used to store the training configuration.

- `log.txt` is used to store the training process result.

- `My_Out.mat` is the final reconstructed HHSI.

### data
This folder is used to store the ground true HHSI and corresponding spectral response of multispectral imager. The HSI data used in [2018 IEEE GRSS Data Fusion Contest](https://hyperspectral.ee.uh.edu/?page_id=1075)  and spectral response of WorldView 2 multispectral imager are given as a example here.

### How to run our code
- Requirements: codes of networks were tested using PyTorch 1.9.0 version (CUDA 11.1) in Python 3.8.10 on Linux system. For the required packages, please using command：```pip install -r requirement.txt```

- Parameters: all the parameters need fine-tunning can be found in `config.py`, including the learning rate decay strategy of three training stages.

- Data: put your HSI data and MSI spectral reponse in `./data/data_name` and `./data/spectral_response`, respectively.The HSI data used in [2018 IEEE GRSS Data Fusion Contest](https://hyperspectral.ee.uh.edu/?page_id=1075)  and spectral response of WorldView 2 multispectral imager are given as a example here.

- Run: just simply run `train.py` after adjust the parameters in `config.py`.

- Results: one folder named `dataname_SF_S1=x1_y1_z1_S2=x2_y2_z2` will be generated once `train.py` is run and all the results will be stored in the new folder. A folder named `houston18_5_S1=0.0008_40000_10000_S2=0.0013_200000_20000` is given as a example here.

### Acknowledgement
The `spatial_downsample.py(SpaDnet)`, `spectral_downsample.py(SpeDnet)`, `spectral_upsample.py(SpeUnet)` is modified from UDALN.

Jiaxin Li, Ke Zheng, Jing Yao, Lianru Gao, Danfeng Hong. Deep unsupervised blind hyperspectral and multispectral data fusion[J]. IEEE Geoscience and Remote Sensing Letters, 2022, 19: 1-5.

### Repo TODOs
Fix Bug. Thanks for Jiaxin Li.

Other description will be released soon.
