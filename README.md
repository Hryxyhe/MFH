# <p align="center">MFH：Marrying Frequency Domain with Handwritten Mathematical Expression Recognition</p>

<p align="center">
  <a href="https://link.springer.com/chapter/10.1007/978-981-97-8511-7_13">
    <img src="https://img.shields.io/badge/Google Scholar-MFH-blue" >
  </a>
</p>

An officical implementation of "MFH: Marrying Frequency Domain with Handwritten Mathematical Expression Recognition" (Accepted by PRCV 2024). We implement our method based on [CoMER](https://arxiv.org/abs/2207.04410).
  
<img src="https://github.com/Hryxyhe/MFH/blob/master/demos/Pipeline.jpg" alt='Pipeline of MFH'>

## News 
* ```2024.9.18 ``` 🚀 MFH is selected as the PRCV 2024 Oral！
* ```2024.6.15 ``` 🚀 MFH is accepted by PRCV 2024！

# Installation
Our experiments are implemented on the following environments: Python:3.7.16  PyTorch-lighting:1.4.9  Pytorch:1.13  CUDA:11.7
```
git clone https://github.com/Hryxyhe/MFH.git
cd MFH
conda create -y -n MFH python=3.7.16
conda activate MFH
pip install pytorch==1.13.1 torchvision==0.14.1 
# training dependency
pip install pytorch-lightning==1.4.9 torchmetrics==0.6.0 pandoc==2.3 scipy torch_dct
pip install -e .
```
# Training
Our code are primarily based on [CoMER Project](https://github.com/Green-Wood/CoMER). Besides, we implement discrete cosine transform (DCT) on input images. This will be done during data loading, before 
the training stage. So you also don't need any additional preprocessing for the data.

<img src="https://github.com/Hryxyhe/MFH/blob/master/demos/data processing.jpg">

We also propose Fusion and Alignment Block (FAB) to mix spatial domain features and frequency domain features.

<div align=center><img src="https://github.com/Hryxyhe/MFH/blob/master/demos/FAB.jpg" width='650px'/></div>

Simply run the following code as same as CoMER to start training:
```
python train.py --config config.yaml 
```
You could modify the ```config.yaml``` according to your available gpus.
```
gpus: 0,1,2,3 #change gpu ids here
accelerator: ddp
```

# Test
As demonstrate in [CoMER Project](https://github.com/Green-Wood/CoMER), metrics used in validation during the training process is not accurate, you could run the test code at ```scripts/test/test.py``` for accurate metrics. This will test all three CROHME datasets in turn.
```
perl --version  # make sure you have installed perl 5
unzip -q data.zip
# evaluation
# evaluate model in lightning_logs/version_0 on all CROHME test sets
# results will be printed in the screen and saved to lightning_logs/version_0 folder
python scripts/test/test.py 0   #change the number 0 according to your saved pretrained weights
```
We provide a pretrained weights at ```MFH/lightning_logs/version_0/checkpoints/```. You can also train your own model and the default saving path will be ```MFH/lightning_logs/version_x/checkpoints/```
