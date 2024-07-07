# MFH
An officical implementation of "MFH: Marrying Frequency Domain with Handwritten Mathematical Expression Recognition" (Accepted by PRCV 2024). We implement our method based on [CoMER](https://arxiv.org/abs/2207.04410).
  
<img src="https://github.com/Hryxyhe/MFH/raw/master/material/Pipeline.jpg" width="800px">
<img src="https://github.com/Hryxyhe/MFH/raw/master/material/FAB.jpg" width="800px" height="400px">

# Installation
Our experiments are implemented on the following environments: Python:3.7.16  PyTorch-lighting:1.4.9  Pytorch:1.13  CUDA:11.7
```
git clone https://github.com/Hryxyhe/MFH.git
cd MFH
conda create -y -n MFH python=3.7.16
conda activate MFH
conda install pytorch=1.8.1 torchvision=0.2.2 cudatoolkit=11.1 pillow=8.4.0 -c pytorch -c nvidia
# training dependency
conda install pytorch-lightning=1.4.9 torchmetrics=0.6.0 -c conda-forge
# evaluating dependency
conda install pandoc=1.19.2.1 -c conda-forge
pip install -e .
```
#Training
Our code are primarily based on [CoMER Project](https://github.com/Green-Wood/CoMER)
