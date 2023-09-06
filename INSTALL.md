## Installation
Tested on Ubuntu 22.04 with: Pytorch 1.9.0, torchvision 0.10.0, detectron2 0.6, Python 3.9, CUDA 11.1<br/> <br/>

**1) Download and install Anaconda:**
- download anaconda: https://www.anaconda.com/download (python 3.x version)
- install anaconda (using the terminal, cd to the directory where the file has been downloaded): bash Anaconda3-[distribution].sh <br/> <br/>

**2) Make a virtual environment (called potatoscan) using the terminal:**
- conda create --name potatoscan python=3.9 pip
- conda activate potatoscan <br/> <br/>

**3) Downgrade setuptools, to prevent this [error](https://github.com/facebookresearch/detectron2/issues/3811):**
- pip uninstall setuptools
- pip install setuptools==59.5.0 <br/> <br/>

**4) Download the code repository including the submodules:**
- git clone --recurse-submodules https://github.com/UTokyo-FieldPhenomics-Lab/PotatoScan.git <br/> <br/>

**5) Install the required software libraries (in the potatoscan virtual environment, using the terminal):**
- cd PotatoScan
- pip install -U torch==1.9.0 torchvision==0.10.0 -f https://download.pytorch.org/whl/cu111/torch_stable.html
- pip install cython
- pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
- pip install jupyter
- pip install opencv-python
- pip install -U fvcore
- pip install scikit-image matplotlib imageio
- pip install black isort flake8 flake8-bugbear flake8-comprehensions
- cd detectron2
- pip install -e . 
- pip install open3d <br/> <br/>

**6) Reboot/restart the computer (sudo reboot)** <br/> <br/>

**7) Check if Pytorch links with CUDA (in the potatoscan virtual environment, using the terminal):**
- python
- import torch
- torch.version.cuda *(should print 11.1)*
- torch.cuda.is_available() *(should True)*
- torch.cuda.get_device_name(0) *(should print the name of the first GPU)*
- quit() <br/> <br/>

**8) Check if detectron2 is found in python (in the potatoscan virtual environment, using the terminal):**
- python
- import detectron2 *(should not print an error)*
- from detectron2 import model_zoo *(should not print an error)*
- quit() <br/><br/>
