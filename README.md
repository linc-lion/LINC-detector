![logo](http://linclion.com/linc/wp-content/uploads/2015/06/lincLogo1200-e1525280990866.png)

# LINC Object Detector

This project intends to help the LINC project in the process of identifying lions by using their pictures. In particular, the purpose of this project is to locate certain lion body parts and their pose with regards to the camera.

## Installation
Python 3.6 or newer is needed.

First run:
```bash
pip install -r requirements.txt
```

Then, if you intend to do training, the package `pycocotools` is needed for running the evaluation scripts while training. Instructions are as follows:
```bash
pip install cython
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
python setup.py build_ext install
```

NOTE: For inference the only necessary requirement is torchvision==0.3.0, the other 3 requirements and `pycocotools` are just for visualizations and eval during training.

## Notes
- Saved some checkpoints in the 'Releases' part of the repo, in pytorch checkpoints are `*.pth` files.
- There are several Jupyter notebooks for several data viewing tasks.
- `python train.py` for training.
- `tensorboard --logdir=runs` for looking at training logs.
- `python predict.py` for predicting local pictures.
- `python create_*_dataset.py` for creating custom datasets from original LINC dataset.
