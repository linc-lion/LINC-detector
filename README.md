![logo](http://linclion.com/linc/wp-content/uploads/2015/06/lincLogo1200-e1525280990866.png)

# LINC Object Detector

This project intends to help the LINC project in the process of identifying lions by using their pictures. In particular, the purpose of this project is to locate certain lion body parts and their pose with regards to the camera.

## Installation
Python 3.6 or newer is needed.
```bash
pip install -r requirements.txt  # If you don't like Pipenv
pipenv install  # If you like Pipenv
```
Only hard requirement is torchvision==0.3.0, the other 3 requirements are just for visualizations really.

## Notes
- Saved some checkpoints in the 'Releases' part of the repo, in pytorch checkpoints are `*.pth` files.
- There are several Jupyter notebooks for several data viewing tasks.
- `python train.py` for training.
- `tensorboard --logdir=runs` for looking at training logs.
- `python predict.py` for predicting local pictures.
- `python create_*_dataset.py` for creating custom datasets from original LINC dataset.
