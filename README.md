# LINC Object Detector

This is README is functioning as a rough doc for the moment. Will be rewritten before project completion.

## Requirements
Python 3.6 or newer.
Look in Pipfile for more detailed requirements

## Notes
- Saved some checkpoints in the 'Releases' part of the repo, in pytorch checkpoints are `*.pth` files.
- There are several Jupyter notebooks for several data viewing tasks.
- `python train.py` for training.
- `tensorboard --logdir=runs` for looking at training logs.
- `python predict.py` for predicting local pictures.
- `python create_*_dataset.py` for creating custom datasets from original LINC dataset.
