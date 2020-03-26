![logo](pictures/logo.png)

# LINC Object Detector

This project intends to help LINC in the process of identifying lions by processing their pictures through software. In particular, the purpose of this project is to act as a preprocessor, and extract usefull parts of the input picture to be used as features for the lion identifying models. [Documentation of design process and model specifications.](https://github.com/tryolabs/LINC/blob/master/LINC_-_Tryolabs.pdf)

Built using pytorch and based on torchvision's reference models.

![lions](pictures/lions_prediction_sample.png)
![whiskers](pictures/whiskers_prediction_sample.png)

## Installation
Python 3.6 or newer is needed.

First, clone this repository and run:
```bash
pip install -r requirements.txt
```

Then, if you intend to do training, the package `pycocotools` is needed for running the evaluation scripts while training. Instructions are as follows:
```bash
pip install cython
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
python setup.py build_ext install
cd -
```

NOTE: For inference the only necessary requirement is torchvision==0.3.0, the other requirements and `pycocotools` are just for visualizations and eval during training.

## Usage

### Creating datasets
The model consumes data in the COCO annotation format. Any object detection dataset in that format is trainable.

In particular, there are several scripts in the `datasets/` directory for converting the original LINC dataset into COCO. Each scripts creates a different dataset, with a different objective. They all inherit from `datasets/convert_to_coco.py` and they are really short as they use a single dict as a source of truth for the output dataset to be created.

### Training
Run `python train.py --help` for training. While training, very useful Tensorboard summaries are saved in the `runs` folder by default. Just run `tensorboard --logdir runs` to see the progress of your training jobs.

### Inference
Run `python predict.py --help` for inference. There is one [whisker spot checkpoint](https://github.com/tryolabs/LINC/releases/download/v1.0/whiskers.pth)(should have as an input pictures of the whisker area of the lion) and one [body parts checkpoint](https://github.com/tryolabs/LINC/releases/download/v1.0/body_parts.pth) (should have as an input any picture of a lion) in the [releases](https://github.com/tryolabs/LINC/releases) page of the repo.

The code used in creating the dataset that was used for training the body parts checkpoint is [here](https://github.com/tryolabs/LINC/blob/master/datasets/create_all_but_ws_and_fb_dataset.py) and it contains the mapping between label number and name for each class. On the whiskers checkpoint label number 1 corresponds to whisker spot, and 0 to background.

### Notebooks
There are several jupyter notebooks in the `notebooks/` directory which are useful for data exploration, and model evaluation results exploration. There is also a demo notebook for running an inference step chaining both models.
