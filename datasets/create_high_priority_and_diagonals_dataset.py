import os
import sys
sys.path.insert(0, os.getcwd())
from datasets.convert_to_coco import LINCDatasetConverter  # noqa

category_grouping_dict = {
    'cv-f': set(['cv-f', 'cv-dl', 'cv-dr']),  # label 1
    'cv-sr': set(['cv-sr']),  # label 2
    'cv-sl': set(['cv-sl']),  # label 3
    'whisker-sl': set(['whisker-sl']),  # label 4
    'whisker-sr': set(['whisker-sr']),  # label 5
}

dataset_creator = LINCDatasetConverter(category_grouping_dict)
dataset_creator.parse_arguments()
dataset_creator.create_coco_dataset()
