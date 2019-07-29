import os
import sys
sys.path.insert(0, os.getcwd())
from datasets.convert_to_coco import LINCDatasetConverter  # noqa

category_grouping_dict = {
    "cv-dl": set(["cv-dl"]),
    "cv-dr": set(["cv-dr"]),
    "cv-f": set(["cv-f"]),
    "cv-sl": set(["cv-sl"]),
    "cv-sr": set(["cv-sr"]),
    "ear-dl-l": set(["ear-dl-l"]),
    "ear-dl-r": set(["ear-dl-r"]),
    "ear-dr-l": set(["ear-dr-l"]),
    "ear-dr-r": set(["ear-dr-r"]),
    "ear-fl": set(["ear-fl"]),
    "ear-fr": set(["ear-fr"]),
    "ear-sl": set(["ear-sl"]),
    "ear-sr": set(["ear-sr"]),
    "eye-dl-l": set(["eye-dl-l"]),
    "eye-dl-r": set(["eye-dl-r"]),
    "eye-dr-l": set(["eye-dr-l"]),
    "eye-dr-r": set(["eye-dr-r"]),
    "eye-fl": set(["eye-fl"]),
    "eye-fr": set(["eye-fr"]),
    "eye-sl": set(["eye-sl"]),
    "eye-sr": set(["eye-sr"]),
    "nose-dl": set(["nose-dl"]),
    "nose-dr": set(["nose-dr"]),
    "nose-f": set(["nose-f"]),
    "nose-sl": set(["nose-sl"]),
    "nose-sr": set(["nose-sr"]),
    "whisker-dl": set(["whisker-dl"]),
    "whisker-dr": set(["whisker-dr"]),
    "whisker-f": set(["whisker-f"]),
    "whisker-sl": set(["whisker-sl"]),
    "whisker-sr": set(["whisker-sr"]),
    "ws": set(["ws"]),
}


dataset_creator = LINCDatasetConverter(category_grouping_dict)
dataset_creator.parse_arguments()
dataset_creator.create_coco_dataset()
