import os
import sys
sys.path.insert(0, os.getcwd())
from datasets.convert_to_coco import LINCDatasetConverter  # noqa

category_grouping_dict = {
    'cv': set([
        'cv-sright', 'cv-r', 'cv-dr-r', 'cv-f', 'cv-l', 'cv-sr', 'cv-dr', 'cv-sl', 'cv-dl-r',
        'cv-dr-l', 'cv-dl', 'cv-front'
    ]),
    'nose': set([
        'nose-dl', 'nose-dr', 'nose-l', 'nose-sl', 'nose-dl-l', 'nose-fl', 'nose-sr',
        'nose', 'nose-f', 'nose-r', 'nose-dr-r', 'nose-slw'
    ]),
    'ear': set([
        'ear-sl', 'ear-sr-r', 'ear-dr-r-marking', 'ear-sr-marking', 'ear-fl', 'ear-dl-l-marking',
        'ear-dr-r', 'ear-fr', 'ear-fr-marking', 'ear-dl-r', 'ear-dl-l', 'ear-dr', 'ear-sr-l',
        'ear-f', 'ear-sr', 'ear-dl', 'ear-f-l', 'ear-dr-l', 'ear-f-r'
    ]),
    'whisker_area': set([
        'whikser-dr', 'whikser-dl', 'whisker-r', 'whisker-s', 'whisker-sr', 'whisker-dl',
        'whisker-sl', 'whisker-l', 'whisker-dr', 'whiske-dr', 'whisker-f'
    ]),
    'mouth': set(['mouth-dr', 'mouth-dl']),
    'eye': set([
        'eye-dr', 'eye-dl-r', 'eye-d-l', 'eye-dr-r', 'eye-fl', 'eye-sl', 'eye-f-r', 'eye-sr-l',
        'eye-sr-r', 'eye-fr', 'eye-f', 'eye-dl', 'eye-sr', 'eye-dr-l', 'eye-f-l', 'eye-dl-l'
    ]),
}

dataset_creator = LINCDatasetConverter(category_grouping_dict)
dataset_creator.parse_arguments()
dataset_creator.create_coco_dataset()
