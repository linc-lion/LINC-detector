from convert_to_coco import LINCDatasetConverter

category_grouping_dict = {
    'cv-f': set(['cv-f']),
    'cv-sr': set(['cv-sr']),
    'cv-sl': set(['cv-sl']),
    'whisker-sl': set(['whisker-sl']),
    'whisker-sr': set(['whisker-sr']),
}

dataset_creator = LINCDatasetConverter(category_grouping_dict)
dataset_creator.parse_arguments()
dataset_creator.create_coco_dataset()
