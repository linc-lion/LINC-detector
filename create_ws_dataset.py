from convert_to_coco import LINCDatasetConverter

category_grouping_dict = {
    "ws": set(["ws"]),
}


dataset_creator = LINCDatasetConverter(category_grouping_dict, crop_ws_area=True)
dataset_creator.parse_arguments()
dataset_creator.create_coco_dataset()
