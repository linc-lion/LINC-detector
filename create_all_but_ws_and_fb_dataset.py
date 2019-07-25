from convert_to_coco import LINCDatasetConverter

category_grouping_dict = {
    "cv-dl": set(["cv-dl"]),  # 1
    "cv-dr": set(["cv-dr"]),  # 2
    "cv-f": set(["cv-f"]),  # 3
    "cv-sl": set(["cv-sl"]),  # 4
    "cv-sr": set(["cv-sr"]),  # 5
    "ear-dl-l": set(["ear-dl-l"]),  # 6
    "ear-dl-r": set(["ear-dl-r"]),  # 7
    "ear-dr-l": set(["ear-dr-l"]),  # 8
    "ear-dr-r": set(["ear-dr-r"]),  # 9
    "ear-fl": set(["ear-fl"]),  # 10
    "ear-fr": set(["ear-fr"]),  # 11
    "ear-sl": set(["ear-sl"]),  # 12
    "ear-sr": set(["ear-sr"]),  # 13
    "eye-dl-l": set(["eye-dl-l"]),  # 14
    "eye-dl-r": set(["eye-dl-r"]),  # 15
    "eye-dr-l": set(["eye-dr-l"]),  # 16
    "eye-dr-r": set(["eye-dr-r"]),  # 17
    "eye-fl": set(["eye-fl"]),  # 18
    "eye-fr": set(["eye-fr"]),  # 19
    "eye-sl": set(["eye-sl"]),  # 20
    "eye-sr": set(["eye-sr"]),  # 21
    "nose-dl": set(["nose-dl"]),  # 22
    "nose-dr": set(["nose-dr"]),  # 23
    "nose-f": set(["nose-f"]),  # 24
    "nose-sl": set(["nose-sl"]),  # 25
    "nose-sr": set(["nose-sr"]),  # 26
    "whisker-dl": set(["whisker-dl"]),  # 27
    "whisker-dr": set(["whisker-dr"]),  # 28
    "whisker-f": set(["whisker-f"]),  # 29
    "whisker-sl": set(["whisker-sl"]),  # 30
    "whisker-sr": set(["whisker-sr"]),  # 31
}


# Ignore pictures with any whisker spot annotation
def ignore_picture_if(objects):
    for o in objects:
        if o['name'] == 'ws':
            return True
    return False


dataset_creator = LINCDatasetConverter(category_grouping_dict, ignore_picture_fn=ignore_picture_if)
dataset_creator.parse_arguments()
dataset_creator.create_coco_dataset()
