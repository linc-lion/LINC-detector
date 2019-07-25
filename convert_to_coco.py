import os
import xml.etree.ElementTree as ET
import collections
import json
import shutil

from PIL import Image
from utils import mkdir


train_val_ratio = 5
coco_train = {
    "licenses": [
        {"url": "http://creativecommons.org/licenses/by-nc-sa/2.0/",
         "id": 1,
         "name": "Attribution-NonCommercial-ShareAlike License"}
    ],
    "info": {
        "description": "LINC train dataset",
        "url": "http://iefrd.com/linc.html",
        "version": "1.0",
        "year": 2019,
        "contributor": "tryolabs",
        "date_created": "2019/07/01"
    },
    "categories": [
        {"supercategory": "lion", "id": 1, "name": "marking"},
        {"supercategory": "lion", "id": 2, "name": "cv"},
        {"supercategory": "lion", "id": 3, "name": "nose"},
        {"supercategory": "lion", "id": 4, "name": "ear"},
        {"supercategory": "lion", "id": 5, "name": "whisker"},
        {"supercategory": "lion", "id": 6, "name": "mouth"},
        {"supercategory": "lion", "id": 7, "name": "eye"},
        {"supercategory": "lion", "id": 8, "name": "ws"},
        {"supercategory": "lion", "id": 9, "name": "full_body"},
    ],
    "images": [],
    "annotations": [],
}

coco_val = {
    "licenses": [
        {"url": "http://creativecommons.org/licenses/by-nc-sa/2.0/",
         "id": 1,
         "name": "Attribution-NonCommercial-ShareAlike License"}
    ],
    "info": {
        "description": "LINC val dataset",
        "url": "http://iefrd.com/linc.html",
        "version": "1.0",
        "year": 2019,
        "contributor": "tryolabs",
        "date_created": "2019/07/01"
    },
    "categories": [
        {"supercategory": "lion", "id": 1, "name": "marking"},
        {"supercategory": "lion", "id": 2, "name": "cv"},
        {"supercategory": "lion", "id": 3, "name": "nose"},
        {"supercategory": "lion", "id": 4, "name": "ear"},
        {"supercategory": "lion", "id": 5, "name": "whisker"},
        {"supercategory": "lion", "id": 6, "name": "mouth"},
        {"supercategory": "lion", "id": 7, "name": "eye"},
        {"supercategory": "lion", "id": 8, "name": "ws"},
        {"supercategory": "lion", "id": 9, "name": "full_body"},
    ],
    "images": [],
    "annotations": [],
}


class LINCDatasetConverter():

    def __init__(self, category_grouping_dict, ignore_picture_fn=None, crop_ws_area=False):
        # Determines what the dataset structure will be
        self.category_grouping_dict = category_grouping_dict
        self.category_labels = [l for l in self.category_grouping_dict.keys()]

        # Get object list as input and output determines if picture will be ignored
        if ignore_picture_fn is not None:
            self.ignore_picture = ignore_picture_fn
        else:
            self.ignore_picture = lambda x: False

        self.crop_ws_area = crop_ws_area

    def category_group(self, category):
        for key, value in self.category_grouping_dict.items():
            if category in value:
                return key

    def get_obj_annotation(self, o):
        annotation = {}

        # Parse categories. COCO Categories are 1 indexed! The 0 category_id is reserved for background.
        category_name = self.category_group(o['name'])
        if not category_name:
            return None
        annotation['category_id'] = self.category_labels.index(category_name) + 1  # 1 indexed

        # COCO format is x1, y1, width, height
        # LINC format is x1, y1, x2, y2
        # Both meassured from top left corner of image
        bbox = o['bndbox']
        bbox = [
            float(bbox['xmin']),
            float(bbox['ymin']),
            float(bbox['xmax']) - float(bbox['xmin']),
            float(bbox['ymax']) - float(bbox['ymin'])
        ]
        annotation['bbox'] = bbox
        annotation['image_id'] = self.img_counter
        annotation['area'] = (bbox[3] - bbox[1]) * (bbox[2] - bbox[0])
        annotation['iscrowd'] = 0
        annotation['id'] = self.obj_counter
        return annotation

    def get_img_annotation(self, image_name, img):
        return {
            'id': self.img_counter, 'file_name': image_name, 'height': img.size[1], 'width': img.size[0]
        }

    def parse_voc_xml(self, node):
        voc_dict = {}
        children = list(node)
        if children:
            def_dic = collections.defaultdict(list)
            for dc in map(self.parse_voc_xml, children):
                for ind, v in dc.items():
                    def_dic[ind].append(v)
            voc_dict = {
                node.tag:
                    {ind: v[0] if len(v) == 1 else v
                     for ind, v in def_dic.items()}
            }
        if node.text:
            text = node.text.strip()
            if not children:
                voc_dict[node.tag] = text
        return voc_dict

    def create_coco_dataset(self):
        # Check existence of root
        if not os.path.isdir(self.args.input_path):
            raise RuntimeError('Dataset not found.')

        # Create output directories
        if os.path.isdir(self.args.output_path):
            shutil.rmtree(self.args.output_path)
            print(f"Deleted '{self.args.output_path}' folder in order to use it as output!\n")
        mkdir(self.args.output_path)
        mkdir(os.path.join(self.args.output_path, 'train'))
        mkdir(os.path.join(self.args.output_path, 'val'))

        # Crawl sub-directories
        print(f"Parsing xml files in {self.args.input_path}", end='')
        self.obj_counter = 0
        self.img_counter = 0
        xml_files = 0
        for root, dirs, files in os.walk(self.args.input_path):
            dirs.sort()  # Lets make this deterministic so our ids are too.
            for file_name in [os.path.join(root, f) for f in files]:
                if os.path.splitext(file_name)[1] == '.xml':
                    xml_files += 1
                    data = self.parse_voc_xml(ET.parse(file_name).getroot())

                    # Ignore images with no objects in them
                    try:
                        objects = data['annotation']['object']
                    except KeyError:
                        continue
                    objects = objects if type(objects) is list else [objects]

                    # Filter images according to custom rule
                    if self.ignore_picture(objects):
                        continue

                    # Determine to which dataset this particular image is destined to
                    dataset = 'val' if self.img_counter % train_val_ratio == 0 else 'train'
                    annotation_dict = coco_val if dataset == 'val' else coco_train

                    # Process objects
                    relevant_objects = []
                    for o in objects:
                        target = self.get_obj_annotation(o)
                        if target:
                            relevant_objects.append(target)
                            self.obj_counter += 1

                    # Skip pictures that don't have the objects we are looking for
                    if len(relevant_objects) == 0:
                        continue

                    # Load image
                    input_image_path = os.path.join(root, data['annotation']['filename'])
                    image_name = os.path.basename(input_image_path)
                    pil_image = Image.open(input_image_path)

                    # Possibly crop image to just the whisker area
                    if self.crop_ws_area:
                        # Find whisker spot real area ('whisker area' annotation is not always present!)
                        min_ws_x, min_ws_y, max_ws_x, max_ws_y = pil_image.size[0], pil_image.size[1], 0, 0
                        for o in objects:
                            if o['name'] == 'ws':
                                min_ws_x = min_ws_x if min_ws_x < int(o['bndbox']['xmin']) else int(o['bndbox']['xmin'])  # noqa
                                min_ws_y = min_ws_y if min_ws_y < int(o['bndbox']['ymin']) else int(o['bndbox']['ymin'])  # noqa
                                max_ws_x = max_ws_x if max_ws_x > int(o['bndbox']['xmax']) else int(o['bndbox']['xmax'])  # noqa
                                max_ws_y = max_ws_y if max_ws_y > int(o['bndbox']['ymax']) else int(o['bndbox']['ymax'])  # noqa

                        # Find crop area
                        ws_area_height = max_ws_y - min_ws_y
                        ws_area_width = max_ws_x - min_ws_x
                        margin_factor = 0.5
                        crop_x_min = max(min_ws_x - margin_factor * ws_area_width, 0)
                        crop_y_min = max(min_ws_y - margin_factor * ws_area_height, 0)
                        crop_x_max = min(max_ws_x + margin_factor * ws_area_width, pil_image.size[0])
                        crop_y_max = min(max_ws_y + margin_factor * ws_area_height, pil_image.size[1])

                        # Crop picture
                        pil_image = pil_image.crop((crop_x_min, crop_y_min, crop_x_max, crop_y_max))

                        # Shift objects coordinates to new cropped picture coordinates
                        for o in relevant_objects:
                            o['bbox'][0] -= crop_x_min
                            o['bbox'][1] -= crop_y_min

                    # Save image
                    pil_image.save(os.path.join(self.args.output_path, dataset, image_name))
                    annotation_dict['images'].append(self.get_img_annotation(image_name, pil_image))

                    # Save objects
                    annotation_dict['annotations'].extend(relevant_objects)

                    self.img_counter += 1
                    print('.', flush=True, end='')

                    # If trimming dataset: escape function when we have reached the max number of images
                    if self.img_counter == self.args.trim_to:
                        # Print results
                        print(f" Done!\n")
                        print("Results:")
                        print(f"Created dataset trimmed dataset to just {self.args.trim_to} pictures.")
                        print(f"Looked at {xml_files} xml files.")
                        print(
                            f"Saved {self.img_counter} images with annotations, "
                            f"{len(coco_train['images'])} to train set and "
                            f"{len(coco_val['images'])} to validation set."
                        )
                        print(f"Dataset saved to '{self.args.output_path}'.")
                        return

        # Print results
        print(f" Done!\n")
        print("Results:")
        print(f"Looked at {xml_files} xml files.")
        print(
            f"Saved {self.img_counter} images with annotations, "
            f"{len(coco_train['images'])} to train set and "
            f"{len(coco_val['images'])} to validation set."
        )
        print(f"Dataset saved to '{self.args.output_path}'.")

        # Save annotations
        with open(os.path.join(self.args.output_path, 'train.json'), 'w') as f:
            json.dump(coco_train, f)
        with open(os.path.join(self.args.output_path, 'val.json'), 'w') as f:
            json.dump(coco_val, f)
        with open(os.path.join(self.args.output_path, 'labels.json'), 'w') as f:
            json.dump(self.category_labels, f)

    def parse_arguments(self):
        import argparse
        parser = argparse.ArgumentParser(description='LINC Dataset Converter')
        parser.add_argument('input_path', help='Path to input dataset folder')
        parser.add_argument('-o', '--output-path', help='Path to output dataset folder')
        parser.add_argument(
            '--trim-to', default=0, type=int,
            help='Create smaller dataset, useful for overfitting/debugging')
        self.args = parser.parse_args()
