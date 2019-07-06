import os
import xml.etree.ElementTree as ET
import collections
import json
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
    def __init__(self):
        self.category_relationships = {
            'markings': set(['markings', 'ear-fl-marking', 'mouth-f-marking', 'eye-dr-l-marking', 'tooth-marking', 'nose-f-marking', 'nose-dl-marking', 'eye-fl-marking', 'ear-dr-l-marking', 'ear-f-r-marking', 'mouth-dr-marking', 'nose-dr-marking', 'mouth-sl-marking', 'ear-dr-marking', 'ear-sl-marking', 'eye-dl-l-marking', 'ear-dl-r-marking', 'mouth-dl-marking', 'full-body-markings', 'marking']),  # noqa
            'cv': set(['cv-sright', 'cv-r', 'cv-dr-r', 'cv-f', 'cv-l', 'cv-sr', 'cv-dr', 'cv-sl', 'cv-dl-r', 'cv-dr-l', 'cv-dl', 'cv-front']),  # noqa
            'nose': set(['nose-dl', 'nose-dr', 'nose-l', 'nose-sl', 'nose-dl-l', 'nose-fl', 'nose-sr', 'nose', 'nose-f', 'nose-r', 'nose-dr-r', 'nose-slw']),  # noqa
            'ear': set(['ear-sl', 'ear-sr-r', 'ear-dr-r-marking', 'ear-sr-marking', 'ear-fl', 'ear-dl-l-marking', 'ear-dr-r', 'ear-fr', 'ear-fr-marking', 'ear-dl-r', 'ear-dl-l', 'ear-dr', 'ear-sr-l', 'ear-f', 'ear-sr', 'ear-dl', 'ear-f-l', 'ear-dr-l', 'ear-f-r']), # noqa
            'whisker_area': set(['whikser-dr', 'whikser-dl', 'whisker-r', 'whisker-s', 'whisker-sr', 'whisker-dl', 'whisker-sl', 'whisker-l', 'whisker-dr', 'whiske-dr', 'whisker-f']),  # noqa
            'mouth': set(['mouth-dr', 'mouth-dl']),
            'eye': set(['eye-dr', 'eye-dl-r', 'eye-d-l', 'eye-dr-r', 'eye-fl', 'eye-sl', 'eye-f-r', 'eye-sr-l', 'eye-sr-r', 'eye-fr', 'eye-f', 'eye-dl', 'eye-sr', 'eye-dr-l', 'eye-f-l', 'eye-dl-l']),  # noqa
            'whisker_spot': set(['ws']),
            'full_body': set(['full-body']),
        }

        # Customize dataset
        self.categories_to_ignore = ['markings', 'whisker_spot', 'full_body']
        for cat in self.categories_to_ignore:
            del(self.category_relationships[cat])
        self.category_order_for_label = ['cv', 'nose', 'ear', 'whisker_area', 'mouth', 'eye']

    def get_parent_category(self, child):
        for parent, children in self.category_relationships.items():
            if child in children:
                return parent
            else:
                continue

    def convert_obj_to_coco_format(self, o, img_counter, obj_counter):
        annotation = {}
        bbox = o['bndbox']
        # COCO format is x1, y1, width, height
        # LINC format is x1, y1, x2, y2
        # Both meassured from top left corner of image
        bbox = [
            float(bbox['xmin']),
            float(bbox['ymin']),
            float(bbox['xmax']) - float(bbox['xmin']),
            float(bbox['ymax']) - float(bbox['ymin'])
        ]
        annotation['bbox'] = bbox
        annotation['image_id'] = img_counter
        annotation['area'] = (bbox[3] - bbox[1]) * (bbox[2] - bbox[0])
        annotation['iscrowd'] = 0
        parent_category = self.get_parent_category(o['name'])
        annotation['category_id'] = self.category_order_for_label.index(parent_category)
        annotation['id'] = obj_counter
        print(annotation)
        return annotation

    def convert_img_to_coco_and_save(self, img_counter, image_name, img, output_folder):
        img.save(os.path.join(output_folder, image_name))
        output = {
            'id': img_counter, 'file_name': image_name, 'height': img.size[1], 'width': img.size[0]
        }
        print(output)
        return output

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

    def convert_to_coco(self, root, output_dir, image_set, trim_to, overfit):
        # Check existence of root
        if not os.path.isdir(root):
            raise RuntimeError('Dataset not found.')

        # Crawl sub-directories
        print(f"Crawling directories for {image_set} set...", end=' ')
        obj_counter = 1
        img_counter = 1
        for root, dirs, files in os.walk(root):
            dirs.sort()  # Lets make this deterministic so our ids are too.
            for file_name in [os.path.join(root, f) for f in files]:
                if os.path.splitext(file_name)[1] == '.xml':
                    data = self.parse_voc_xml(ET.parse(file_name).getroot())

                    # Ignore images with no objects in them
                    try:
                        objects = data['annotation']['object']
                    except KeyError:
                        continue
                    objects = objects if type(objects) is list else [objects]
                    image_path = os.path.join(root, data['annotation']['filename'])
                    image_name = os.path.basename(image_path)
                    img = Image.open(image_path)

                    # Separate train/val sets
                    # Ignore image if it doesn't belong to train set
                    if overfit or (image_set == 'train' and img_counter % train_val_ratio > 0):
                        coco_train['images'].append(
                            self.convert_img_to_coco_and_save(
                                img_counter, image_name, img, os.path.join(output_dir, 'train')
                            )
                        )
                        for o in objects:
                            try:
                                target = self.convert_obj_to_coco_format(
                                    o, img_counter, obj_counter
                                )
                            # Ignore categories filtered from category_relationships
                            except ValueError:
                                continue
                            coco_train['annotations'].append(target)
                            obj_counter += 1
                    # Ignore image if it doesn't belong to val set
                    if overfit or (image_set == 'val' and img_counter % train_val_ratio == 0):
                        coco_val['images'].append(
                            self.convert_img_to_coco_and_save(
                                img_counter, image_name, img, os.path.join(output_dir, 'val')
                            )
                        )
                        for o in objects:
                            try:
                                target = self.convert_obj_to_coco_format(
                                    o, img_counter, obj_counter
                                )
                            # Ignore categories filtered from category_relationships
                            except ValueError:
                                continue
                            coco_val['annotations'].append(target)
                            obj_counter += 1

                    img_counter += 1
                    if img_counter == trim_to:
                        print("Created trimmed dataset")
                        print(f"Done, parsed {img_counter} images.")
                        print("Make sure you customized 'category_order_for_label' and"
                              "'categories_to_ignore' before running!")
                        return

        print(f"Done, parsed {img_counter} images.")
        print("Make sure you customized 'category_order_for_label' and 'categories_to_ignore'"
              "before running!")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='LINC Dataset Converter')
    parser.add_argument('input_path', help='Path to input dataset folder')
    parser.add_argument('-o', '--output-path', help='Path to output dataset folder')
    parser.add_argument(
        '--trim-to', default=0, type=int,
        help='Create smaller dataset, useful for overfitting/debugging')
    parser.add_argument(
        "--overfit",
        dest="overfit",
        help="Make train and val datasets be the same, for overfitting for debugging.",
        action="store_true",
    )
    args = parser.parse_args()

    mkdir(args.output_path)
    mkdir(os.path.join(args.output_path, 'train'))
    mkdir(os.path.join(args.output_path, 'val'))

    # Create train dataset
    converter = LINCDatasetConverter()
    converter.convert_to_coco(args.input_path, args.output_path, 'train', args.trim_to, args.overfit)
    with open(os.path.join(args.output_path, 'train.json'), 'w') as f:
        json.dump(coco_train, f)

    # Create validation dataset
    converter.convert_to_coco(args.input_path, args.output_path, 'val', args.trim_to, args.overfit)
    with open(os.path.join(args.output_path, 'val.json'), 'w') as f:
        json.dump(coco_val, f)

    # Save text labels to file
    with open(os.path.join(args.output_path, 'labels.json'), 'w') as f:
        json.dump(converter.category_order_for_label, f)
