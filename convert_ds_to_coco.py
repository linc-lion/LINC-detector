import os
import xml.etree.ElementTree as ET
import collections
import hashlib
import json
from PIL import Image


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

CATEGORIES_TO_IGNORE = [0, 7, 8]


def parse_voc_xml(node):
    voc_dict = {}
    children = list(node)
    if children:
        def_dic = collections.defaultdict(list)
        for dc in map(parse_voc_xml, children):
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


def deterministic_hash(string):
    """Hashes to a 4 byte int deterministically."""
    md5 = hashlib.md5()
    md5.update(string.encode('utf-8'))
    return int.from_bytes(md5.digest()[:4], byteorder='big')


class ConvertLINC():
    num_of_classes = 9
    name_to_label = {
        # Markings
        'markings': 0,
        'ear-fl-marking': 0,
        'mouth-f-marking': 0,
        'eye-dr-l-marking': 0,
        'tooth-marking': 0,
        'nose-f-marking': 0,
        'nose-dl-marking': 0,
        'eye-fl-marking': 0,
        'ear-dr-l-marking': 0,
        'ear-f-r-marking': 0,
        'mouth-dr-marking': 0,
        'nose-dr-marking': 0,
        'mouth-sl-marking': 0,
        'ear-dr-marking': 0,
        'ear-sl-marking': 0,
        'eye-dl-l-marking': 0,
        'ear-dl-r-marking': 0,
        'mouth-dl-marking': 0,
        'full-body-markings': 0,
        'marking': 0,

        # CV
        'cv-sright': 1,
        'cv-r': 1,
        'cv-dr-r': 1,
        'cv-f': 1,
        'cv-l': 1,
        'cv-sr': 1,
        'cv-dr': 1,
        'cv-sl': 1,
        'cv-dl-r': 1,
        'cv-dr-l': 1,
        'cv-dl': 1,
        'cv-front': 1,

        # Nose
        'nose-dl': 2,
        'nose-dr': 2,
        'nose-l': 2,
        'nose-sl': 2,
        'nose-dl-l': 2,
        'nose-fl': 2,
        'nose-sr': 2,
        'nose': 2,
        'nose-f': 2,
        'nose-r': 2,
        'nose-dr-r': 2,
        'nose-slw': 2,

        # Ear
        'ear-sl': 3,
        'ear-sr-r': 3,
        'ear-dr-r-marking': 3,
        'ear-sr-marking': 3,
        'ear-fl': 3,
        'ear-dl-l-marking': 3,
        'ear-dr-r': 3,
        'ear-fr': 3,
        'ear-fr-marking': 3,
        'ear-dl-r': 3,
        'ear-dl-l': 3,
        'ear-dr': 3,
        'ear-sr-l': 3,
        'ear-f': 3,
        'ear-sr': 3,
        'ear-dl': 3,
        'ear-f-l': 3,
        'ear-dr-l': 3,
        'ear-f-r': 3,

        # Whisker Area
        'whikser-dr': 4,
        'whikser-dl': 4,
        'whisker-r': 4,
        'whisker-s': 4,
        'whisker-sr': 4,
        'whisker-dl': 4,
        'whisker-sl': 4,
        'whisker-l': 4,
        'whisker-dr': 4,
        'whiske-dr': 4,
        'whisker-f': 4,

        # Mouth
        'mouth-dr': 5,
        'mouth-dl': 5,

        # Eye
        'eye-dr': 6,
        'eye-dl-r': 6,
        'eye-d-l': 6,
        'eye-dr-r': 6,
        'eye-fl': 6,
        'eye-sl': 6,
        'eye-f-r': 6,
        'eye-sr-l': 6,
        'eye-sr-r': 6,
        'eye-fr': 6,
        'eye-f': 6,
        'eye-dl': 6,
        'eye-sr': 6,
        'eye-dr-l': 6,
        'eye-f-l': 6,
        'eye-dl-l': 6,

        # Whisker Spot
        'ws': 7,

        # Full Body
        'full-body': 8,

    }

    def convert_obj_to_coco_format(self, o, img_counter, obj_counter):
        annotation = {}
        bbox = o['bndbox']
        bbox = [float(bbox['xmin']), float(bbox['ymin']), float(bbox['xmax']), float(bbox['ymax'])]
        annotation['bbox'] = bbox
        annotation['image_id'] = img_counter
        annotation['area'] = (bbox[3] - bbox[1]) * (bbox[2] - bbox[0])
        annotation['iscrowd'] = 0
        annotation['category_id'] = self.name_to_label[o['name']]
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


def convert_to_coco(root, output_dir, image_set):
    converter = ConvertLINC()

    # Check existence of root
    if not os.path.isdir(root):
        raise RuntimeError('Dataset not found.')

    # Crawl sub-directories
    print(f"Crawling directories for {image_set} set...", end=' ')
    obj_counter = 1
    img_counter = 1
    for root, dirs, files in os.walk(root):
        dirs.sort()  # Lets make this deterministic so our ids are too. NOTE: Not sure if this works
        for file_name in [os.path.join(root, f) for f in files]:
            if os.path.splitext(file_name)[1] == '.xml':
                data = parse_voc_xml(ET.parse(file_name).getroot())

                # Ignore images with no objects in them
                try:
                    objects = data['annotation']['object']
                except KeyError:
                    continue
                objects = objects if type(objects) is list else [objects]
                image_path = os.path.join(root, data['annotation']['filename'])
                image_name = os.path.basename(image_path)
                img = Image.open(image_path)

                # Define train/val sets
                # Ignore image that don't belong in each set
                if image_set == 'train' and img_counter % train_val_ratio > 0:
                    coco_train['images'].append(
                        converter.convert_img_to_coco_and_save(
                            img_counter, image_name, img, os.path.join(output_dir, 'train')
                        )
                    )
                    for o in objects:
                        target = converter.convert_obj_to_coco_format(o, img_counter, obj_counter)
                        if target['category_id'] in CATEGORIES_TO_IGNORE:
                            continue
                        coco_train['annotations'].append(target)
                        obj_counter += 1
                elif image_set == 'val' and img_counter % train_val_ratio == 0:
                    coco_val['images'].append(
                        converter.convert_img_to_coco_and_save(
                            img_counter, image_name, img, os.path.join(output_dir, 'val')
                        )
                    )
                    for o in objects:
                        target = converter.convert_obj_to_coco_format(o, img_counter, obj_counter)
                        if target['category_id'] in CATEGORIES_TO_IGNORE:
                            continue
                        coco_val['annotations'].append(target)
                        obj_counter += 1

                img_counter += 1

    print(f"Done, got {img_counter} images.")


if __name__ == "__main__":
    output_folder = '/home/lalo/linc/coco_easy'
    input_folder = '/home/lalo/linc/Verified_Annotation/'

    convert_to_coco(input_folder, output_folder, 'train')
    with open(os.path.join(output_folder, 'train.json'), 'w') as f:
        json.dump(coco_train, f)
    convert_to_coco(input_folder, output_folder, 'val')
    with open(os.path.join(output_folder, 'val.json'), 'w') as f:
        json.dump(coco_val, f)
