import numpy as np
import torch
import random

from torchvision.transforms import functional as F


def _flip_coco_person_keypoints(kps, width):
    flip_inds = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
    flipped_data = kps[:, flip_inds]
    flipped_data[..., 0] = width - flipped_data[..., 0]
    # Maintain COCO convention that if visibility == 0, then x, y = 0
    inds = flipped_data[..., 2] == 0
    flipped_data[inds] = 0
    return flipped_data


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class RandomHorizontalFlip(object):
    def __init__(self, prob, label_names):
        self.prob = prob
        self.label_names = label_names
        self.label_flipper = {
            # Left -> Right
            "cv-dl": "cv-dr",
            "cv-sl": "cv-sr",
            "ear-dl-l": "ear-dr-r",
            "ear-dr-l": "ear-dl-r",
            "ear-fl": "ear-fr",
            "ear-sl": "ear-sr",
            "eye-dl-l": "eye-dr-r",
            "eye-dr-l": "eye-dl-r",
            "eye-fl": "eye-fr",
            "eye-sl": "eye-sr",
            "nose-dl": "nose-dr",
            "nose-sl": "nose-sr",
            "whisker-dl": "whisker-dr",
            "whisker-sl": "whisker-sr",

            # Right -> Left
            "cv-dr": "cv-dl",
            "cv-sr": "cv-sl",
            "ear-dl-r": "ear-dr-l",
            "ear-dr-r": "ear-dl-l",
            "ear-fr": "ear-fl",
            "ear-sr": "ear-sl",
            "eye-dl-r": "eye-dr-l",
            "eye-dr-r": "eye-dl-l",
            "eye-fr": "eye-fl",
            "eye-sr": "eye-sl",
            "nose-dr": "nose-dl",
            "nose-sr": "nose-sl",
            "whisker-dr": "whisker-dl",
            "whisker-sr": "whisker-sl",
        }

    def __call__(self, image, target):
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            image = image.flip(-1)

            # Flip boxes
            bbox = target["boxes"]
            bbox[:, [0, 2]] = width - bbox[:, [2, 0]]
            target["boxes"] = bbox

            # Flip some spacially aware labels
            target["labels"] = self._flip_spacially_aware_linc_labels(target["labels"])

            if "masks" in target:
                target["masks"] = target["masks"].flip(-1)
            if "keypoints" in target:
                keypoints = target["keypoints"]
                keypoints = _flip_coco_person_keypoints(keypoints, width)
                target["keypoints"] = keypoints
        return image, target

    def _flip_spacially_aware_linc_labels(self, labels):
        flipped_labels = torch.empty_like(labels)
        text_labels = self.label_names[labels - 1] if len(labels) > 1 else [self.label_names[labels - 1]]

        for idx, (t, l) in enumerate(zip(text_labels, labels)):
            try:
                flipped_text_label = self.label_flipper[t]
                new_label = np.where(self.label_names == flipped_text_label)[0] + 1
                flipped_labels[idx] = int(new_label)
            except KeyError:
                # Same labels aren't left/right aware, so we don't flip them
                flipped_labels[idx] = l

        return flipped_labels


class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target
