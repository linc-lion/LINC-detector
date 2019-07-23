import torch
from torchvision.ops.boxes import nms

# We assign each class a new temporal class during this nms operation.
# Access each class by its possition in this tensor, and the number
# stored corresponds to the temporal class we'll assign to each class
# during nms.
CLASS_MERGER = torch.LongTensor([
    0,  # Not used, 0 is reserved for the 'background' class!
    1,  # cv-dl
    1,  # cv-dr
    1,  # cv-f
    1,  # cv-sl
    1,  # cv-sr
    2,  # ear-dl-l
    3,  # ear-dl-r
    2,  # ear-dr-l
    3,  # ear-dr-r
    2,  # ear-fl
    3,  # ear-fr
    2,  # ear-sl
    3,  # ear-sr
    4,  # eye-dl-l
    5,  # eye-dl-r
    4,  # eye-dr-l
    5,  # eye-dr-r
    4,  # eye-fl
    5,  # eye-fr
    4,  # eye-sl
    5,  # eye-sr
    6,  # nose-dl
    6,  # nose-dr
    6,  # nose-f
    6,  # nose-sl
    6,  # nose-sr
    7,  # whisker-dl
    7,  # whisker-dr
    7,  # whisker-f
    7,  # whisker-sl
    7,  # whisker-sr
    8   # ws
])


def batched_nms_linc(boxes, scores, idxs, iou_threshold):
    """
    Performs non-maximum suppression in a batched fashion.
    Each index value correspond to a category, and NMS
    will not be applied between elements of different categories.

    This is a custom implementation for the LINC project which is
    aware of what LINC classes are exclusive. For example, a left
    eye, and a left diagonal eye, will be considered as the same class
    when applying this nms filter, as they both can't be in the same
    part of the image at the same time on top of each other.

    Arguments:
        boxes (Tensor[N, 4]): boxes where NMS will be performed. They
            are expected to be in (x1, y1, x2, y2) format
        scores (Tensor[N]): scores for each one of the boxes
        idxs (Tensor[N]): indices of the categories for each
            one of the boxes.
        iou_threshold (float): discards all overlapping boxes
            with IoU < iou_threshold
    Returns:
        keep (Tensor): int64 tensor with the indices of
            the elements that have been kept by NMS, sorted
            in decreasing order of scores
    """
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=boxes.device)

    idxs_merged = torch.empty_like(idxs)
    for i, idx in enumerate(idxs):
        idxs_merged[i] = CLASS_MERGER[int(idx)]

    # strategy: in order to perform NMS independently per class.
    # we add an offset to all the boxes. The offset is dependent
    # only on the class idx, and is large enough so that boxes
    # from different classes do not overlap
    max_coordinate = boxes.max()
    offsets = idxs_merged.to(boxes) * (max_coordinate + 1)
    boxes_for_nms = boxes + offsets[:, None]
    keep = nms(boxes_for_nms, scores, iou_threshold)
    return keep
