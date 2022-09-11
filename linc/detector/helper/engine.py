import math
import sys
import time
import torch

import torchvision.models.detection.mask_rcnn

from .coco_utils import get_coco_api_from_dataset
from .coco_eval import CocoEvaluator
from . import utils


def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq, writer, label_names):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ", device=device)
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    first_step_of_epoch = True
    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        if first_step_of_epoch:
            # Write input image and target to summary. The model modifies its input images in place it
            # seems (normalization), so we save them before running them through model
            image_with_boxes = utils.draw_boxes(
                images[0], targets[0]['boxes'], targets[0]['labels'], label_names, vert_size=300,
                image_id=int(targets[0]['image_id']),
            )
            writer.add_image('Target image', image_with_boxes, global_step=epoch)
            first_step_of_epoch = False

        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    writer.add_scalar('Loss', losses_reduced, global_step=epoch)


def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types


@torch.no_grad()
def evaluate(model, data_loader, epoch, writer, draw_threshold, label_names, num_draw_predictions, device):
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ", device=device)
    header = 'Test:'

    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)

    images_evaluated, images_written_to_summary = 0, 0
    for image, targets in metric_logger.log_every(data_loader, 100, header):
        # The model modifies its input images in place it seems (normalization), so we save them
        # for drawing before running them through model.
        pre_model_image = image[0]

        image = list(img.to(device) for img in image)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(image)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

        # Write evaluated images to summary
        vert_size = 400
        if images_evaluated % int(round(len(data_loader) / num_draw_predictions)) == 0:
            scores = outputs[0]['scores']
            top_scores_filter = scores > draw_threshold
            top_scores = scores[top_scores_filter]
            top_boxes = outputs[0]['boxes'][top_scores_filter]
            top_labels = outputs[0]['labels'][top_scores_filter]
            if len(top_scores) > 0:
                # Draw targets. Convert targets to cpu for drawing first.
                targets = [{k: v.to('cpu') for k, v in t.items()} for t in targets]
                image_with_boxes = utils.draw_boxes(
                    pre_model_image, targets[0]['boxes'], targets[0]['labels'], label_names,
                    vert_size=vert_size
                )

                # Image was scaled in previos step, but predictions still need to be scaled
                scaled_top_boxes = top_boxes * (1 / (pre_model_image.shape[1] / vert_size))

                # Draw predictions
                image_with_boxes = utils.draw_boxes(
                    image_with_boxes, scaled_top_boxes, top_labels, label_names, scores, vert_size=vert_size,
                    image_id=int(targets[0]['image_id'])
                )

                writer.add_image(
                    f'Eval image {images_written_to_summary}', image_with_boxes, global_step=epoch
                )
            images_written_to_summary += 1
        images_evaluated += 1

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    torch.set_num_threads(n_threads)

    # Write evaluation results to summary
    writer.add_scalar('mAP_0.50-0.95', coco_evaluator.bbox_map_50_95, global_step=epoch)
    writer.add_scalar('mAP_0.50', coco_evaluator.bbox_map_50, global_step=epoch)
    writer.add_scalar('mAP_0.75', coco_evaluator.bbox_map_75, global_step=epoch)
    return coco_evaluator
