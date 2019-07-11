import sys
import time
import torch
from PIL import Image
import torchvision
from utils import draw_boxes

draw_confidence_threshold = 0.5

to_tensor = torchvision.transforms.ToTensor()
convert_to_pil = torchvision.transforms.ToPILImage()


@torch.no_grad()
def main(image_path, model_path, output_path, cpu):
    device = 'cuda' if torch.has_cuda and not cpu else 'cpu'
    print(f"Running inference on {device} device")

    print('Loading image... ', end='', flush=True)
    image = to_tensor(Image.open(image_path)).to(device)
    print('Done.')

    print('Loading checkpoint from hardrive... ', end='', flush=True)
    checkpoint = torch.load(model_path, map_location=device)
    label_names = checkpoint['label_names']
    print('Done.')

    print('Building model and loading checkpoint into it... ', end='', flush=True)
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        num_classes=len(label_names) + 1, pretrained_backbone=False
    )
    model.to(device)

    model.load_state_dict(checkpoint['model'])
    model.eval()
    print('Done.')

    print('Running image through model... ', end='', flush=True)
    tic = time.time()
    outputs = model([image])
    toc = time.time()
    print(f'Done in {toc - tic:.2f} seconds!')

    print(f'Saving image to {output_path}... ', end='', flush=True)
    scores = outputs[0]['scores']
    top_scores_filter = scores > draw_confidence_threshold
    top_scores = scores[top_scores_filter]
    top_boxes = outputs[0]['boxes'][top_scores_filter]
    top_labels = outputs[0]['labels'][top_scores_filter]
    if len(top_scores) > 0:
        image_with_boxes = draw_boxes(
            image.cpu(), top_boxes, top_labels.cpu(), label_names, scores
        )
    else:
        print("The model didn't find any object it feels confident about enough to show")
        exit()
    pil_picture = convert_to_pil(image_with_boxes)
    pil_picture.save(output_path)
    print('Done.')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='LINC Detector Prediction')
    parser.add_argument('input_image_path', help='Path of image to run prediction on')
    parser.add_argument('model_checkpoint_path', help='Path of checkpoint of model to load into network')
    parser.add_argument(
        '--output_image_path', default='out.jpg', help='Path where output image should be saved'
    )
    parser.add_argument("--cpu", dest="cpu", help="Force model to use CPU", action="store_true")
    args = parser.parse_args()

    main(args.input_image_path, args.model_checkpoint_path, args.output_image_path, args.cpu)
