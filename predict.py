import sys
import torch
from PIL import Image
import torchvision
from utils import draw_boxes

draw_confidence_threshold = 0.5

to_tensor = torchvision.transforms.ToTensor()
convert_to_pil = torchvision.transforms.ToPILImage()


def main(image_path, model_path, output_path):
    device = 'cuda' if torch.has_cuda else 'cpu'
    image = to_tensor(Image.open(image_path)).to(device)

    print('Loading checkpoint from hardrive... ', end='')
    checkpoint = torch.load(model_path, map_location=device)
    label_names = checkpoint['label_names']
    print('Done!')

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(num_classes=len(label_names) + 1)
    model.to(device)

    model.load_state_dict(checkpoint['model'])
    model.eval()

    print('Running image through model... ', end='')
    outputs = model([image])
    print('Done!')

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


if __name__ == '__main__':
    image_path = sys.argv[1]
    model_path = sys.argv[2]
    output_path = sys.argv[3]
    main(image_path, model_path, output_path)
