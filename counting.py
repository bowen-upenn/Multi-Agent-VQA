import torch
from torch.cuda.amp import autocast
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
import cv2
import numpy as np

# from CLIP_Count.util.constant import SCALE_FACTOR
SCALE_FACTOR = 100

preprocess = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        # T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def query_clip_count(device, image, clip_count, prompts, verbose=False, save_heat_map=True):
    # Prepare the image
    image = Image.fromarray(image)
    image = preprocess(image).float().unsqueeze(0)

    # Perform inference
    with torch.no_grad():
        with autocast():
            output = clip_count(image.to(device), prompts)[0]
        pred_cnt = torch.sum(output / SCALE_FACTOR).item()
        count = int(round(pred_cnt))

        if save_heat_map:
            transform = T.ToPILImage()
            pil_img = transform(output)
            pil_img.save("heat_map.jpg", "JPEG")
            pil_img = transform(image[0])
            pil_img.save("original_image.jpg", "JPEG")

            heat_map_overlay = draw_heat_map(output, image)
            pil_img = transform(heat_map_overlay)
            pil_img.save("heat_map_overlay.jpg", "JPEG")

        if verbose:
            print('[Reattempted Answer] ' + str(count))

    return '[Reattempted Answer] ' + str(count)


def draw_heat_map(output, img):
    pred_density = output.detach().cpu().numpy()
    # normalize
    pred_density = pred_density / pred_density.max()
    pred_density_write = 1. - pred_density
    pred_density_write = cv2.applyColorMap(np.uint8(255 * pred_density_write), cv2.COLORMAP_JET)
    pred_density_write = pred_density_write / 255.

    img = TF.resize(img, (384))
    img = img.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()

    heatmap_pred = 0.33 * img + 0.67 * pred_density_write
    heatmap_pred = heatmap_pred / heatmap_pred.max()
    return heatmap_pred