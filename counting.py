import torch
from torch.cuda.amp import autocast
import torchvision.transforms as T
from PIL import Image

# from CLIP_Count.util.constant import SCALE_FACTOR
SCALE_FACTOR = 80

preprocess = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        # T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def query_clip_count(device, image, clip_count, prompts, verbose=False, save_heat_map=False):
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

        if verbose:
            print('[Reattempted Answer] ' + str(count))

    return '[Reattempted Answer] ' + str(count)