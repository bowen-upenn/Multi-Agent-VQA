import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2


def calculate_iou_batch(a, b):
    """
    Vectorized calculation of IoU for pairs of bounding boxes in a and b.
    Parameters:
    - a: PyTorch tensor of shape (N, 4), representing bounding boxes.
    - b: PyTorch tensor of shape (M, 4), representing bounding boxes.
    Returns:
    - iou: PyTorch tensor of shape (M, N), IoU values.
    """
    # Expand dimensions to support broadcasting: (N, 1, 4) with (1, M, 4)
    a = a.unsqueeze(1)  # Shape: (N, 1, 4)
    b = b.unsqueeze(0)  # Shape: (1, M, 4)
    print('a', a.shape, a, 'b', b.shape, b)

    # Calculate intersection coordinates
    max_xy = torch.min(a[..., 2:], b[..., 2:])
    min_xy = torch.max(a[..., :2], b[..., :2])
    inter = torch.clamp((max_xy - min_xy), min=0)
    intersection = inter[..., 0] * inter[..., 1]

    # Calculate areas
    a_area = (a[..., 2] - a[..., 0]) * (a[..., 3] - a[..., 1])
    b_area = (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])

    # Calculate union
    union = a_area + b_area - intersection

    # Compute IoU
    iou = intersection / union
    print('iou', iou.shape)

    return iou


def filter_boxes_pytorch(a, b, iou_threshold=0.5):
    """
    Filters boxes in b based on IoU threshold with boxes in a using PyTorch.
    Parameters:
    - a, b: PyTorch tensors of shapes (N, 4) and (M, 4) respectively.
    - iou_threshold: float, threshold for filtering.
    Returns:
    - filtered_b: PyTorch tensor of filtered bounding boxes from b.
    """
    iou = calculate_iou_batch(a, b)  # Shape: (M, N)
    # Check if any IoU value exceeds the threshold for each box in b
    max_iou, _ = torch.max(iou, dim=0)
    keep = max_iou > iou_threshold
    print('b', b.shape, 'keep', keep.shape, keep, 'iou', max_iou)
    return b[keep]


def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    # ax = plt.gca()
    # ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    # ax.imshow(img)
    plt.imsave('masks', img)



