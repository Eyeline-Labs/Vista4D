import cv2
import numpy as np
from scipy import ndimage
import torch
from torch import nn
from tqdm.auto import tqdm


def contract_mask(mask, radius=4, iterations=3, desc=None):
    mask = mask.copy()
    iterator = range(mask.shape[0])
    if desc is not None:
        iterator = tqdm(iterator, desc=desc, leave=False)
    for i in iterator:
        mask[i] = cv2.erode(
            mask[i].astype(np.uint8), np.ones((radius, radius), np.uint8), iterations=iterations,
        ).astype(np.bool_)
    return mask


def get_depths_outliers(depths, mask, mode="pool", desc=None):
    if mode == "gaussian":
        return get_depths_outliers_gaussian(depths, mask, sigma=4, threshold=0.1, desc=desc)
    elif mode == "pool":
        return get_depths_outliers_pool(depths, mask, window_size=5, threshold=0.05, eps=1e-6)
    raise NotImplementedError(f"Unrecognized depth outlier mode `{mode}`.")


def get_depths_outliers_gaussian(depths, mask, sigma=4, threshold=0.5, desc=None):  # Mask-normalized
    depths_masked = depths.copy()
    depths_masked[~mask] = 0.0
    masks_float = mask.astype(depths.dtype)

    frame_iter = range(depths.shape[0])
    if desc is not None:
        frame_iter = tqdm(list(frame_iter), desc=desc, leave=False)
    depths_masked_blurred = np.stack([
        ndimage.gaussian_filter(depths_masked[i], sigma=sigma, mode="constant", cval=0)
        for i in frame_iter
    ], axis=0)
    masks_blurred = np.stack([
        ndimage.gaussian_filter(masks_float[i], sigma=sigma, mode="constant", cval=0)
        for i in range(depths.shape[0])
    ], axis=0)
    masks_blurred[masks_blurred == 0.0] = 1.0  # Avoid division by zero
    depths_masked_blurred = depths_masked_blurred / masks_blurred  # Normalize the blurred result

    inliers = (
        (pow(2, -threshold) * depths_masked_blurred <= depths_masked) &
        (depths_masked <= pow(2, threshold) * depths_masked_blurred)
    )
    outliers = ~inliers & mask
    return outliers


def get_depths_outliers_pool(depths, mask, window_size=5, threshold=0.05, eps=1e-6):  # Modified from GEN3C
    original_shape = depths.shape
    is_numpy = isinstance(depths, np.ndarray)
    if is_numpy:
        depths = torch.from_numpy(depths)
        mask = torch.from_numpy(mask)

    assert window_size % 2 == 1, f"Window size ({window_size}) must be odd"

    if depths.dim() == 3:   # Input shape f h w
        depths = depths[:, None]
    elif depths.dim() == 4:  # Already has shape f 1 h w
        pass
    else:
        raise ValueError("Depth tensor must be of shape (f h w) or (f 1 h w)")

    if mask.dim() == 3:   # Input shape f h w
        mask = mask[:, None]
    elif mask.dim() == 4:  # Already has shape f 1 h w
        pass
    else:
        raise ValueError("Mask tensor must be of shape (f h w) or (f 1 h w)")

    local_max = nn.functional.max_pool2d(depths, kernel_size=window_size, stride=1, padding=window_size // 2)
    local_min = -nn.functional.max_pool2d(-depths, kernel_size=window_size, stride=1, padding=window_size // 2)
    local_mean = nn.functional.avg_pool2d(depths, kernel_size=window_size, stride=1, padding=window_size // 2)
    ratio = (local_max - local_min) / (local_mean + eps)
    inliers = (ratio < threshold) & (depths > 0)

    # Ignored masked values by setting them to -inf for max and inf for min
    local_max = nn.functional.max_pool2d(
        depths.masked_fill(~mask, -float("inf")), kernel_size=window_size, stride=1, padding=window_size // 2,
    )
    local_min = -nn.functional.max_pool2d(
        -depths.masked_fill(~mask, float("inf")), kernel_size=window_size, stride=1, padding=window_size // 2,
    )
    # Ignore masked values by dividing/normalizing by the mask average pool
    local_mean = (
        nn.functional.avg_pool2d(depths * mask, kernel_size=window_size, stride=1, padding=window_size // 2) /
        (nn.functional.avg_pool2d(mask.float(), kernel_size=window_size, stride=1, padding=window_size // 2) + eps)
    )
    ratio = (local_max - local_min) / (local_mean + eps)
    inliers = (ratio < threshold) & (depths > 0)

    inliers = inliers.reshape(*original_shape)
    mask = mask.reshape(*original_shape)
    if is_numpy:
        inliers = inliers.numpy().astype(np.bool_)
        mask = mask.numpy().astype(np.bool_)
    return ~inliers & mask
