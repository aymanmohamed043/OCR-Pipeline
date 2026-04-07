import cv2 as cv 
import numpy as np



def to_grayscale(image: np.ndarray) -> np.ndarray:
    if image is None:
        raise ValueError("Input image is None. Check that the image was loaded correctly.")

    if len(image.shape) == 2:
        return image

    if len(image.shape) == 3 and image.shape[2] == 3:
        gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        return gray_image

    raise ValueError(
        f"Unsupported image shape: {image.shape}. Expected (H, W) or (H, W, 3)."
    )

