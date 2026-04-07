import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Load image even from path or it's np array 
def load_image(image) -> np.ndarray:
    # Handle both path and array input
    if isinstance(image, str):
        original_image = cv.imread(image)
        input_source = image
    elif isinstance(image, np.ndarray):
        original_image = image.copy()
        input_source = "numpy array"
    else:
        raise ValueError("Input must be a file path (str) or NumPy array.")

    return original_image


def debug_preprocessing_step(
    image_path: str,
    processing_function,
    save_path: str = None,
    **kwargs
) -> np.ndarray:
   

    # Step 1: load the original image
    original_image = load_image(image_path)

    # Step 2: apply the preprocessing function
    processed_image = processing_function(original_image, **kwargs)

    # Step 3: validate output
    if not isinstance(processed_image, np.ndarray):
        raise ValueError(
            f"The processing function '{processing_function.__name__}' "
            f"did not return a NumPy array."
        )

    # Step 4: print useful debug information
    print("=" * 60)
    print(f"Processing function : {processing_function.__name__}")
    print(f"Input image path    : {image_path}")
    print(f"Original shape      : {original_image.shape}")
    print(f"Processed shape     : {processed_image.shape}")
    print(f"Original dtype      : {original_image.dtype}")
    print(f"Processed dtype     : {processed_image.dtype}")
    print("=" * 60)

    # Step 5: prepare images for matplotlib display
    # OpenCV loads color images in BGR, but matplotlib expects RGB.
    if len(original_image.shape) == 3 and original_image.shape[2] == 3:
        original_display = cv.cvtColor(original_image, cv.COLOR_BGR2RGB)
    else:
        original_display = original_image

    if len(processed_image.shape) == 3 and processed_image.shape[2] == 3:
        processed_display = cv.cvtColor(processed_image, cv.COLOR_BGR2RGB)
        processed_cmap = None
    else:
        processed_display = processed_image
        processed_cmap = "gray"

    original_cmap = None if len(original_display.shape) == 3 else "gray"

    # Step 6: visualize side-by-side
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(original_display, cmap=original_cmap)
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(processed_display, cmap=processed_cmap)
    plt.title(f"Processed: {processing_function.__name__}")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

    # Step 7: save processed image if requested
    if save_path is not None:
        save_dir = os.path.dirname(save_path)

        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)

        cv.imwrite(save_path, processed_image)
        print(f"Processed image saved to: {save_path}")

    return processed_image