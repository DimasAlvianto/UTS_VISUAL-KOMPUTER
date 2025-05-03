import cv2
import numpy as np
import matplotlib.pyplot as plt
from google.colab import files

def read_and_show_image(image_path, title="Image"):
    """Reads an image and displays it using matplotlib.pyplot."""
    try:
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Could not read image from {image_path}")
            return None

        # OpenCV reads images in BGR format, matplotlib uses RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        plt.imshow(img_rgb)
        plt.title(title)
        plt.axis('off')  # Turn off axis labels
        plt.show()

        return img
    except Exception as e:
        print(f"An error occurred while reading or displaying the image: {e}")
        return None

def convert_to_grayscale(image):
    """Converts an RGB image to grayscale and displays it."""
    if image is None:
        return None
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    plt.imshow(gray_image, cmap='gray')  # Use 'gray' colormap
    plt.title("Grayscale Image")
    plt.axis('off')
    plt.show()
    return gray_image

def apply_thresholding(image, threshold_value):
    """Applies thresholding to a grayscale image and displays it."""
    if image is None:
        return None
    _, thresholded_image = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)
    plt.imshow(thresholded_image, cmap='gray')
    plt.title("Thresholded Image")
    plt.axis('off')
    plt.show()
    return thresholded_image

def apply_complement(image):
    """Applies the complement operation and displays the result."""
    if image is None:
        return None
    complemented_image = cv2.bitwise_not(image)
    plt.imshow(complemented_image, cmap='gray')
    plt.title("Complemented Image")
    plt.axis('off')
    plt.show()
    return complemented_image

def apply_morphological_operations(image):
    """Applies morphological operations and displays each result."""
    if image is None:
        return None

    # Filling Holes
    image_copy = image.copy()
    h, w = image_copy.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(image_copy, mask, (0, 0), 255)
    filled_holes_image = cv2.bitwise_not(image_copy) | image
    plt.imshow(filled_holes_image, cmap='gray')
    plt.title("Filled Holes Image")
    plt.axis('off')
    plt.show()

    # Opening
    kernel = np.ones((5, 5), np.uint8)
    opened_image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    plt.imshow(opened_image, cmap='gray')
    plt.title("Opened Image")
    plt.axis('off')
    plt.show()

    # Erosion
    eroded_image = cv2.erode(image, kernel, iterations=1)
    plt.imshow(eroded_image, cmap='gray')
    plt.title("Eroded Image")
    plt.axis('off')
    plt.show()

    return eroded_image

def replace_background(original_image, binary_mask, background_image):
    """Replaces the background of the original image and displays the final result."""
    if original_image is None or binary_mask is None or background_image is None:
        return None

    background_resized = cv2.resize(background_image, (original_image.shape[1], original_image.shape[0]))

    if len(binary_mask.shape) == 2:
        binary_mask = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)

    mask_inv = cv2.bitwise_not(binary_mask)
    foreground = cv2.bitwise_and(original_image, binary_mask)

    # Resize background_resized to match foreground's shape
    background_resized = cv2.resize(background_resized, (foreground.shape[1], foreground.shape[0]))
    background = cv2.bitwise_and(background_resized, mask_inv)

    final_image = cv2.add(foreground, background)
    final_image_rgb = cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB)  # Convert back to RGB for matplotlib
    plt.imshow(final_image_rgb)
    plt.title("Final Image")
    plt.axis('off')
    plt.show()
    return final_image

# Main Execution
try:
    # Upload and process the main image
    uploaded_main = files.upload()
    main_image_path = list(uploaded_main.keys())[0]
    original_image = read_and_show_image(main_image_path, "Original Image")

    if original_image is not None:
        gray_image = convert_to_grayscale(original_image)
        if gray_image is not None:
            thresholded_image = apply_thresholding(gray_image, 128)
            if thresholded_image is not None:
                complemented_image = apply_complement(thresholded_image)
                if complemented_image is not None:
                    morphological_image = apply_morphological_operations(complemented_image)

                    # Upload and process the background image
                    uploaded_bg = files.upload()
                    background_image_path = list(uploaded_bg.keys())[0]
                    background_image = read_and_show_image(background_image_path, "Background Image")

                    if background_image is not None:
                        replace_background(original_image, complemented_image, background_image)

except Exception as e:
    print(f"An error occurred during the main execution: {e}")
