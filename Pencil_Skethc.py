import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


def pencil_sketch(image_path, blur_kernel=21):
    """
    Convert an image to a pencil sketch effect.

    Args:
        image_path (str): Path to input image
        blur_kernel (int): Gaussian blur kernel size (must be odd)

    Returns:
        tuple: (original_rgb, sketch) or (None, None) if error
    """
    try:
        if blur_kernel % 2 == 0:
            raise ValueError("Blur kernel size should be an odd number")

        # Step 1: Load image
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError("Image not found or invalid format")

        # Convert BGR → RGB for display
        original_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Step 2: Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Step 3: Invert grayscale image
        inverted = 255 - gray

        # Step 4: Apply Gaussian blur
        blurred = cv2.GaussianBlur(inverted, (blur_kernel, blur_kernel), 0)

        # Step 5: Invert blurred image
        inverted_blur = 255 - blurred

        # Step 6: Divide and scale (Dodge blend)
        sketch = gray.astype(float) / (inverted_blur.astype(float) + 1e-6) * 256
        sketch = np.clip(sketch, 0, 255).astype(np.uint8)

        return original_rgb, sketch

    except Exception as e:
        print(f"Error: {e}")
        return None, None


def display_result(original, sketch, save_path=None):
    """
    Display original and sketch side-by-side.

    Args:
        original: Original image (RGB)
        sketch: Sketch image (grayscale)
        save_path (str): Optional path to save the sketch
    """
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(original)
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(sketch, cmap="gray")
    plt.title("Pencil Sketch")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

    if save_path:
        cv2.imwrite(save_path, sketch)
        print(f"Sketch saved to: {save_path}")


def main():
    """
    Main function to run the pencil sketch converter.
    """
    image_path = input("Enter image path: ").strip()
    blur_kernel = input("Enter blur kernel size: ").strip()

    if blur_kernel == "":
        blur_kernel = 21
    else:
        blur_kernel = int(blur_kernel)

    original, sketch = pencil_sketch(image_path, blur_kernel)

    if original is not None:
        output_dir = "output_sketches"
        os.makedirs(output_dir, exist_ok=True)

        output_path = os.path.join(
            output_dir, os.path.basename(image_path)
        )

        display_result(original, sketch, output_path)


if __name__ == "__main__":
    main()
