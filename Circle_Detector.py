import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


def preprocess_image(image_path):
    """
    Load and preprocess image for circle detection .

    Args :
    image_path : Path to input image

    Returns :
    tuple : ( original_color , preprocessed_gray ) or (None , None )
    """
    image = cv2.imread(image_path)
    if image is None:
        return None, None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    return image, blurred


def detect_circles(gray_image, dp=1, minDist=50, param1=50, param2=30, minRadius=10, maxRadius=100):
    """
    Detect circles using Hough Circle Transform .

    Args :
    gray_image : Preprocessed grayscale image
    dp: Inverse accumulator resolution ratio
    minDist : Minimum distance between circle centers
    param1 : Upper Canny threshold
    param2 : Accumulator threshold
    minRadius : Minimum circle radius
    maxRadius : Maximum circle radius

    Returns :
    numpy array of circles (x, y, radius ) or None
    """
    if gray_image is None:
        return None
    circles = cv2.HoughCircles(gray_image, cv2.HOUGH_GRADIENT, dp, minDist, param1=param1, param2=param2, minRadius=minRadius, maxRadius=maxRadius)
    if circles is not None:
        circles = np.uint16(np.around(circles[0]))
    return circles


def visualize_circles(image, circles, save_path=None):
    """
    Draw detected circles on image and display .

    Args :
    image : Original color image
    circles : Array of detected circles
    save_path : Optional path to save annotated image
    """
    result_circles = image.copy()
    if circles is not None:
        for i in circles[0, :]: 
            cv2.circle(result_circles, (i[0], i[1]), i[2], (0, 255, 0), 2)
            cv2.circle(result_circles, (i[0], i[1]), 2, (0, 0, 255), 3)
            cv2.putText(result_circles, f"{i}:{i[2]}", (i[0] - 20, i[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    if save_path:
        cv2.imwrite(save_path, result_circles)
    combined = np.hstack((image, result_circles))
    plt.imshow(cv2.cvtColor(combined, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()


def calculate_statistics(circles):
    """
    Calculate and display statistics about detected circles .

    Args :
    circles : Array of detected circles

    Returns :
    dict : Statistics dictionary
    """
    if circles is None or len(circles) == 0:
        return {"count": 0, "min": 0, "max": 0, "avg": 0}
    radii = [r for (_, _, r) in circles]
    return {"count": len(circles), "min": int(np.min(radii)), "max": int(np.max(radii)), "avg": float(np.mean(radii))}


def main():
    """ Main function ."""
    image_path = "test_images/test1.jpg"
    result_path = "results/result3.jpg"
    os.makedirs("results", exist_ok=True)
    image, gray = preprocess_image(image_path)
    circles = detect_circles(gray)
    visualize_circles(image, circles, result_path)
    statistics = calculate_statistics(circles)
    print(statistics)


if __name__ == "__main__":
    main()
