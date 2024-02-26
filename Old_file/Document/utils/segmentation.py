import numpy as np
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt


def get_only_one_document(image_list):
     """
    This function takes a list of binary mask and returns the index of the image with the largest contour.

    Args:
        image_list (list): List of binary mask.

    Returns:
        i """

    max_contour_area = 0
    max_contour_img_index = 0

    for i, img in enumerate(image_list):
        # Chuyển hình ảnh sang ảnh xám
        img = img.astype(np.uint8)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Tìm tất cả các contour trong hình ảnh
        contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Nếu không tìm thấy contour nào, tiếp tục với hình ảnh tiếp theo
        if not contours:
            continue

        # Tìm contour lớn nhất trong hình ảnh
        largest_contour = max(contours, key=cv2.contourArea)

        # Tính diện tích của contour lớn nhất
        largest_contour_area = cv2.contourArea(largest_contour)

        # Nếu contour này lớn hơn contour lớn nhất hiện tại, cập nhật contour lớn nhất
        if largest_contour_area > max_contour_area:
            max_contour_area = largest_contour_area
            max_contour_img_index = i

    return max_contour_img_index



def binary_mask(model, path_img):
    """
    This function uses YOLOv8 for image segmentation and returns a list of binary masks and original segmented images.

    Parameters:
        model: YOLOv8 segmentation model.
        path_img: Path to the input image.

    Returns:
        mask_list: List of binary masks.
        background_list: List of original segmented images.
    """
    # Read the input image
    img = cv2.imread(path_img)

    # Padding
    black_color = [0, 0, 0]
    white_color = [255, 255, 255]
    padding_size = 250
    img = cv2.copyMakeBorder(img, padding_size, padding_size, padding_size, padding_size, cv2.BORDER_CONSTANT, value=black_color)

    # Get predictions from the YOLO model
    predict = model(img, conf=0.7)
    mask_list = []
    background_list = []

    for mask_index in range(len(predict[0].masks)):
        # Get raw mask
        mask_raw = predict[0].masks[mask_index].cpu().data.numpy().transpose(1, 2, 0)
        # Convert mask to 3-channel color
        mask_3channel = cv2.merge((mask_raw, mask_raw, mask_raw))
        # Resize the mask to match the size of the original image
        h2, w2, c2 = predict[0].orig_img.shape
        mask = cv2.resize(mask_3channel, (w2, h2))

        # Define the brightness range in the HSV color space
        lower_black = np.array([0, 0, 0])
        upper_black = np.array([0, 0, 1])

        # Create a mask based on the brightness range
        mask = cv2.inRange(mask, lower_black, upper_black)

        # Invert the mask to get everything except black
        mask = cv2.bitwise_not(mask)

        # Apply the mask to the original image
        masked = cv2.bitwise_and(predict[0].orig_img, predict[0].orig_img, mask=mask)
        background_list.append(masked)

        # Change all black pixels to white
        result_image = np.where(np.all(masked == black_color, axis=-1, keepdims=True), masked, white_color)
        mask_list.append(result_image)

    return mask_list, background_list
