import cv2
import numpy as np
import matplotlib.pyplot as plt
from deskew import determine_skew
import math
from typing import Tuple, Union

#Rotate an image before segmentation
def rotate(image: np.ndarray, angle: float, background: Union[int, Tuple[int, int, int]]) -> np.ndarray:
    """
    Rotate an image by a given angle.

    Parameters:
        image: Input image (numpy array).
        angle: Rotation angle in degrees.
        background: Color of the background after rotation.

    Returns:
        Rotated image.
    """
    old_width, old_height = image.shape[:2]
    angle_radian = math.radians(angle)
    width = abs(np.sin(angle_radian) * old_height) + abs(np.cos(angle_radian) * old_width)
    height = abs(np.sin(angle_radian) * old_width) + abs(np.cos(angle_radian) * old_height)

    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    rot_mat[1, 2] += (width - old_width) / 2
    rot_mat[0, 2] += (height - old_height) / 2

    return cv2.warpAffine(image, rot_mat, (int(round(height)), int(round(width))), borderValue=background)

#Order four points 
def order_points(pts):
    """
    Order four points in the order: top-left, top-right, bottom-right, bottom-left.

    Parameters:
        pts: List of four points.

    Returns:
        Ordered list of four points.
    """
    rect = np.zeros((4, 2), dtype="float32")
    pts = np.array(pts)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect.astype("int").tolist()


def distance(point1, point2):
    """
    Calculate the Euclidean distance between two points.

    Parameters:
        point1: First point as a tuple (x, y).
        point2: Second point as a tuple (x, y).

    Returns:
        Euclidean distance between the two points.
    """
    return np.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2)


def find_dest(pts):
    """
    Find destination points for perspective transform.

    Parameters:
        pts: List of four points.

    Returns:
        Ordered list of destination points.
    """
    (tl, tr, br, bl) = pts
    # Calculate the width and height of the bounding box
    maxWidth = max(int(distance(br, bl)), int(distance(tr, tl)))
    maxHeight = max(int(distance(tr, br)), int(distance(tl, bl)))

    destination_corners = [[0, 0], [maxWidth, 0], [maxWidth, maxHeight], [0, maxHeight]]
    return order_points(destination_corners)

def expand(corners,epsilon):
    """
    Expand the rectangle defined by corners.

    Parameters:
        corners: List of four points representing the corners of the rectangle.
        epsilon: Expansion factor.

    Returns:
        Expanded corners.
    """
    # Destructure the corners into individual points: top-left (tl), top-right (tr),
    # bottom-right (br), and bottom-left (bl)
    (tl, tr, br, bl) = corners

    # Calculate the width and height of the bounding box
    maxWidth = max(int(distance(br, bl)), int(distance(tr, tl)))
    maxHeight = max(int(distance(tr, br)), int(distance(tl, bl)))

    # Increase the x-coordinate of the top-right and bottom-right points
    corners[1][0] += maxWidth/epsilon
    corners[2][0] += maxWidth/epsilon   

    # Decrease the x-coordinate of the top-left and bottom-left points
    corners[0][0] -= maxWidth/epsilon
    corners[3][0] -= maxWidth/epsilon
    # Increase the y-coordinate of the bottom-right and bottom-left points
    corners[2][1] += maxHeight/epsilon
    corners[3][1] += maxHeight/epsilon

    # Decrease the y-coordinate of the top-left and top-right points
    corners[0][1] -= maxHeight/epsilon
    corners[1][1] -= maxHeight/epsilon
    return corners

def extract(seg_img, mask, IMAGE_SIZE=384, BUFFER=100,expanded=True):
    """
    Extract a custom document using perspective transform.

    Parameters:
        seg_img: Segmentation image with black background.
        mask: Binary mask.
        IMAGE_SIZE: Size of the output image.
        BUFFER: Buffer for padding.

    Returns:
        Resulting document.
    """
    # Convert the mask to uint8 data type
    mask = mask.astype(np.uint8)
    # Convert the mask to grayscale
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    # Determine the skew angle of the mask
    angle = determine_skew(mask)
    # Rotate both the mask and segmentation image by the determined angle
    mask = rotate(mask, angle, (0, 0, 0))
    seg_img = rotate(seg_img, angle, (0, 0, 0))
    imH, imW = mask.shape
    # Resize the binary mask to the specified IMAGE_SIZE
    mask_rs = cv2.resize(mask, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_NEAREST)
    # Find the scale factors
    scale_x = imW / IMAGE_SIZE
    scale_y = imH / IMAGE_SIZE
    # Find edges using Canny edge detection
    canny = cv2.Canny(mask_rs, 225, 255)
    # Dilate the edges for better contour detection
    canny = cv2.dilate(canny, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    # Find contours in the binary mask
    contours, _ = cv2.findContours(canny, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    # Find the largest contour
    page = sorted(contours, key=cv2.contourArea, reverse=True)[0]
    # Find approximate corners of the contour
    epsilon = (0.02 * cv2.arcLength(page, True))
    corners = cv2.approxPolyDP(page, epsilon, True)
    corners = np.concatenate(corners).astype(np.float32)
    # Scale the corners back to the original image size
    corners[:, 0] *= scale_x
    corners[:, 1] *= scale_y
    # Check if more than 4 corners are detected
    if len(corners) > 4:
        # Find the minimum bounding box around the corners
        rect = cv2.minAreaRect(corners.reshape((-1, 1, 2)))
        box = cv2.boxPoints(rect)
        box_corners = np.int32(box)
        # Adjust the bounding box to fit within the image bounds
        box_x_min = np.min(box_corners[:, 0])
        box_x_max = np.max(box_corners[:, 0])
        box_y_min = np.min(box_corners[:, 1])
        box_y_max = np.max(box_corners[:, 1])
        left_pad, top_pad, right_pad, bottom_pad = 0, 0, 0, 0
        if box_x_min <= 0:
            left_pad = abs(box_x_min) + BUFFER
        if box_x_max >= imW:
            right_pad = (box_x_max - imW) + BUFFER
        if box_y_min <= 0:
            top_pad = abs(box_y_min) + BUFFER
        if box_y_max >= imH:
            bottom_pad = (box_y_max - imH) + BUFFER
        box_corners[:, 0] += left_pad
        box_corners[:, 1] += top_pad
        corners = box_corners
    # Arrange corners
    corners = order_points(corners)
    if expanded ==True:
        # Expand corners
        corners = expand(corners, 10)
    # Find the destination corners for perspective transformation
    destination_corners = find_dest(corners)
    # Compute the perspective transformation matrix
    M = cv2.getPerspectiveTransform(np.float32(corners), np.float32(destination_corners))
    # Apply the perspective transformation to the segmentation image
    final = cv2.warpPerspective(seg_img, M, (destination_corners[2][0], destination_corners[2][1]),
                                flags=cv2.INTER_LANCZOS4)
    # Return the final result
    return final
