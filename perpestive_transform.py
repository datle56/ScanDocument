import cv2
import numpy as np
import matplotlib.pyplot as plt
from deskew import determine_skew
import math
from typing import Tuple, Union
def rotate(image: np.ndarray, angle: float, background: Union[int, Tuple[int, int, int]]) -> np.ndarray:
    old_width, old_height = image.shape[:2]
    angle_radian = math.radians(angle)
    width = abs(np.sin(angle_radian) * old_height) + abs(np.cos(angle_radian) * old_width)
    height = abs(np.sin(angle_radian) * old_width) + abs(np.cos(angle_radian) * old_height)

    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    rot_mat[1, 2] += (width - old_width) / 2
    rot_mat[0, 2] += (height - old_height) / 2

    return cv2.warpAffine(image, rot_mat, (int(round(height)), int(round(width))), borderValue=background)


def check(my_list):
    unique_elements = []

    # Sử dụng vòng lặp để kiểm tra từng phần tử trong danh sách
    for item in my_list:
        # Nếu phần tử không xuất hiện trong danh sách các phần tử duy nhất, thêm nó vào danh sách đó
        if item not in unique_elements:
            unique_elements.append(item)
    return len(unique_elements)


def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    pts = np.array(pts)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect.astype("int").tolist()


def find_dest(pts):
    (tl, tr, br, bl) = pts
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    destination_corners = [[0, 0], [maxWidth, 0], [maxWidth, maxHeight], [0, maxHeight]]
    return order_points(destination_corners)
def extract(ori_img, img, image_size=384, BUFFER=100):
    img=img.astype(np.uint8)
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    angle = determine_skew(gray_image)
    img = rotate(img, angle, (0, 0, 0))
    ori_img = rotate(ori_img, angle, (0, 0, 0))
    #get size of image
    size = img.shape
    top_pad = size[1]      # Số pixel padding ở phía trên
    bottom_pad = size[1]   # Số pixel padding ở phía dưới
    left_pad = size[0]   # Số pixel padding ở phía trái
    right_pad = size[0]   # Số pixel padding ở phía phải

    # Tạo hình ảnh mới với kích thước lớn hơn, bằng cách thêm pixel màu đen (0) vào xung quanh
    height, width, channels = img.shape
    new_height = height + top_pad + bottom_pad
    new_width = width + left_pad + right_pad

    # Tạo một hình ảnh mới với màu đen (0) là màu nền
    padded_img = np.zeros((new_height, new_width, channels), dtype=np.uint8)

    # Copy nội dung của hình ảnh gốc vào vị trí tương ứng trong hình ảnh mới
    padded_img[top_pad:top_pad + height, left_pad:left_pad + width] = img
    img = padded_img

    height, width, channels = ori_img.shape
    new_height = height + top_pad + bottom_pad
    new_width = width + left_pad + right_pad

    # Tạo một hình ảnh mới với màu đen (0) là màu nền
    padded_ori_img = np.full((new_height, new_width, channels), 255, dtype=np.uint8)

    # Copy nội dung của hình ảnh gốc vào vị trí tương ứng trong hình ảnh mới
    padded_ori_img[top_pad:top_pad + height, left_pad:left_pad + width] = ori_img
    ori_img = padded_ori_img

    imH, imW, C = img.shape
    IMAGE_SIZE = image_size
    img_rs = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_NEAREST)

    # imH, imW, C = img.shape
    # IMAGE_SIZE=image_size
    scale_x = imW / IMAGE_SIZE
    scale_y = imH / IMAGE_SIZE
    # img=cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_NEAREST)

    canny = cv2.Canny(img_rs.astype(np.uint8), 225, 255)
    canny = cv2.dilate(canny, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    contours, _ = cv2.findContours(canny, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    page = sorted(contours, key=cv2.contourArea, reverse=True)[0]
    epsilon = (0.02* cv2.arcLength(page, True))
    corners = cv2.approxPolyDP(page, epsilon, True)
    corners = np.concatenate(corners).astype(np.float32)
    corners[:, 0] *= scale_x
    corners[:, 1] *= scale_y

    # corners[:, 0] -= half
    # corners[:, 1] -= half
    for corner in corners:
        x, y = corner.astype(int)
        cv2.circle(img, (int(x), int(y)), 20, (0, 255, 0), -1)  # Vẽ một hình tròn màu xanh lên ảnh

    if len(corners) > 4:
        left_pad, top_pad, right_pad, bottom_pad = 0, 0, 0, 0
        rect = cv2.minAreaRect(corners.reshape((-1, 1, 2)))
        box = cv2.boxPoints(rect)
        box_corners = np.int32(box)
        #     box_corners = minimum_bounding_rectangle(corners)

        box_x_min = np.min(box_corners[:, 0])
        box_x_max = np.max(box_corners[:, 0])
        box_y_min = np.min(box_corners[:, 1])
        box_y_max = np.max(box_corners[:, 1])

        # Find corner point which doesn't satify the image constraint
        # and record the amount of shift required to make the box
        # corner satisfy the constraint
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
    
    if check(order_points(corners)) >= 4:
        corners = order_points(corners)
    else:
        pass

# Define the amount to increase the rectangle size
    (tl, tr, br, bl) = corners
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # Increase the x-coordinate of the top-right and bottom-right points
    corners[1][0] += maxWidth/30
    corners[2][0] += maxWidth/30

    # Decrease the x-coordinate of the top-left and bottom-left points
    corners[0][0] -= maxWidth/30
    corners[3][0] -= maxWidth/30
    # Increase the y-coordinate of the bottom-right and bottom-left points
    corners[2][1] += maxHeight/30
    corners[3][1] += maxHeight/30

    # Decrease the y-coordinate of the top-left and top-right points
    corners[0][1] -= maxHeight/30
    corners[1][1] -= maxHeight/30

    # print(corners)

    destination_corners = find_dest(corners)
    M = cv2.getPerspectiveTransform(np.float32(corners), np.float32(destination_corners))
    final = cv2.warpPerspective(ori_img, M, (destination_corners[2][0], destination_corners[2][1]),
                                flags=cv2.INTER_LANCZOS4)
    return final
# ori=cv2.imread("runs\segment\predict\image0.jpg")
# img=cv2.imread("mask/2.png")
# final=extract(ori,img)
# plt.imshow(final)
# plt.show()
# # print(img.shape)