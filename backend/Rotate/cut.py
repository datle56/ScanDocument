import cv2
import os
import ast
import shutil
def crop(image):
    # Đọc file txt chứa tọa độ các hộp
    with open('Rotate/output/det_results.txt', 'r') as f:
        lines = f.readlines()

    folder_name = "Rotate/output/cut"

    # Kiểm tra xem thư mục đã tồn tại chưa
    if os.path.exists(folder_name):
        shutil.rmtree(folder_name)
    os.makedirs(folder_name)
    lines=lines[0]
    lines = eval(lines)
    for i, rect in enumerate(lines):
        # Chuyển đổi tọa độ sang số nguyên
        rect = [[int(j) for j in i] for i in rect]
        # Cắt hình ảnh
        cropped = image[rect[0][1]:rect[2][1], rect[0][0]:rect[2][0]]
        h, w = cropped.shape[:2]
        if h<w:
            cropped_rect = cropped[:,:2*h]
        else:
            cropped_rect = cropped[:2*w, :]
        try:
          # Lưu hình ảnh đã cắt
          cv2.imwrite(f'Rotate/output/cut/cropped_{i}.jpg', cropped_rect)
        except:
          pass
# crop(cv2.imread("2_1.1.jpg"))