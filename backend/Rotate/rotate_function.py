from Rotate.predict_cls import *
from Rotate.predict_det import *
from Rotate.cut import *
import cv2
import matplotlib.pyplot as plt

def is_90():
    file_path="Rotate/output/det_results.txt"
    with open(file_path, 'r') as file:
        data = file.read().splitlines()

    count_width_greater = 0
    count_height_greater = 0

    for line in data:
        _, boxes = line.split('\t', 1)
        boxes = eval(boxes)
        for coordinates in boxes:
            width = ((coordinates[1][0] - coordinates[0][0])**2 + (coordinates[1][1] - coordinates[0][1])**2)**0.5
            height = ((coordinates[2][0] - coordinates[1][0])**2 + (coordinates[2][1] - coordinates[1][1])**2)**0.5

            if width > height:
                count_width_greater += 1
            else:
                count_height_greater += 1

    return count_width_greater < count_height_greater

def is_180(img):
    crop(img)
    run_text_classifier(
        image_dir="Rotate/output/cut",
        use_gpu=False,
        cls_model_dir="Rotate/ch_ppocr_mobile_v2.0_cls_infer/",
        draw_img_save_dir="Rotate/output/"
    )
    file_path = "Rotate/output/rotate.txt"
    # Khởi tạo biến đếm
    count_180 = 0
    count_0 = 0
    # Đọc từ file và đếm
    with open(file_path, "r") as file:
        for line in file:
            # Chuyển đổi dòng thành số nguyên
            number = int(line.strip())
            # Đếm số lượng
            if number == 180:
                count_180 += 1
            elif number == 0:
                count_0 += 1
    return count_180>count_0

def rotate_180(img):
    height, width = img.shape[:2]

    # Tính toán ma trận biến đổi để xoay hình ảnh 180 độ theo chiều kim đồng hồ
    # Trung tâm xoay là tâm của hình ảnh
    rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), 180, 1)

    # Thực hiện việc xoay hình ảnh
    rotated_image = cv2.warpAffine(img, rotation_matrix, (width, height))
    return rotated_image
def rotate_90(img):
    rotated_image = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    return rotated_image
def rotate(img):
    run_text_detector(
    img=img,
    use_gpu=False,
    # det_algorithm="DB",
    det_model_dir="Rotate/ch_PP-OCRv4_det_infer/",
    draw_img_save_dir='Rotate/output/'
)
    if is_90() == True : 
        img=rotate_90(img)
        run_text_detector(
        img=img,
        use_gpu=False,
        # det_algorithm="DB",
        det_model_dir="Rotate/ch_PP-OCRv4_det_infer/",
        draw_img_save_dir='Rotate/output/'
    )
    if is_180(img)==True:
        result=rotate_180(img)
    else:
        result=img
    return result
# img=cv2.imread("0.png")
# result=rotate(img)
# plt.imshow(result)
# plt.show()