from ultralytics import YOLO
import cv2
import os
import shutil
import numpy as np
def black_mask(results):
    # Đường dẫn đến thư mục "runs"
    runs_folder = "mask"
    bl_mask=[]
# Kiểm tra xem thư mục "mask" đã tồn tại chưa
    if not os.path.exists(runs_folder):
        # Nếu chưa tồn tại, tạo thư mục "mask"
        os.makedirs(runs_folder)
    for mask_index in range(len(results[0].masks)):
        # Lấy mặt nạ
            mask_raw = results[0].masks[mask_index].cpu().data.numpy().transpose(1, 2, 0)
            # Chuyển đổi mặt nạ thành 3 kênh màu
            mask_3channel = cv2.merge((mask_raw, mask_raw, mask_raw))
            # Lấy kích thước của ảnh gốc (chiều cao, chiều rộng, số kênh)
            h2, w2, c2 = results[0].orig_img.shape
            # Thay đổi kích thước mặt nạ thành cùng kích thước với ảnh gốc
            mask = cv2.resize(mask_3channel, (w2, h2))
            # Chuyển đổi ảnh mặt nạ thành không gian màu HSV
            hsv = cv2.cvtColor(mask, cv2.COLOR_BGR2HSV)
            # Xác định phạm vi độ sáng trong không gian màu HSV
            lower_black = np.array([0, 0, 0])
            upper_black = np.array([0, 0, 1])
            # Tạo mặt nạ dựa trên phạm vi độ sáng
            mask = cv2.inRange(mask, lower_black, upper_black)
            # Đảo ngược mặt nạ để lấy mọi thứ trừ màu đen
            mask = cv2.bitwise_not(mask)
            # Áp dụng mặt nạ vào ảnh gốc
            masked = cv2.bitwise_and(results[0].orig_img, results[0].orig_img, mask=mask)
            # Chuyển mọi pixel màu đen thành màu trắng
            black_color = [0, 0, 0]
            white_color = [255, 255, 255]
            result_image = np.where(np.all(masked == black_color, axis=-1, keepdims=True), masked, white_color)
            # name = "mask/"+str(mask_index) + '.png'
            # # Show the masked part of the image
            # cv2.imwrite(name,result_image)
            bl_mask.append(result_image)
    return bl_mask
def segmentation_doc(image,model):
    # runs_folder = "runs"
    # folder="mask"
    # # Xóa thư mục runs
    # if os.path.exists(runs_folder) and os.path.isdir(runs_folder):
    #     # Xóa thư mục runs
    #     shutil.rmtree(runs_folder)
    # if os.path.exists(folder) and os.path.isdir(folder):
    #     # Xóa thư mục runs
    #     shutil.rmtree(folder)
    color = [255, 255, 255] # Màu trắng
    padding_size = 50
    image = cv2.copyMakeBorder(image, padding_size, padding_size, padding_size, padding_size, cv2.BORDER_CONSTANT, value=color)
    results = model(image,save=False,retina_masks = True, conf=0.7)
    bl_mask=black_mask(results)
    return image,bl_mask
# image=cv2.imread("test1.jpg")
# segmentation_doc(image)