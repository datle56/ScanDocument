
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import numpy as np
from ultralytics import YOLO
import cv2
import os
import shutil
import math
from typing import Tuple, Union
from extract_orginal import *
from PIL import Image, ImageOps
from demopaperedge import run_code
import torch
model = YOLO('65ep.pt')
r_model=YOLO('3.pt')
size=(500,500)z
def check_orientation_and_rotate(image):
    # Khởi tạo giá trị lớn nhất ban đầu là 0
    max_conf = 0
    # Khởi tạo ảnh kết quả
    result_image = None
    # Xoay ảnh 4 lần
    for _ in range(4):
        # Dự đoán với mô hình
        results = r_model(image)[0]

        # Kiểm tra nếu ảnh đang ở định hướng đúng
        if results.names[results.probs.top1] == 'right':
            # Lấy giá trị confidence
            conf = results.probs.top1conf.item()

            # Nếu giá trị confidence lớn hơn giá trị lớn nhất hiện tại
            if conf > max_conf:
                # Cập nhật giá trị lớn nhất
                max_conf = conf
                # Lưu ảnh này
                result_image = image
        # Xoay ảnh
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    # Trả về ảnh có giá trị confidence lớn nhất
    return result_image

def check_orientation_and_rotate2(image):
    # Khởi tạo giá trị nhỏ nhất ban đầu là 1
    min_conf = 1
    # Khởi tạo ảnh kết quả
    result_image = None
    # Xoay ảnh 4 lần
    for _ in range(4):
        # Dự đoán với mô hình
        results = r_model(image)[0]
        # Kiểm tra nếu ảnh đang ở định hướng sai
        if results.names[results.probs.top1] == 'wrong':
            # Lấy giá trị confidence
            conf = results.probs.top1conf.item()

            # Nếu giá trị confidence nhỏ hơn giá trị nhỏ nhất hiện tại
            if conf < min_conf:
                # Cập nhật giá trị nhỏ nhất
                min_conf = conf
                # Lưu ảnh này
                result_image = image
        # Xoay ảnh
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    # Trả về ảnh có giá trị confidence nhỏ nhất
    return result_image

def process_image(model,image_path):
    runs_folder = "runs"
    folder="mask"
    # Xóa thư mục runs
    if os.path.exists(runs_folder) and os.path.isdir(runs_folder):
        # Xóa thư mục runs
        shutil.rmtree(runs_folder)
    if os.path.exists(folder) and os.path.isdir(folder):
        # Xóa thư mục runs
        shutil.rmtree(folder)
    results = model(image_path,save=True,retina_masks = True, conf=0.7)
    crop_path = str(results[0].save_dir)
    return results,crop_path
def black(results,index):
  # Lấy mặt nạ
  mask_raw = results[0].masks[index].cpu().data.numpy().transpose(1, 2, 0)
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
  return masked
def black_mask(results):
    # Đường dẫn đến thư mục "runs"
    runs_folder = "mask"
 
# Kiểm tra xem thư mục "runs" đã tồn tại chưa
    if not os.path.exists(runs_folder):
        # Nếu chưa tồn tại, tạo thư mục "runs"
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
            name = "mask/"+str(mask_index) + '.png'
            # Show the masked part of the image
            cv2.imwrite(name,result_image)
st.title("Document Preprocessing Demo")
col1, col2 = st.columns((5, 5))
uploaded_image = st.file_uploader("Chọn một tệp ảnh", type=["jpg", "jpeg", "png"])
if uploaded_image is not None:
    image = cv2.imdecode(np.fromstring(uploaded_image.read(), np.uint8), cv2.IMREAD_COLOR)
    with col1 :
        st.image(image,channels="bgr", caption="Ảnh đã tải lên", use_column_width=True)
    color = [0, 0, 0] # Màu trắng
    padding_size = 100
    image = cv2.copyMakeBorder(image, padding_size, padding_size, padding_size, padding_size, cv2.BORDER_CONSTANT, value=color)
    try:
        results,mask_path=process_image(model,image)
        black_mask(results)
        # crop_path=mask_path+'\\crops\\document'
        # image_files = [f for f in os.listdir(crop_path) if os.path.isfile(os.path.join(crop_path, f))]
        mask_files=[f for f in os.listdir(mask_path) if os.path.isfile(os.path.join(mask_path, f))]
        with col2 :
            for image_file in mask_files:
                image_path = os.path.join(mask_path, image_file)
                st.image(image_path, caption="Mask", use_column_width=True)
        # for image_file in image_files:
        #     image_path = os.path.join(crop_path, image_file)
        #     image=cv2.imread(image_path)
        #     col1, col2 = st.columns((10, 10))
        #     with col1:
        #         st.image(image, caption="Yolov8",channels="BGR" ,use_column_width=True)
        path="mask/"
        image_files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
        i=0

        def tim_contour_lon_nhat(img):
            canny = cv2.Canny(img.astype(np.uint8), 225, 255)
            canny = cv2.dilate(canny, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
            contours, _ = cv2.findContours(canny, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            contour_lon_nhat = max(contours, key=cv2.contourArea)
            return contour_lon_nhat

        def tim_hinh_max_contour(image_files):
            max_dien_tich = 0
            path_hinh_max_contour = None
            i = 0
            for image_file in image_files:
                black_img=black(results,i)
                image_path = os.path.join(path, image_file)
                img=cv2.imread(image_path)
            

                contour_lon_nhat = tim_contour_lon_nhat(img)
                dien_tich = cv2.contourArea(contour_lon_nhat)

                if dien_tich > max_dien_tich:
                    max_dien_tich = dien_tich
                    path_hinh_max_contour = image_path
                i += 1
            return path_hinh_max_contour, i
        

        path_hinh_max_contour,i = tim_hinh_max_contour(image_files)
        black_img=black(results,i)
        image_path = path_hinh_max_contour
        col2,col3 = st.columns((10,10))
        image_black=cv2.imread(image_path)
        extract_img,img1=extract(black_img,image_black)
        extract_img=extract_img.astype(np.uint8)
        # cv2.imwrite("/home/fustudents/t  est1/Restormer/input/extract.png",extract_img)
        i = i +1
        name="/home/fustudents/Document/Ndat_test1/test/i/" + str(i) +'.jpg'
        cv2.imwrite(name,extract_img)
        run_code('/home/fustudents/Document/Ndat_test1/paperedge/PaperEdge/model/G_w_checkpoint_13820.pt',
        '/home/fustudents/Document/Ndat_test1/paperedge/PaperEdge/model/L_w_checkpoint_27640.pt',
        name,
        '/home/fustudents/Document/Ndat_test1/test/o')
        pp=cv2.imread("/home/fustudents/Document/Ndat_test1/test/o/result_gs.png")
        pp2=cv2.imread("/home/fustudents/Document/Ndat_test1/test/o/result_ls.png")
        with col2:
            st.image(extract_img, caption="Extract",channels="BGR" ,use_column_width=True)
        try:
            rpp2=check_orientation_and_rotate(pp)
            with col3:
                st.image(rpp2, caption="Extract2",channels="BGR" ,use_column_width=True)
        except:
            rpp2=check_orientation_and_rotate2(pp)
            with col3:
                st.image(rpp2, caption="Extract2",channels="BGR" ,use_column_width=True)
        
# except:
#     name="/home/fustudents/Document/Ndat_test1/test/i" +'no_mask.jpg'
#     cv2.imwrite(name,image)
#     run_code('/home/fustudents/Document/Ndat_test1/paperedge/PaperEdge/model/G_w_checkpoint_13820.pt',
#     '/home/fustudents/Document/Ndat_test1/paperedge/PaperEdge/model/L_w_checkpoint_27640.pt',
#     name,
#     '/home/fustudents/Document/Ndat_test1/test/o')
#     pp=cv2.imread("/home/fustudents/Document/Ndat_test1/test/o/result_gs.png")
#     try:
#         rpp=check_orientation_and_rotate(pp)
#         with col2:
#             st.image(rpp, caption="Ảnh đã cắt",channels="BGR" ,use_column_width=True)
#     except:
#         rpp=check_orientation_and_rotate2(pp)
#         with col2:
#             st.image(rpp, caption="Ảnh đã cắt",channels="BGR" ,use_column_width=True)

    #     for image_file in image_files:
    #         black_img=black(results,i)
    #         image_path = os.path.join(path, image_file)
    #         col2,col3 = st.columns((10,10))
    #         image_black=cv2.imread(image_path)
    #         extract_img,img1=extract(black_img,image_black)
    #         extract_img=extract_img.astype(np.uint8)
    #         # cv2.imwrite("/home/fustudents/t  est1/Restormer/input/extract.png",extract_img)
    #         i = i +1
    #         name="/home/fustudents/Document/Ndat_test1/test/i/" + str(i) +'.jpg'
    #         cv2.imwrite(name,extract_img)
    #         run_code('/home/fustudents/Document/Ndat_test1/paperedge/PaperEdge/model/G_w_checkpoint_13820.pt',
    #         '/home/fustudents/Document/Ndat_test1/paperedge/PaperEdge/model/L_w_checkpoint_27640.pt',
    #         name,
    #         '/home/fustudents/Document/Ndat_test1/test/o')
    #         pp=cv2.imread("/home/fustudents/Document/Ndat_test1/test/o/result_gs.png")
    #         pp2=cv2.imread("/home/fustudents/Document/Ndat_test1/test/o/result_ls.png")
    #         with col2:
    #             st.image(extract_img, caption="Extract",channels="BGR" ,use_column_width=True)
    #         try:
    #             rpp2=check_orientation_and_rotate(pp)
    #             with col3:
    #                 st.image(rpp2, caption="Extract2",channels="BGR" ,use_column_width=True)
    #         except:
    #             rpp2=check_orientation_and_rotate2(pp)
    #             with col3:
    #                 st.image(rpp2, caption="Extract2",channels="BGR" ,use_column_width=True)
            
    # except:
    #     name="/home/fustudents/Document/Ndat_test1/test/i" +'no_mask.jpg'
    #     cv2.imwrite(name,image)
    #     run_code('/home/fustudents/Document/Ndat_test1/paperedge/PaperEdge/model/G_w_checkpoint_13820.pt',
    #     '/home/fustudents/Document/Ndat_test1/paperedge/PaperEdge/model/L_w_checkpoint_27640.pt',
    #     name,
    #     '/home/fustudents/Document/Ndat_test1/test/o')
    #     pp=cv2.imread("/home/fustudents/Document/Ndat_test1/test/o/result_gs.png")
    #     try:
    #         rpp=check_orientation_and_rotate(pp)
    #         with col2:
    #             st.image(rpp, caption="Ảnh đã cắt",channels="BGR" ,use_column_width=True)
    #     except:
    #         rpp=check_orientation_and_rotate2(pp)
    #         with col2:
    #             st.image(rpp, caption="Ảnh đã cắt",channels="BGR" ,use_column_width=True)
        
       
