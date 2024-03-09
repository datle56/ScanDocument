import gradio as gr
import numpy as np
from PIL import Image
import os
from Segmentation import *
from perpestive_transform import *
from Rotate.rotate_function import *
from Rotate import rotate_function
def process_image(input_image):
    original_img=segmentation_doc(input_image)
    output_images=[]
    mask_path="mask"
    masks = [f for f in os.listdir(mask_path) if os.path.isfile(os.path.join(mask_path, f))]
    for mask_name in masks:
        mask=os.path.join(mask_path, mask_name)
        mask_img=cv2.imread(mask)
        extract_document=extract(original_img,mask_img)
        result=rotate_function.rotate(extract_document)
        output_images.append(result)
    return output_images

def gradio_interface(input_image):
    # Chuyển đổi hình ảnh đầu vào từ định dạng PIL sang numpy
    input_image_np = np.array(input_image)
    
    # Xử lý hình ảnh
    output_images = process_image(input_image_np)
    
    # Chuyển đổi hình ảnh đầu ra từ định dạng numpy sang PIL
    output_images_pil = [Image.fromarray(img) for img in output_images]
    
    return output_images_pil

# Tạo giao diện Gradio
iface = gr.Interface(fn=process_image, inputs="image", outputs=gr.Gallery(label="Documents", columns=[2]))
iface.launch()
