import gradio as gr
import numpy as np
from PIL import Image
import os
from Segmentation import *
from perpestive_transform import *
from Rotate.rotate_function import *
from Rotate import rotate_function
from Restomer import *
model = YOLO('/Model/Seg65ep.pt')
model_re=load_model()
def process_image(input_image):
    original_img,masks=segmentation_doc(input_image,model)
    output_images=[]
    for mask_img in masks:
        extract_document=extract(original_img,mask_img)
        result=rotate_function.rotate(extract_document)
        result=run_script(result,model_re, 720, 32)
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
iface.launch(share=True)
