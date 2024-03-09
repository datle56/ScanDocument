from Segmentation import *
from perpestive_transform import *
from Rotate.rotate_function import *
from Rotate import rotate_function
img=cv2.imread("test4.png")
original_img=segmentation_doc(img)
mask_path="mask"
masks = [f for f in os.listdir(mask_path) if os.path.isfile(os.path.join(mask_path, f))]
for mask_name in masks:
    mask=os.path.join(mask_path, mask_name)
    mask_img=cv2.imread(mask)
    extract_document=extract(original_img,mask_img)
    result=rotate_function.rotate(extract_document)
    plt.imshow(result)
    plt.show()