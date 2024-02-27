from Segmentation import *
from perpestive_transform import *
img=cv2.imread("test1.jpg")
original_img=segmentation_doc(img)
mask_path="mask"
masks = [f for f in os.listdir(mask_path) if os.path.isfile(os.path.join(mask_path, f))]
for mask_name in masks:
    mask=os.path.join(mask_path, mask_name)
    mask_img=cv2.imread(mask)
    extract_document=extract(original_img,mask_img)
    paperEdge_img=run_PaperEdge("/content/G_w_checkpoint_13820.pt",extract_document)
    plt.imshow(paperEdge_img)
    plt.show()