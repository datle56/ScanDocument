import argparse
import os
import cv2
import shutil
from utils.segmentation import *
from utils.extract import *
from utils.extractPaperEdge import *


    

def main(image_path, output_path,only_one_document):
    model = YOLO('./models/segmentation.pt')
    paper_edge_model = "./models/PaperEdge.pt"

    # Perform binary mask and segmentation
    mask, seg_image = binary_mask(model, image_path)
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    # Create the output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)

    if only_one_document == True:
        index = get_only_one_document(mask)
        mask = [mask[index]]
        seg_image = [seg_image[index]]

    for i in range(len(mask)):
        # Extract the image using the mask
        extract_img = extract(seg_image[i], mask[i])

        # Extract paper edge using the extracted image and the PaperEdge model
        paper_edge = extractPaperEdge(extract_img, paper_edge_model)

        # Save the result to the output directory
        output_filename = os.path.join(output_path, f"output_{i}.jpg")
        cv2.imwrite(output_filename, paper_edge)
        print(f"Saved result {i} to {output_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract paper edge from an input image.")
    parser.add_argument("image_path", type=str, help="Path to the input image.")
    parser.add_argument("output_path", type=str, help="Path to the output directory.")
    parser.add_argument("only_one_document", type=bool, default=False, help="Whether to create only one document or more.")

    args = parser.parse_args()

    main(args.image_path, args.output_path,args.only_one_document)
