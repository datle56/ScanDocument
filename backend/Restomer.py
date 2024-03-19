# import argparse
# import sys
import torch
import torch.nn.functional as F
# import torchvision.transforms.functional as TF
import os
from runpy import run_path
from skimage import img_as_ubyte
from natsort import natsorted
from glob import glob
import cv2
from tqdm import tqdm
import argparse
# from pdb import set_trace as stx
import numpy as np
def load_model():
  parameters = {'inp_channels':3, 'out_channels':3, 'dim':48, 'num_blocks':[4,6,6,8], 'num_refinement_blocks':4, 'heads':[1,2,4,8], 'ffn_expansion_factor':2.66, 'bias':False, 'LayerNorm_type':'WithBias', 'dual_pixel_task':False}
  weights = "Model/net_g_160000.pth"
  parameters['LayerNorm_type'] =  'BiasFree'
  load_arch = run_path("restormer_arch.py")
  model = load_arch['Restormer'](**parameters)
  
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model.to(device)

  checkpoint = torch.load(weights)
  model.load_state_dict(checkpoint['params'])
  model.eval()
  return model
def run_script(input_dir,model, tile, tile_overlap):

    def load_img(filepath):
        return cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB)

    def save_img(filepath, img):
        cv2.imwrite(filepath,cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    def load_gray_img(filepath):
        return np.expand_dims(cv2.imread(filepath, cv2.IMREAD_GRAYSCALE), axis=2)

    def save_gray_img(filepath, img):
        cv2.imwrite(filepath, img)


    img=cv2.cvtColor(input_dir, cv2.COLOR_BGR2RGB)

    img_multiple_of = 8

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with torch.no_grad():
        # for file_ in tqdm(files):
        if torch.cuda.is_available():
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()

        input_ = torch.from_numpy(img).float().div(255.).permute(2,0,1).unsqueeze(0).to(device)

        # Pad the input if not_multiple_of 8
        height,width = input_.shape[2], input_.shape[3]
        H,W = ((height+img_multiple_of)//img_multiple_of)*img_multiple_of, ((width+img_multiple_of)//img_multiple_of)*img_multiple_of
        padh = H-height if height%img_multiple_of!=0 else 0
        padw = W-width if width%img_multiple_of!=0 else 0
        input_ = F.pad(input_, (0,padw,0,padh), 'reflect')

        if tile is None:
            ## Testing on the original resolution image
            restored = model(input_)
        else:
            # test the image tile by tile
            b, c, h, w = input_.shape
            tile = min(tile, h, w)
            assert tile % 8 == 0, "tile size should be multiple of 8"
            tile_overlap = tile_overlap

            stride = tile - tile_overlap
            h_idx_list = list(range(0, h-tile, stride)) + [h-tile]
            w_idx_list = list(range(0, w-tile, stride)) + [w-tile]
            E = torch.zeros(b, c, h, w).type_as(input_)
            W = torch.zeros_like(E)

            for h_idx in h_idx_list:
                for w_idx in w_idx_list:
                    in_patch = input_[..., h_idx:h_idx+tile, w_idx:w_idx+tile]
                    out_patch = model(in_patch)
                    out_patch_mask = torch.ones_like(out_patch)

                    E[..., h_idx:(h_idx+tile), w_idx:(w_idx+tile)].add_(out_patch)
                    W[..., h_idx:(h_idx+tile), w_idx:(w_idx+tile)].add_(out_patch_mask)
            restored = E.div_(W)

        restored = torch.clamp(restored, 0, 1)

        # Unpad the output
        restored = restored[:,:,:height,:width]

        restored = restored.permute(0, 2, 3, 1).cpu().detach().numpy()
        restored = img_as_ubyte(restored[0])

        return restored
# img=run_script(cv2.imread("/content/test4.png"), "Real_Denoising", 720, 32)
# cv2.imwrite("/content/test5.png",img)