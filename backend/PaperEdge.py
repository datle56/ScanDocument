import copy
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from networks.paperedge import GlobalWarper, WarperUtil

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

def run_PaperEdge(Enet_ckpt, img):
    netG = GlobalWarper().to('cuda')
    netG.load_state_dict(torch.load(Enet_ckpt)['G'])

    netG.eval()

    warpUtil = WarperUtil(64).to('cuda')

    gs_d = None
    with torch.no_grad():
        im = img.astype(np.float32) / 255.0
        im = im[:, :, (2, 1, 0)]
        im = cv2.resize(im, (256, 256), interpolation=cv2.INTER_AREA)
        im = torch.from_numpy(np.transpose(im, (2, 0, 1)))
        x=im
        x = x.unsqueeze(0)
        x = x.to('cuda')
        d = netG(x)  # d_E the edged-based deformation field
        d = warpUtil.global_post_warp(d, 64)
        gs_d = copy.deepcopy(d)

    im = img.astype(np.float32) / 255.0
    im = torch.from_numpy(np.transpose(im, (2, 0, 1)))
    im = im.to('cuda').unsqueeze(0)

    gs_d = F.interpolate(gs_d, (im.size(2), im.size(3)), mode='bilinear', align_corners=True)
    gs_y = F.grid_sample(im, gs_d.permute(0, 2, 3, 1), align_corners=True).detach()
    tmp_y = gs_y.squeeze().permute(1, 2, 0).cpu().numpy()
    return tmp_y
# img=cv2.imread("/content/29.1.jpg")
# result=run_PaperEdge("/content/G_w_checkpoint_13820.pt",img)
# cv2.imwrite("result.jpg",result)