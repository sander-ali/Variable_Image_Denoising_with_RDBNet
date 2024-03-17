import os.path
import numpy as np
from collections import OrderedDict
import torch
from utils import utils_model
from utils import utils_image as util


def main():

    x8 = False  # default: False, x8 to boost performance
    model_path = 'model_zoo/RDBBNet_denoise.pth'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.empty_cache()

    L_path = os.path.join('testsets/Mytestset') # L_path, for Low-quality images
    E_path = os.path.join('results')   # E_path, for Estimated images
    util.mkdir(E_path)

    from models.network_RDBNet import RDBNetRes as net
    n_channels = 3
    model = net(in_nc=n_channels+1, out_nc=n_channels, nc=[64, 128, 256, 512], nb=4, act_mode='R', downsample_mode="strideconv", upsample_mode="convtranspose")
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()
    for k, v in model.named_parameters():
        v.requires_grad = False
    model = model.to(device)

    test_results = OrderedDict()
    test_results['psnr'] = []
    test_results['ssim'] = []

    L_paths = util.get_image_paths(L_path)

    for idx, img in enumerate(L_paths):

        img_name, ext = os.path.splitext(os.path.basename(img))
        img_H = util.imread_uint(img, n_channels=n_channels)
        img_L = util.uint2single(img_H)

        np.random.seed(seed=0)  # for reproducibility
        img_L += np.random.normal(0, 0.5/255., img_L.shape)

        img_L = util.single2tensor4(img_L)
        img_L = torch.cat((img_L, torch.FloatTensor([0.5/255.]).repeat(1, 1, img_L.shape[2], img_L.shape[3])), dim=1)
        img_L = img_L.to(device)

        if not x8 and img_L.size(2)//8==0 and img_L.size(3)//8==0:
            img_E = model(img_L)
        elif not x8 and (img_L.size(2)//8!=0 or img_L.size(3)//8!=0):
            img_E = utils_model.test_mode(model, img_L, refield=64, mode=5)
        elif x8:
            img_E = utils_model.test_mode(model, img_L, mode=3)

        img_E = util.tensor2uint(img_E)

        util.imsave(img_E, os.path.join(E_path, img_name+ext))

if __name__ == '__main__':

    main()
