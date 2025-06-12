import sys
sys.path.append('core')
import argparse
import glob
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path
from models.models.stereoplus.stereoplus_aanet import StereoNet
from models.models.stereonet.TinyHITNet_stereonet import StereoNet
from core.utils.utils import InputPadder
from PIL import Image
from matplotlib import pyplot as plt
import os
import skimage.io
import cv2
import torch.nn.functional as F
import sys
import time
from config import Stereo
Stereo = Stereo()
DEVICE = 'cuda'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

class NormalizeTensor(object):
    """Normalize a tensor by given mean and std."""
    
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)
    
    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
            
        Returns:
            Tensor: Normalized Tensor image.
        """
        # Ensure mean and std have the same number of channels as the input tensor
        Device = tensor.device
        self.mean = self.mean.to(Device)
        self.std = self.std.to(Device)

        # Normalize the tensor
        if self.mean.ndimension() == 1:
            self.mean = self.mean[:, None, None]
        if self.std.ndimension() == 1:
            self.std = self.std[:, None, None]

        return (tensor - self.mean) / self.std

def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)
def visualize_disp(disp, colormap=cv2.COLORMAP_MAGMA):
    norm = ((disp - disp.min()) / (disp.max() - disp.min()) * 255).astype(np.uint8)
    depth_colormap = cv2.applyColorMap(norm, cv2.COLORMAP_PLASMA)
    # # 归一化到 0-255
    # depth_min = 0.3376  # np.min(depth_filtered)
    # depth_max = 20.0000  # np.max(depth_filtered)
    # depth_norm = (depth_filtered - depth_min) / (depth_max - depth_min)  # 归一化到 0-1
    # depth_vis = (depth_norm * 255).astype(np.uint8)  # 转换为 0-255 范围

    # # 伪彩色映射
    # depth_colormap = cv2.applyColorMap(depth_vis, colormap)
    return depth_colormap
def demo(args):
    model = torch.nn.DataParallel(StereoNet(args), device_ids=[0])

    assert os.path.exists(args.restore_ckpt)
    checkpoint = torch.load(args.restore_ckpt)
    ckpt = dict()
    if 'state_dict' in checkpoint.keys():
        checkpoint = checkpoint['state_dict']
    for key in checkpoint:
        ckpt['module.' + key] = checkpoint[key]

    model.load_state_dict(ckpt, strict=True)

    model = model.module
    model.to(DEVICE)
    model.eval()

    output_directory = Path(args.output_directory)
    output_directory.mkdir(exist_ok=True)

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    normalize = NormalizeTensor(mean, std)

    with torch.no_grad():
        left_images = sorted(glob.glob(args.left_imgs, recursive=True))
        right_images = sorted(glob.glob(args.right_imgs, recursive=True))
        print(f"Found {len(left_images)} images. Saving files to {output_directory}/")

        for (imfile1, imfile2) in tqdm(list(zip(left_images, right_images))):
            image1 = load_image(imfile1)
            image2 = load_image(imfile2)
            padder = InputPadder(image1.shape, divis_by=32)
            image1, image2 = padder.pad(image1, image2)

            # torch.onnx.export(
            #     model,
            #     (torch.zeros(1,3,480,640).cuda(), torch.zeros(1,3,480,640).cuda()),
            #     "stereoplus_aanet.onnx",
            #     opset_version=16,  # ONNX opset 版本
            #     do_constant_folding=True,  # 是否进行常量折叠优化
            #     input_names=["left", "right"],  # 输入名称
            #     output_names=["output"],  # 输出名称
            #     dynamic_axes=None  # 动态维度（可选）
            # )


            if image1.shape[1] == 4:
                image1 = image1[:, :3, :, :]
                image2 = image2[:, :3, :, :]
            start_time = time.time()
            disp = model(image1, image2, None, test_mode=True)

            end_time = time.time()
            inference_time = end_time - start_time
            print(f"Inference time: {inference_time:.4f} seconds")
            disp = padder.unpad(disp).cpu().numpy().squeeze()
            print(np.max(disp), np.min(disp), disp.shape)
            disp_color = visualize_disp(disp)
            file_stem = os.path.join(output_directory, imfile1.split('/')[-1])
            cv2.imwrite(file_stem, disp_color)
            if image1.ndim == 4:
                image1 = image1.squeeze(0).permute(1, 2, 0).cpu().numpy()
                image1 = cv2.cvtColor(image1, cv2.COLOR_RGB2BGR)
                image1 = image1.astype(np.uint8)
            print(disp.shape, image1.shape)
            Stereo.show_depth_point(disp, image1)
            # skimage.io.imsave(file_stem, disp)
            if args.save_numpy:
                np.save(output_directory / f"{file_stem}.npy", disp.squeeze())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--restore_ckpt', help="restore checkpoint", default="kitti_2012/final.pth")

    parser.add_argument('--save_numpy', action='store_true', help='save output as numpy arrays')
    parser.add_argument('-l', '--left_imgs', help="path to all first (left) frames", default="./data/222/im0.png")
    parser.add_argument('-r', '--right_imgs', help="path to all second (right) frames", default="./data/222/im1.png")

    parser.add_argument('--output_directory', help="directory to save output", default="kitti_2012")

    parser.add_argument('--max_disp', type=int, default=192, help="max disp of geometry encoding volume")
    
    args = parser.parse_args()

    demo(args)