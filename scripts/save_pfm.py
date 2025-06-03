import sys
sys.path.append('../core')
import argparse
import glob
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path
from models.Monster.monster import Monster
from core.utils.utils import InputPadder
from PIL import Image
import os

DEVICE = 'cuda'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)[..., :3]
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)

def demo(args):
    model = torch.nn.DataParallel(Monster(args), device_ids=[0])

    checkpoint = torch.load(args.restore_ckpt)
    ckpt = dict()
    if 'state_dict' in checkpoint.keys():
        checkpoint = checkpoint['state_dict']
    for key in checkpoint:
        ckpt['module.' + key] = checkpoint[key]

    model.load_state_dict(checkpoint, strict=True)

    model = model.module
    model.to(DEVICE)
    model.eval()

    output_directory = Path(args.output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        left_images = sorted(glob.glob(os.path.join('/data2/cjd/StereoDatasets/middlebury/', "MiddEval3", 'testF', '*/im0.png')))
        right_images = sorted(glob.glob(os.path.join('/data2/cjd/StereoDatasets/middlebury/', "MiddEval3", 'testF', '*/im1.png')))
        print(f"Found {len(left_images)} images. Saving files to {output_directory}/")

        for (imfile1, imfile2) in tqdm(list(zip(left_images, right_images))):
            image1 = load_image(imfile1)
            image2 = load_image(imfile2)

            padder = InputPadder(image1.shape, divis_by=32)
            image1, image2 = padder.pad(image1, image2)

            starter.record()
            disp = model(image1, image2, iters=args.valid_iters, test_mode=True)

            ender.record()
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)

            disp = disp.cpu().numpy()
            disp = padder.unpad(disp).squeeze()
            file_stem = imfile1.split('/')[-2]
            filedir = Path(os.path.join(output_directory, file_stem))
            filedir.mkdir(exist_ok=True)
            filename = os.path.join(output_directory, file_stem, 'disp0MonoStereo.pfm')
            with open(filename, 'wb') as f:
                H, W = disp.shape
                headers = ["Pf\n", f"{W} {H}\n", "-1\n"]
                for header in headers:
                    f.write(str.encode(header))
                array = np.flip(disp, axis=0).astype(np.float32)
                f.write(array.tobytes())

            filename = os.path.join(output_directory, file_stem, 'timeMonoStereo.txt')
            with open(filename, 'wb') as f:
                time = '%.2f' % (curr_time / 1000)
                f.write(str.encode(time))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--restore_ckpt', help="restore checkpoint", default="/data2/cjd/mono_fusion/checkpoints/middlebury.pth")
    parser.add_argument('--save_numpy', action='store_true', help='save output as numpy arrays')
    parser.add_argument('-l', '--left_imgs', help="path to all first (left) frames", default=None)
    parser.add_argument('-r', '--right_imgs', help="path to all second (right) frames", default=None)
    parser.add_argument('--output_directory', help="directory to save output", default='./test_output/middlebury')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--valid_iters', type=int, default=32, help='number of flow-field updates during forward pass')

    # Architecture choices
    parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl', 'vitg'])
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128]*3, help="hidden state and context dimensions")
    parser.add_argument('--corr_implementation', choices=["reg", "alt", "reg_cuda", "alt_cuda"], default="reg", help="correlation volume implementation")
    parser.add_argument('--shared_backbone', action='store_true', help="use a single backbone for the context and feature encoders")
    parser.add_argument('--corr_levels', type=int, default=2, help="number of levels in the correlation pyramid")
    parser.add_argument('--corr_radius', type=int, default=4, help="width of the correlation pyramid")
    parser.add_argument('--n_downsample', type=int, default=2, help="resolution of the disparity field (1/2^K)")
    parser.add_argument('--slow_fast_gru', action='store_true', help="iterate the low-res GRUs more frequently")
    parser.add_argument('--n_gru_layers', type=int, default=3, help="number of hidden GRU levels")
    parser.add_argument('--max_disp', type=int, default=768, help="max disp of geometry encoding volume")
    
    args = parser.parse_args()

    demo(args)
