import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F
import logging
import os
import re
import copy
import math
import random
from pathlib import Path
from glob import glob
import os.path as osp

from core.utils import frame_utils
from core.utils.augmentor import FlowAugmentor, SparseFlowAugmentor


class StereoDataset(data.Dataset):
    def __init__(self, aug_params=None, sparse=False, reader=None):
        self.augmentor = None
        self.sparse = sparse
        self.img_pad = aug_params.pop("img_pad", None) if aug_params is not None else None
        if aug_params is not None and "crop_size" in aug_params:
            if sparse:
                self.augmentor = SparseFlowAugmentor(**aug_params)
            else:
                self.augmentor = FlowAugmentor(**aug_params)

        if reader is None:
            self.disparity_reader = frame_utils.read_gen
        else:
            self.disparity_reader = reader

        self.is_test = False
        self.init_seed = False
        self.flow_list = []
        self.disparity_list = []
        self.image_list = []
        self.extra_info = []

    def __getitem__(self, index):

        if self.is_test:
            img1 = frame_utils.read_gen(self.image_list[index][0])
            img2 = frame_utils.read_gen(self.image_list[index][1])
            img1 = np.array(img1).astype(np.uint8)[..., :3]
            img2 = np.array(img2).astype(np.uint8)[..., :3]
            img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
            img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
            return img1, img2, self.extra_info[index]

        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                random.seed(worker_info.id)
                self.init_seed = True

        index = index % len(self.image_list)
        disp = self.disparity_reader(self.disparity_list[index])

        if isinstance(disp, tuple):
            disp, valid = disp
        else:
            valid = disp < 1024

        img1 = frame_utils.read_gen(self.image_list[index][0])
        img2 = frame_utils.read_gen(self.image_list[index][1])

        img1 = np.array(img1).astype(np.uint8)
        img2 = np.array(img2).astype(np.uint8)
        disp = np.array(disp).astype(np.float32)
        flow = np.stack([disp, np.zeros_like(disp)], axis=-1)

        # grayscale images
        if len(img1.shape) == 2:
            img1 = np.tile(img1[..., None], (1, 1, 3))
            img2 = np.tile(img2[..., None], (1, 1, 3))
        else:
            img1 = img1[..., :3]
            img2 = img2[..., :3]

        if self.augmentor is not None:
            if self.sparse:
                img1, img2, flow, valid = self.augmentor(img1, img2, flow, valid)
            else:

                img1, img2, flow = self.augmentor(img1, img2, flow)

        img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
        flow = torch.from_numpy(flow).permute(2, 0, 1).float()

        if self.sparse:
            valid = torch.from_numpy(valid)
        else:
            valid = (flow[0].abs() < 1024) & (flow[1].abs() < 1024)

        if self.img_pad is not None:
            padH, padW = self.img_pad
            img1 = F.pad(img1, [padW] * 2 + [padH] * 2)
            img2 = F.pad(img2, [padW] * 2 + [padH] * 2)

        flow = flow[:1]
        return self.image_list[index] + [self.disparity_list[index]], img1, img2, flow, valid.float()

    def __mul__(self, v):
        copy_of_self = copy.deepcopy(self)
        copy_of_self.flow_list = v * copy_of_self.flow_list
        copy_of_self.image_list = v * copy_of_self.image_list
        copy_of_self.disparity_list = v * copy_of_self.disparity_list
        copy_of_self.extra_info = v * copy_of_self.extra_info
        return copy_of_self

    def __len__(self):
        return len(self.image_list)


class SceneFlowDatasets(StereoDataset):
    def __init__(self, aug_params=None, root='D:/project2024/stereo matching/DATA/SceneFlow/', dstype='frames_finalpass',
                 things_test=False):
        super(SceneFlowDatasets, self).__init__(aug_params)
        self.root = root
        self.dstype = dstype

        if things_test:
            self._add_things("TEST")
        else:
            self._add_things("TRAIN")
            self._add_monkaa("TRAIN")
            self._add_driving("TRAIN")

    def _add_things(self, split='TRAIN'):
        """ Add FlyingThings3D data """

        original_length = len(self.disparity_list)
        # root = osp.join(self.root, 'FlyingThings3D')

        root = self.root
        dstype = self.dstype + '/things'
        left_images = sorted( glob(osp.join(root, dstype, split, '*/*/left/*.png')) )
        right_images = [ im.replace('left', 'right') for im in left_images ]
        # disparity_images = [ im.replace(self.dstype, 'disparity').replace('.png', '.pfm') for im in left_images ]
        disparity_images = [ im.replace('.png', '.pfm') for im in left_images ]

        # Choose a random subset of 400 images for validation
        added_count = 0
        state = np.random.get_state()
        np.random.seed(1000)
        # val_idxs = set(np.random.permutation(len(left_images))[:100])
        val_idxs = set(np.random.permutation(len(left_images)))
        np.random.set_state(state)

        for idx, (img1, img2, disp) in enumerate(zip(left_images, right_images, disparity_images)):
            if osp.exists(img1) and osp.exists(img2) and osp.exists(disp):
                self.image_list += [[img1, img2]]
                self.disparity_list += [disp]
                added_count += 1
            else:
                print(f"Missing files for index {idx} in FlyingThings3D ({split}):")
                if not osp.exists(img1):
                    print(f"  - Left image: {img1}")
                if not osp.exists(img2):
                    print(f"  - Right image: {img2}")
                if not osp.exists(disp):
                    print(f"  - Disparity: {disp}")
        print(f"Added {len(self.disparity_list) - original_length} from FlyingThings {self.dstype}")

    def _add_monkaa(self, split="TRAIN"):
        """ Add FlyingThings3D data """

        original_length = len(self.disparity_list)
        root = self.root
        dstype = self.dstype + '/monkaa'
        left_images = sorted( glob(osp.join(root, dstype, split, '*/left/*.png')) )
        right_images = [ image_file.replace('left', 'right') for image_file in left_images ]
        # disparity_images = [ im.replace(self.dstype, 'disparity').replace('.png', '.pfm') for im in left_images ]
        disparity_images = [ im.replace('.png', '.pfm') for im in left_images ]

        added_count = 0
        for img1, img2, disp in zip(left_images, right_images, disparity_images):
            if osp.exists(img1) and osp.exists(img2) and osp.exists(disp):
                self.image_list += [[img1, img2]]
                self.disparity_list += [disp]
                added_count += 1
            else:
                print(f"Missing files in Monkaa ({split}):")
                if not osp.exists(img1):
                    print(f"  - Left image: {img1}")
                if not osp.exists(img2):
                    print(f"  - Right image: {img2}")
                if not osp.exists(disp):
                    print(f"  - Disparity: {disp}")
        print(f"Added {len(self.disparity_list) - original_length} from Monkaa {self.dstype}")


    def _add_driving(self, split="TRAIN"):
        """ Add FlyingThings3D data """

        original_length = len(self.disparity_list)
        root = self.root
        dstype = self.dstype + '/driving'
        left_images = sorted( glob(osp.join(root, dstype, split, '*/*/*/left/*.png')) )
        right_images = [ image_file.replace('left', 'right') for image_file in left_images ]
        # disparity_images = [ im.replace(self.dstype, 'disparity').replace('.png', '.pfm') for im in left_images ]
        disparity_images = [ im.replace('.png', '.pfm') for im in left_images ]

        added_count = 0
        for img1, img2, disp in zip(left_images, right_images, disparity_images):
            if osp.exists(img1) and osp.exists(img2) and osp.exists(disp):
                self.image_list += [ [img1, img2] ]
                self.disparity_list += [ disp ]
                added_count += 1
            else:
                print(f"Missing files in Driving ({split}):")
                if not osp.exists(img1):
                    print(f"  - Left image: {img1}")
                if not osp.exists(img2):
                    print(f"  - Right image: {img2}")
                if not osp.exists(disp):
                    print(f"  - Disparity: {disp}")
        print(f"Added {len(self.disparity_list) - original_length} from Driving {self.dstype}")


class ETH3D(StereoDataset):
    def __init__(self, aug_params=None, root='/data/StereoDatasets/eth3d', split='training'):
        super(ETH3D, self).__init__(aug_params, sparse=True)

        image1_list = sorted(glob(osp.join(root, f'two_view_{split}/*/im0.png')))
        image2_list = sorted(glob(osp.join(root, f'two_view_{split}/*/im1.png')))
        disp_list = sorted(glob(osp.join(root, 'two_view_training_gt/*/disp0GT.pfm'))) if split == 'training' else [
                                                                                                                       osp.join(root, 'two_view_training_gt/playground_1l/disp0GT.pfm')] * len(image1_list)

        for img1, img2, disp in zip(image1_list, image2_list, disp_list):
            self.image_list += [[img1, img2]]
            self.disparity_list += [disp]


class SintelStereo(StereoDataset):
    def __init__(self, aug_params=None, root='/data/StereoDatasets/sintelstereo'):
        super().__init__(aug_params, sparse=True, reader=frame_utils.readDispSintelStereo)

        image1_list = sorted(glob(osp.join(root, 'training/*_left/*/frame_*.png')))
        image2_list = sorted(glob(osp.join(root, 'training/*_right/*/frame_*.png')))
        disp_list = sorted(glob(osp.join(root, 'training/disparities/*/frame_*.png'))) * 2

        for img1, img2, disp in zip(image1_list, image2_list, disp_list):
            assert img1.split('/')[-2:] == disp.split('/')[-2:]
            self.image_list += [[img1, img2]]
            self.disparity_list += [disp]


class FallingThings(StereoDataset):
    def __init__(self, aug_params=None, root='/data/StereoDatasets/fallingthings'):
        super().__init__(aug_params, reader=frame_utils.readDispFallingThings)
        assert os.path.exists(root)

        image1_list = sorted(glob(root + '/*/*/*left.jpg'))
        image2_list = sorted(glob(root + '/*/*/*right.jpg'))
        disp_list = sorted(glob(root + '/*/*/*left.depth.png'))

        for img1, img2, disp in zip(image1_list, image2_list, disp_list):
            self.image_list += [[img1, img2]]
            self.disparity_list += [disp]


class TartanAir(StereoDataset):
    def __init__(self, aug_params=None, root='/data/StereoDatasets/tartanair'):
        super().__init__(aug_params, reader=frame_utils.readDispTartanAir)
        assert os.path.exists(root)

        image1_list = sorted(glob(osp.join(root, '*/*/*/*/image_left/*.png')))
        image2_list = sorted(glob(osp.join(root, '*/*/*/*/image_right/*.png')))
        disp_list = sorted(glob(osp.join(root, '*/*/*/*/depth_left/*.npy')))

        for img1, img2, disp in zip(image1_list, image2_list, disp_list):
            self.image_list += [[img1, img2]]
            self.disparity_list += [disp]


class CREStereoDataset(StereoDataset):
    def __init__(self, aug_params=None, root='/data/StereoDatasets/crestereo'):
        super(CREStereoDataset, self).__init__(aug_params, reader=frame_utils.readDispCREStereo)
        assert os.path.exists(root)

        image1_list = sorted(glob(os.path.join(root, '*/*_left.jpg')))
        image2_list = sorted(glob(os.path.join(root, '*/*_right.jpg')))
        disp_list = sorted(glob(os.path.join(root, '*/*_left.disp.png')))

        for idx, (img1, img2, disp) in enumerate(zip(image1_list, image2_list, disp_list)):
            self.image_list += [[img1, img2]]
            self.disparity_list += [disp]


class CARLA(StereoDataset):
    def __init__(self, aug_params=None, root='/data/StereoDatasets/carla-highres'):
        super(CARLA, self).__init__(aug_params)
        assert os.path.exists(root)

        image1_list = sorted(glob(root + '/trainingF/*/im0.png'))
        image2_list = sorted(glob(root + '/trainingF/*/im1.png'))
        disp_list = sorted(glob(root + '/trainingF/*/disp0GT.pfm'))

        for idx, (img1, img2, disp) in enumerate(zip(image1_list, image2_list, disp_list)):
            self.image_list += [[img1, img2]]
            self.disparity_list += [disp]


class InStereo2K(StereoDataset):
    def __init__(self, aug_params=None, root='/data/StereoDatasets/instereo2k'):
        super(InStereo2K, self).__init__(aug_params, sparse=True, reader=frame_utils.readDispInStereo2K)
        assert os.path.exists(root)

        image1_list = sorted(glob(root + '/train/*/*/left.png') + glob(root + '/test/*/left.png'))
        image2_list = sorted(glob(root + '/train/*/*/right.png') + glob(root + '/test/*/right.png'))
        disp_list = sorted(glob(root + '/train/*/*/left_disp.png') + glob(root + '/test/*/left_disp.png'))

        for idx, (img1, img2, disp) in enumerate(zip(image1_list, image2_list, disp_list)):
            self.image_list += [[img1, img2]]
            self.disparity_list += [disp]


class KITTI(StereoDataset):
    def __init__(self, aug_params=None, root='/data/StereoDatasets/kitti', image_set='training', year=2015):
        super(KITTI, self).__init__(aug_params, sparse=True, reader=frame_utils.readDispKITTI)
        assert os.path.exists(root)

        if year == 2012:
            root_12 = '/data/StereoDatasets/kitti/2012'
            image1_list = sorted(glob(os.path.join(root_12, image_set, 'colored_0/*_10.png')))
            image2_list = sorted(glob(os.path.join(root_12, image_set, 'colored_1/*_10.png')))
            disp_list = sorted(glob(os.path.join(root_12, 'training', 'disp_occ/*_10.png'))) if image_set == 'training' else [
                                                                                                                                 osp.join(root, 'training/disp_occ/000085_10.png')] * len(image1_list)

        if year == 2015:
            root_15 = '/data/StereoDatasets/kitti/2015'
            image1_list = sorted(glob(os.path.join(root_15, image_set, 'image_2/*_10.png')))
            image2_list = sorted(glob(os.path.join(root_15, image_set, 'image_3/*_10.png')))
            disp_list = sorted(glob(os.path.join(root_15, 'training', 'disp_occ_0/*_10.png'))) if image_set == 'training' else [
                                                                                                                                   osp.join(root, 'training/disp_occ_0/000085_10.png')] * len(image1_list)

        for idx, (img1, img2, disp) in enumerate(zip(image1_list, image2_list, disp_list)):
            self.image_list += [[img1, img2]]
            self.disparity_list += [disp]


class VKITTI2(StereoDataset):
    def __init__(self, aug_params=None, root='/data/StereoDatasets/vkitti2'):
        super(VKITTI2, self).__init__(aug_params, sparse=True, reader=frame_utils.readDispVKITTI2)
        assert os.path.exists(root)

        image1_list = sorted(glob(os.path.join(root, 'Scene*/*/frames/rgb/Camera_0/rgb*.jpg')))
        image2_list = sorted(glob(os.path.join(root, 'Scene*/*/frames/rgb/Camera_1/rgb*.jpg')))
        disp_list = sorted(glob(os.path.join(root, 'Scene*/*/frames/depth/Camera_0/depth*.png')))

        assert len(image1_list) == len(image2_list) == len(disp_list)

        for idx, (img1, img2, disp) in enumerate(zip(image1_list, image2_list, disp_list)):
            self.image_list += [[img1, img2]]
            self.disparity_list += [disp]


class Middlebury(StereoDataset):
    def __init__(self, aug_params=None, root='D:/project2024/stereo matching/DATA/middlebury', split='2014',
                 resolution='F'):
        super(Middlebury, self).__init__(aug_params, sparse=True, reader=frame_utils.readDispMiddlebury)
        assert os.path.exists(root)
        assert split in ["2005", "2006", "2014", "2021", "MiddEval3"]
        if split == "2005":
            scenes = list((Path(root) / "2005").glob("*"))
            for scene in scenes:
                if os.path.exists(str(scene / "disp1.png")):
                    self.image_list += [[str(scene / "view1.png"), str(scene / "view5.png")]]
                    self.disparity_list += [str(scene / "disp1.png")]
                    for illum in ["1", "2", "3"]:
                        for exp in ["0", "1", "2"]:
                            self.image_list += [[str(scene / f"Illum{illum}/Exp{exp}/view1.png"),
                                                 str(scene / f"Illum{illum}/Exp{exp}/view5.png")]]
                            self.disparity_list += [str(scene / "disp1.png")]
            for idx, data in enumerate(self.image_list):
                if os.path.exists(data[0]) and os.path.exists(data[1]) and os.path.exists(self.disparity_list[idx]):
                    # print('is ok')
                    pass
                else:
                    print(data[0], data[1], self.disparity_list[idx])
                    pass

        elif split == "2006":
            scenes = list((Path(root) / "2006").glob("*"))
            for scene in scenes:
                self.image_list += [[str(scene / "view1.png"), str(scene / "view5.png")]]
                self.disparity_list += [str(scene / "disp1.png")]
                for illum in ["1", "2", "3"]:
                    for exp in ["0", "1", "2"]:
                        self.image_list += [[str(scene / f"Illum{illum}/Exp{exp}/view1.png"),
                                             str(scene / f"Illum{illum}/Exp{exp}/view5.png")]]
                        self.disparity_list += [str(scene / "disp1.png")]

        elif split == "2014":
            scenes = list((Path(root) / "2014").glob("*"))
            for scene in scenes:
                for s in ["E", "L", ""]:
                    self.image_list += [[str(scene / "im0.png"), str(scene / f"im1{s}.png")]]
                    self.disparity_list += [str(scene / "disp0.pfm")]
            # print(len(self.image_list), len(self.disparity_list))
            # print(self.image_list[0], self.disparity_list[0])
            # for idx, data in enumerate(self.image_list):
            #     if os.path.exists(data[0]) and os.path.exists(data[1]) and os.path.exists(self.disparity_list[idx]):
            #         pass
            #     else:
            #         print(data[0], data[1], self.disparity_list[idx])
            #         print('is not ok')
        elif split == "2021":
            scenes = list((Path(root) / "2021/data").glob("*"))
            for scene in scenes:
                self.image_list += [[str(scene / "im0.png"), str(scene / "im1.png")]]
                self.disparity_list += [str(scene / "disp0.pfm")]
                for s in ["0", "1", "2", "3"]:
                    if os.path.exists(str(scene / f"ambient/L0/im0e{s}.png")):
                        self.image_list += [
                            [str(scene / f"ambient/L0/im0e{s}.png"), str(scene / f"ambient/L0/im1e{s}.png")]]
                        self.disparity_list += [str(scene / "disp0.pfm")]
                        print(len(self.image_list), len(self.disparity_list))
            # print(self.image_list[0], self.disparity_list[0])
            # for idx, data in enumerate(self.image_list):
            #     if os.path.exists(data[0]) and os.path.exists(data[1]) and os.path.exists(self.disparity_list[idx]):
            #         pass
            #     else:
            #         print(data[0], data[1], self.disparity_list[idx])
            #         print('is not ok')
        else:
            image1_list = sorted(glob(os.path.join(root, "MiddEval3", f'training{resolution}', '*/im0.png')))
            image2_list = sorted(glob(os.path.join(root, "MiddEval3", f'training{resolution}', '*/im1.png')))
            disp_list = sorted(glob(os.path.join(root, "MiddEval3", f'training{resolution}', '*/disp0GT.pfm')))
            assert len(image1_list) == len(image2_list) == len(disp_list) > 0, [image1_list, split]
            for img1, img2, disp in zip(image1_list, image2_list, disp_list):
                self.image_list += [[img1, img2]]
                self.disparity_list += [disp]
            # for idx, data in enumerate(self.image_list):
            #     if os.path.exists(data[0]) and os.path.exists(data[1]) and os.path.exists(self.disparity_list[idx]):
            #         print('is ok')
            #     else:
            #         print(data[0], data[1], self.disparity_list[idx])
            #         pass


def fetch_dataloader(args):
    """ Create the data loader for the corresponding trainign set """

    aug_params = {'crop_size': args.image_size, 'min_scale': args.spatial_scale[0], 'max_scale': args.spatial_scale[1],
                  'do_flip': False, 'yjitter': not args.noyjitter}
    if hasattr(args, "saturation_range") and args.saturation_range is not None:
        aug_params["saturation_range"] = args.saturation_range
    if hasattr(args, "img_gamma") and args.img_gamma is not None:
        aug_params["gamma"] = args.img_gamma
    if hasattr(args, "do_flip") and args.do_flip is not None:
        aug_params["do_flip"] = args.do_flip

    train_dataset = None
    # for dataset_name in args.train_datasets:
    if args.train_datasets == 'sceneflow':
        aug_params['spatial_scale'] = False
        new_dataset = SceneFlowDatasets(aug_params, dstype='frames_finalpass')
        logging.info(f"Adding {len(new_dataset)} samples from SceneFlow")
    elif args.train_datasets == 'vkitti2':
        new_dataset = VKITTI2(aug_params)
        logging.info(f"Adding {len(new_dataset)} samples from VKITTI2")
    elif args.train_datasets == 'kitti':
        kitti12 = KITTI(aug_params, year=2012)
        logging.info(f"Adding {len(kitti12)} samples from KITTI 2012")
        kitti15 = KITTI(aug_params, year=2015)
        logging.info(f"Adding {len(kitti15)} samples from KITTI 2015")
        new_dataset = kitti12 + kitti15
        logging.info(f"Adding {len(new_dataset)} samples from KITTI")
    elif args.train_datasets == 'eth3d_train':
        tartanair = TartanAir(aug_params)
        logging.info(f"Adding {len(tartanair)} samples from Tartain Air")
        sceneflow = SceneFlowDatasets(aug_params, dstype='frames_finalpass')
        logging.info(f"Adding {len(sceneflow)} samples from SceneFlow")
        sintel = SintelStereo(aug_params)
        logging.info(f"Adding {len(sintel)} samples from Sintel Stereo")
        crestereo = CREStereoDataset(aug_params)
        logging.info(f"Adding {len(crestereo)} samples from CREStereo Dataset")
        eth3d = ETH3D(aug_params)
        logging.info(f"Adding {len(eth3d)} samples from ETH3D")
        instereo2k = InStereo2K(aug_params)
        logging.info(f"Adding {len(instereo2k)} samples from InStereo2K")
        new_dataset = tartanair + sceneflow + sintel * 50 + eth3d * 1000 + instereo2k * 100 + crestereo * 2
        logging.info(f"Adding {len(new_dataset)} samples from ETH3D Mixture Dataset")
    elif args.train_datasets == 'eth3d_finetune':
        crestereo = CREStereoDataset(aug_params)
        logging.info(f"Adding {len(crestereo)} samples from CREStereo Dataset")
        eth3d = ETH3D(aug_params)
        logging.info(f"Adding {len(eth3d)} samples from ETH3D")
        instereo2k = InStereo2K(aug_params)
        logging.info(f"Adding {len(instereo2k)} samples from InStereo2K")
        new_dataset = eth3d * 1000 + instereo2k * 10 + crestereo
        logging.info(f"Adding {len(new_dataset)} samples from ETH3D Mixture Dataset")
    elif args.train_datasets == 'middlebury_train':
        tartanair = TartanAir(aug_params)
        logging.info(f"Adding {len(tartanair)} samples from Tartain Air")
        sceneflow = SceneFlowDatasets(aug_params, dstype='frames_finalpass')
        logging.info(f"Adding {len(sceneflow)} samples from SceneFlow")
        fallingthings = FallingThings(aug_params)
        logging.info(f"Adding {len(fallingthings)} samples from FallingThings")
        carla = CARLA(aug_params)
        logging.info(f"Adding {len(carla)} samples from CARLA")
        crestereo = CREStereoDataset(aug_params)
        logging.info(f"Adding {len(crestereo)} samples from CREStereo Dataset")
        instereo2k = InStereo2K(aug_params)
        logging.info(f"Adding {len(instereo2k)} samples from InStereo2K")
        mb2005 = Middlebury(aug_params, split='2005')
        logging.info(f"Adding {len(mb2005)} samples from Middlebury 2005")
        mb2006 = Middlebury(aug_params, split='2006')
        logging.info(f"Adding {len(mb2006)} samples from Middlebury 2006")
        mb2014 = Middlebury(aug_params, split='2014')
        logging.info(f"Adding {len(mb2014)} samples from Middlebury 2014")
        mb2021 = Middlebury(aug_params, split='2021')
        logging.info(f"Adding {len(mb2021)} samples from Middlebury 2021")
        mbeval3 = Middlebury(aug_params, split='MiddEval3', resolution='H')
        logging.info(f"Adding {len(mbeval3)} samples from Middlebury Eval3")
        new_dataset = tartanair + sceneflow + fallingthings + instereo2k * 50 + carla * 50 + crestereo + mb2005 * 200 + mb2006 * 200 + mb2014 * 200 + mb2021 * 200 + mbeval3 * 200
        logging.info(f"Adding {len(new_dataset)} samples from Middlebury Mixture Dataset")
    elif args.train_datasets == 'middlebury_finetune':
        crestereo = CREStereoDataset(aug_params)
        logging.info(f"Adding {len(crestereo)} samples from CREStereo Dataset")
        instereo2k = InStereo2K(aug_params)
        logging.info(f"Adding {len(instereo2k)} samples from InStereo2K")
        carla = CARLA(aug_params)
        logging.info(f"Adding {len(carla)} samples from CARLA")
        mb2005 = Middlebury(aug_params, split='2005')
        logging.info(f"Adding {len(mb2005)} samples from Middlebury 2005")
        mb2006 = Middlebury(aug_params, split='2006')
        logging.info(f"Adding {len(mb2006)} samples from Middlebury 2006")
        mb2014 = Middlebury(aug_params, split='2014')
        logging.info(f"Adding {len(mb2014)} samples from Middlebury 2014")
        mb2021 = Middlebury(aug_params, split='2021')
        logging.info(f"Adding {len(mb2021)} samples from Middlebury 2021")
        mbeval3 = Middlebury(aug_params, split='MiddEval3', resolution='H')
        logging.info(f"Adding {len(mbeval3)} samples from Middlebury Eval3")
        mbeval3_f = Middlebury(aug_params, split='MiddEval3', resolution='F')
        logging.info(f"Adding {len(mbeval3)} samples from Middlebury Eval3")
        fallingthings = FallingThings(aug_params)
        logging.info(f"Adding {len(fallingthings)} samples from FallingThings")
        new_dataset = crestereo + instereo2k * 50 + carla * 50 + mb2005 * 200 + mb2006 * 200 + mb2014 * 200 + mb2021 * 200 + mbeval3 * 200 + mbeval3_f * 200 + fallingthings * 10
        logging.info(f"Adding {len(new_dataset)} samples from Middlebury Mixture Dataset")
    elif args.train_datasets == 'only_sceneflow':
        sceneflow_train = SceneFlowDatasets(aug_params, dstype='frames_finalpass')
        sceneflow_test = SceneFlowDatasets(aug_params, dstype='frames_finalpass', things_test=True)
        logging.info(f"Adding {len(sceneflow_train)} samples from SceneFlow")
        logging.info(f"Adding {len(sceneflow_test)} samples from SceneFlow")
        new_dataset = sceneflow_train
    elif args.train_datasets == 'middlebury_sceneflow':
        mb2005 = Middlebury(aug_params, split='2005')
        logging.info(f"Adding {len(mb2005)} samples from Middlebury 2005")
        mb2006 = Middlebury(aug_params, split='2006')
        logging.info(f"Adding {len(mb2006)} samples from Middlebury 2006")
        mb2014 = Middlebury(aug_params, split='2014')
        logging.info(f"Adding {len(mb2014)} samples from Middlebury 2014")
        mb2021 = Middlebury(aug_params, split='2021')
        logging.info(f"Adding {len(mb2021)} samples from Middlebury 2021")
        mbeval3 = Middlebury(aug_params, split='MiddEval3', resolution='H')
        logging.info(f"Adding {len(mbeval3)} samples from Middlebury Eval3")
        mbeval3_f = Middlebury(aug_params, split='MiddEval3', resolution='F')
        logging.info(f"Adding {len(mbeval3)} samples from Middlebury Eval3")

        sceneflow_train = SceneFlowDatasets(aug_params, dstype='frames_finalpass')
        sceneflow_test = SceneFlowDatasets(aug_params, dstype='frames_finalpass', things_test=True)
        logging.info(f"Adding {len(sceneflow_train)} samples from SceneFlow")
        logging.info(f"Adding {len(sceneflow_test)} samples from SceneFlow")
        new_dataset = sceneflow_train + mb2005 * 200 + mb2006 * 200 + mb2014 * 200 + mb2021 * 200 + mbeval3 * 200 + mbeval3_f * 200
        print(f"Adding {len(new_dataset)} samples from train_dataset")
        print(f"Adding {len(sceneflow_test)} samples from test_dataset")

    train_dataset = new_dataset if train_dataset is None else train_dataset + new_dataset
    test_dataset = sceneflow_test
    # train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size,
    #                                pin_memory=True, shuffle=True, num_workers=8, drop_last=True)

    logging.info('Training with %d image pairs' % len(train_dataset))
    return train_dataset, test_dataset


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='rt-igev-stereo', help="name your experiment")
    parser.add_argument('--restore_ckpt', default=None, help='load the weights from a specific checkpoint')
    parser.add_argument('--logdir', default='./checkpoints_rt', help='the directory to save logs and checkpoints')
    parser.add_argument('--mixed_precision', default=True, action='store_true', help='use mixed precision')
    parser.add_argument('--precision_dtype', default='float16', choices=['float16', 'bfloat16',
                                                                         'float32'], help='Choose precision type: float16 or bfloat16 or float32')

    # Training parameters
    parser.add_argument('--batch_size', type=int, default=8, help="batch size used during training.")
    parser.add_argument('--train_datasets', default='middlebury_sceneflow', choices=['sceneflow', 'kitti',
                                                                               'middlebury_small_train',
                                                                               'only_sceneflow'], help="training datasets.")
    parser.add_argument('--lr', type=float, default=0.0002, help="max learning rate.")
    parser.add_argument('--num_steps', type=int, default=200000, help="length of training schedule.")
    parser.add_argument('--image_size', type=int, nargs='+', default=[320,
                                                                      768], help="size of the random image crops used during training.")
    parser.add_argument('--train_iters', type=int, default=22, help="number of updates to the disparity field in each forward pass.")
    parser.add_argument('--wdecay', type=float, default=.00001, help="Weight decay in optimizer.")

    # Validation parameters
    parser.add_argument('--valid_iters', type=int, default=32, help='number of flow-field updates during validation forward pass')

    # Architecure choices
    parser.add_argument('--corr_levels', type=int, default=2, help="number of levels in the correlation pyramid")
    parser.add_argument('--corr_radius', type=int, default=4, help="width of the correlation pyramid")
    parser.add_argument('--n_downsample', type=int, default=2, help="resolution of the disparity field (1/2^K)")
    parser.add_argument('--slow_fast_gru', action='store_true', help="iterate the low-res GRUs more frequently")
    parser.add_argument('--n_gru_layers', type=int, default=1, help="number of hidden GRU levels")
    parser.add_argument('--hidden_dim', nargs='+', type=int, default=96, help="hidden state and context dimensions")
    parser.add_argument('--max_disp', type=int, default=192, help="max disp of geometry encoding volume")

    # Data augmentation
    parser.add_argument('--img_gamma', type=float, nargs='+', default=None, help="gamma range")
    parser.add_argument('--saturation_range', type=float, nargs='+', default=[0, 1.4], help='color saturation')
    parser.add_argument('--do_flip', default=False, choices=['h',
                                                             'v'], help='flip the images horizontally or vertically')
    parser.add_argument('--spatial_scale', type=float, nargs='+', default=[-0.4,
                                                                           0.8], help='re-scale the images randomly')
    parser.add_argument('--noyjitter', action='store_true', help='don\'t simulate imperfect rectification')
    args = parser.parse_args()
    train_dataset, test_dataset = fetch_dataloader(args)

    from torch.utils.data import ConcatDataset


    def extract_image_and_disp_lists(dataset):
        image_list = []
        disparity_list = []

        if isinstance(dataset, ConcatDataset):
            for sub_dataset in dataset.datasets:
                sub_images, sub_disps = extract_image_and_disp_lists(sub_dataset)
                image_list.extend(sub_images)
                disparity_list.extend(sub_disps)
        else:
            if hasattr(dataset, 'image_list') and hasattr(dataset, 'disparity_list'):
                image_list.extend(dataset.image_list)
                disparity_list.extend(dataset.disparity_list)

        return image_list, disparity_list
    # 将datase里的image_liust和disp数据路径保存到txt里，每行左图、右图、disp
    replace_root = "D:/project2024/stereo matching/DATA/"
    replace_with = ""  # 你可以改成 "./" 或 "relative/path/" 等
    train_image_list, train_disp_list = extract_image_and_disp_lists(train_dataset)
    test_image_list, test_disp_list = extract_image_and_disp_lists(test_dataset)

    # 写入训练集
    with open('train_dataset_middlebury_sceneflow.txt', 'w') as f:
        for img_pair, disp in zip(train_image_list, train_disp_list):
            if 'disp1.png' in disp:
                continue
            left_img = img_pair[0].replace('\\', '/').replace(replace_root, replace_with)
            right_img = img_pair[1].replace('\\', '/').replace(replace_root, replace_with)
            disp_path = disp.replace('\\', '/').replace(replace_root, replace_with)
            f.write(f"{left_img} {right_img} {disp_path}\n")

    # 写入测试集
    with open('test_dataset_middlebury_sceneflow.txt', 'w') as f:
        for img_pair, disp in zip(test_image_list, test_disp_list):
            if 'disp1.png' in disp:
                continue
            left_img = img_pair[0].replace('\\', '/').replace(replace_root, replace_with)
            right_img = img_pair[1].replace('\\', '/').replace(replace_root, replace_with)
            disp_path = disp.replace('\\', '/').replace(replace_root, replace_with)
            f.write(f"{left_img} {right_img} {disp_path}\n")
