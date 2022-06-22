from data.base_dataset import BaseDataset
import os
import cv2
from PIL import Image, ImageFilter
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import json
from skimage import morphology

def get_transforms(transform_variant, out_size):
    if transform_variant == "train":
        transform = A.Compose([
            A.SmallestMaxSize(out_size),
            A.RandomCrop(out_size, out_size),
            A.Rotate(30, border_mode=cv2.BORDER_REFLECT101),
            A.HorizontalFlip(),
            A.Normalize((0.5), (0.5)),
            ToTensorV2()
        ], additional_targets={"image1": "image", "image2": "image"})

    elif transform_variant == "val" or transform_variant == "test":
        transform = A.Compose([
            A.Resize(out_size, out_size),
            A.Normalize((0.5),(0.5)),
            ToTensorV2()
        ], additional_targets={"image1": "image", "image2": "image"})
    else:
        transform = None
    return transform


class SketchShadeDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot

        ### input A (label maps)
        if opt.phase == "test":
            self.A_paths = json.load(open(f"{self.root}/val.lst"))
        else:
            self.A_paths = json.load(open(f"{self.root}/train.lst"))
        self.A_paths = [os.path.splitext(p)[0] for p in self.A_paths]

        ### input B (real images)
        if opt.isTrain or opt.use_encoded_image:
            self.B_paths = self.A_paths

        ### instance maps
        if not opt.no_instance:
            #self.dir_inst = os.path.join(opt.dataroot, opt.phase + '_inst')
            #self.inst_paths = sorted(make_dataset(self.dir_inst))
            pass

        ### load precomputed instance-wise encoded features
        if opt.load_features:
            #self.dir_feat = os.path.join(opt.dataroot, opt.phase + '_feat')
            #print('----------- loading features from %s ----------' % self.dir_feat)
            #self.feat_paths = sorted(make_dataset(self.dir_feat))
            pass

        self.transform = get_transforms(self.opt.phase, self.opt.loadSize)
        self.dataset_size = len(self.A_paths)

    def __getitem__(self, index):
        ### input A (label maps)
        sketch_path = f"{self.root}/sketch_2edge/{self.A_paths[index]}.png"
        sketch = np.array(Image.open(sketch_path).convert('L'))

        ### input B (real images)
        if self.opt.isTrain or self.opt.use_encoded_image:
            shade_path = f"{self.root}/shade/{self.B_paths[index]}.png"
            #mask_path = f"{self.root}/mask/{self.B_paths[index]}.jpg"
            shade = np.array(Image.open(shade_path).convert('L'))
            #mask = np.array(Image.open(mask_path).convert('L'))
            trans = self.transform(image=sketch,
                                   image1=shade)
            input_dict = {'label': trans["image"], 'inst': 0,
                          'image': trans["image1"],
                          'feat': 0, 'path': sketch_path}
        else:
            trans = self.transform(image=sketch)
            input_dict = {'label': trans["image"], 'inst': 0, 'image': 0,
                          'feat': 0, 'path': sketch_path}

        return input_dict

    def __len__(self):
        return len(self.A_paths) // self.opt.batchSize * self.opt.batchSize

    def name(self):
        return 'SketchShadeDataset'

