import torch
from PIL import Image, ImageFilter
from skimage import morphology, measure
from skimage.segmentation import flood_fill
import torchvision.transforms as transforms
from models.pix2pixHD_model import InferenceModel
import numpy as np
from argparse import Namespace
import os
import time

def tensor2im(image_tensor, imtype=np.uint8, normalize=True):
    image_numpy = image_tensor.detach().cpu().float().numpy()
    if normalize:
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    else:
        image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0
    image_numpy = np.clip(image_numpy, 0, 255)
    if image_numpy.shape[2] == 1 or image_numpy.shape[2] > 3:
        image_numpy = image_numpy[:,:,0]
    return image_numpy.astype(imtype)


def get_crop_bbox(mask_arr):
    y, x = np.nonzero(mask_arr)
    ymin = np.min(y)
    ymax = np.max(y)
    xmin = np.min(x)
    xmax = np.max(x)
    bbox = [max(xmin - 20, 0),
            max(ymin - 20, 0),
            min(xmax + 20, mask_arr.shape[1]),
            min(ymax + 20, mask_arr.shape[0])]
    return bbox


def get_mask_whitebg(img_arr, line):
    img_arr = (img_arr > line).astype(np.uint8)*255
    h, w= img_arr.shape[:2]
    img_tmp = np.ones((h+2, w+2))*255
    img_tmp[1:-1, 1:-1] = img_arr
    mask = flood_fill(img_tmp/255., (0, 0), -1, tolerance=0.1)
    mask_bin = (mask[1:-1, 1:-1] >= 0).astype(np.uint8)
    return mask_bin


def find_lagest_area(img_arr):
    label_img, num = measure.label(img_arr, background=0,
                                   return_num=True)

    if num <= 1:
        return img_arr

    max_label = 0
    max_num = 0
    for i in range(1, num + 1):
        num = np.sum(label_img == i)
        if num > max_num:
            max_num = num
            max_label = i
    res = (label_img == max_label).astype(np.uint8)
    return res


def skeleton(gray, line):
    binary = (gray < line).astype(np.uint8)
    skeleton0 = morphology.skeletonize(binary)
    skeleton = skeleton0.astype(np.uint8)*255
    return skeleton


def sketch2edge(sketch_arr):
    # skeleton and set the line size 3
    sk1_arr = skeleton(sketch_arr, 224)
    sk1_pil = Image.fromarray(255-sk1_arr).filter(ImageFilter.MinFilter(size=3))
    sk1_pil.save("./sk1_pil.png")
    mask_bin = get_mask_whitebg(np.array(sk1_pil), 128)

    # Make the two edge sketch
    sk2_pil = sk1_pil.filter(ImageFilter.MinFilter(size=5))
    sk2_arr_bin = np.array(sk2_pil) > 128
    sk1_arr_bin = np.array(sk1_pil) > 128
    sknew_arr_bin = ((~sk2_arr_bin) & sk1_arr_bin).astype(np.uint8)

    mask_arr = mask_bin*128 + (1-mask_bin)*255
    img_arr = mask_arr*sknew_arr_bin
    return img_arr, mask_bin*255


class FoldModel(object):
    def __init__(self):
        opt = Namespace(name='avg', gpu_ids=[0], checkpoints_dir='./checkpoints', model='pix2pixHD',
                        norm='instance', use_dropout=False, data_type=16, verbose=False, fp16=False,
                        local_rank=0, batchSize=1, loadSize=800, fineSize=800, label_nc=0, input_nc=1, output_nc=1,
                        resize_or_crop='resize', serial_batches=True, netG='global', ngf=64, n_downsample_global=4,
                        n_blocks_global=9, n_blocks_local=3, n_local_enhancers=1, niter_fix_global=0, no_instance=True,
                        instance_feat=False, label_feat=False, feat_num=3, load_features=False, n_downsample_E=4, nef=16,
                        n_clusters=10, aspect_ratio=1.0, phase='test', which_epoch='40', how_many=-1,
                        cluster_path='features_clustered_010.npy', use_encoded_image=False,  isTrain=False)
        self.model = InferenceModel()
        self.model.initialize(opt)
        self.model.half()
        self.trans = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize((0.5),
                                                              (0.5))])

    def predict_(self, sketch_pil):
        raw_size = sketch_pil.size
        sketch_pil = sketch_pil.resize([800, 800])
        sketch_arr = np.array(sketch_pil)

        sk2edge_arr, mask_arr = sketch2edge(sketch_arr)
        sk2edge_pil = Image.fromarray(sk2edge_arr)
        sk2edge_pil.save("sk2edge_pil.png")
        #Image.fromarray(mask_arr).save("mask.png")
        img_tensor = self.trans(sk2edge_pil)

        label_tensor = img_tensor.unsqueeze(0).half()
        inst_tensor = torch.Tensor([0.])
        img_tensor = torch.Tensor([0.])

        beg = time.time()
        res_tensor = self.model.inference(label_tensor, inst_tensor, img_tensor)
        end = time.time()
        print(end - beg)

        res_arr = tensor2im(res_tensor[0])

        mask_arr_bin = (mask_arr > 128).astype(np.uint8)
        res_arr = res_arr * mask_arr_bin + (1 - mask_arr_bin) * 255
        res_pil = Image.fromarray(res_arr).resize(raw_size)
        return res_pil

    def predict(self, img_path):
        img_pil = Image.open(img_path).convert('L')
        img_pil_filter = img_pil.filter(ImageFilter.MinFilter(size=3))
        raw_size = img_pil.size
        res_img = Image.fromarray(
            np.ones((raw_size[1], raw_size[0]), dtype=np.uint8) * 255)
        mask_bin = get_mask_whitebg(np.array(img_pil_filter), 224)
        mask_bin = find_lagest_area(mask_bin)
        Image.fromarray(mask_bin*255).save("mask_bin.png")
        bbox = get_crop_bbox(mask_bin)
        # remove other rubbish
        img_white = Image.fromarray(np.ones((raw_size[1], raw_size[0]), dtype=np.uint8)*255)
        img_white.paste(img_pil, mask=Image.fromarray(mask_bin*255))
        img_pil_crop = img_white.crop(bbox)
        img_pil_crop.save("img_pil_crop.png")
        res_pil = self.predict_(img_pil_crop)
        res_img.paste(res_pil, bbox)
        return res_img


def test_single():
    #img_path = "/home/ubuntu/桌面/test2.jpeg"
    #mask_path = "./20211119-142037A_mask.png"
    img_path = "/data/Dataset/sketch_test/svg2/25.jpeg"
    model = FoldModel()
    import time
    beg = time.time()
    res_pil = model.predict(img_path)
    end = time.time()
    print(end - beg)
    res_pil.save('./out.png')


def test_exp():
    #exp_name = "sketch_test_white"
    exp_name = "svg2"
    os.makedirs(f"image/{exp_name}", exist_ok=True)
    base_path = f"/data/Dataset/sketch_test/{exp_name}" #/sketch
    imgs = os.listdir(f"{base_path}")

    model = FoldModel()
    for img in imgs:
        img_path = f"{base_path}/{img}"
        res_img = model.predict(img_path)
        res_img.save(f"./image/{exp_name}/{img}")

#test_single()
test_exp()