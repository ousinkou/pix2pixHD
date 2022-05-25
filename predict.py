import torch
from PIL import Image, ImageFilter
import cv2
import torchvision.transforms as transforms
from models.pix2pixHD_model import InferenceModel
import numpy as np
from argparse import Namespace
import os

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

def sketch2edge(sketch_pil, mask_pil):
    bold_factor = 5
    sk_new = sketch_pil.filter(ImageFilter.MinFilter(size=bold_factor))
    sk_arr = np.array(sketch_pil) > 128
    sk_new_arr = np.array(sk_new) > 128
    sk_new2_arr = ((~sk_new_arr) & sk_arr).astype(np.uint8)
    mask_bin = (np.array(mask_pil) > 128).astype(np.uint8)
    mask =  mask_bin*128 + (1-mask_bin)*255
    #Image.fromarray(mask).save("./tmp1.png")
    #Image.fromarray(sk_new2_arr*255).save("./tmp2.png")
    img_arr = mask*sk_new2_arr
    return Image.fromarray(img_arr)

#img_path = "/data/Dataset/sketch_test/sketch_test3/sketch/4.jpg"
#mask_path = "/data/Dataset/sketch_test/sketch_test3/mask/4.jpg"
#img_path = "/data/Dataset/sketch_test/new/sketch/54733238_a6bd9d22714887c8cec4e52b25c5d621_0000_4.png"
#mask_path = "/data/Dataset/sketch_test/new/mask/54733238_a6bd9d22714887c8cec4e52b25c5d621_0000_4.png"

model = InferenceModel()
opt = Namespace(name='sketchshade0524',
                gpu_ids=[0],
                checkpoints_dir='./checkpoints',
                model='pix2pixHD',
                norm='instance',
                use_dropout=False,
                data_type=16,
                verbose=False,
                fp16=False,
                local_rank=0,
                batchSize=1, loadSize=800, fineSize=800, label_nc=0, input_nc=1, output_nc=1,
                resize_or_crop='resize', serial_batches=True, netG='global', ngf=64, n_downsample_global=4,
                n_blocks_global=9, n_blocks_local=3, n_local_enhancers=1, niter_fix_global=0,
                no_instance=True, instance_feat=False, label_feat=False, feat_num=3, load_features=False,
                n_downsample_E=4, nef=16, n_clusters=10, aspect_ratio=1.0, phase='test', which_epoch='latest', how_many=-1,
                cluster_path='features_clustered_010.npy', use_encoded_image=False,  isTrain=False)
model.initialize(opt)
model.half()

exp_name = "sketech_test4"
os.makedirs(f"image/{exp_name}", exist_ok=True)

base_path = f"/data/Dataset/sketch_test/{exp_name}"
imgs = os.listdir(f"{base_path}/sketch")
#base_path = "/data/Dataset/sketch_test/sketch_test_white"
#imgs = os.listdir(f"{base_path}/raw")

for img in imgs:
    img_path = f"{base_path}/sketch/{img}"
    mask_path = f"{base_path}/mask/{img}"
    img_pil = Image.open(img_path).convert('L')
    mask_pil = Image.open(mask_path).convert('L')

    """
    img_path = f"{base_path}/raw/{img}"
    img_pil = Image.open(img_path)
    img_pil.getchannel("A").save(f"{base_path}/mask/{img}")
    img_pil.convert("L").save(f"{base_path}/sketch/{img}")
    continue
    """

    sk2edge_pil = sketch2edge(img_pil, mask_pil)
    #sk2edge_pil.save("./image/sk2edge_pil.png")

    trans = transforms.Compose([transforms.Resize([800, 800], Image.BICUBIC),
             transforms.ToTensor(),
             transforms.Normalize((0.5),
                                  (0.5))])
    img_tensor=trans(sk2edge_pil)

    label_tensor = img_tensor.unsqueeze(0).half()
    inst_tensor = torch.Tensor([0.])
    img_tensor = torch.Tensor([0.])

    res_tensor = model.inference(label_tensor, inst_tensor, img_tensor)
    res_arr = tensor2im(res_tensor[0])
    mask_arr = np.array(mask_pil.resize([800, 800], Image.NEAREST))
    mask_arr_bin = (mask_arr > 128).astype(np.uint8)
    res_arr = res_arr*mask_arr_bin + (1-mask_arr_bin)*255

    Image.fromarray(res_arr).resize(img_pil.size).save(f"./image/{exp_name}/{img}")


