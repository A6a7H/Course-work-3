import sys
sys.path.append('..')
sys.path.append('.')

import io
import numpy as np
import cv2
import torch
import base64
from io import BytesIO
import PIL

from torch.autograd import Variable
from models.viton import Viton, make_image
from models.keypoint import KeyPointPredictor
from models.segmentation import Segmentator
from models.posenet import DensePosePredictor, HumanpartPredictor
from network_utils.network_utils import CPDataset
from utils.utils import convert_image
from torchvision import transforms
from utils.utils import get_params, get_transform
from models.pix2pixHD_model import create_model

from flask import Flask, jsonify, request
from PIL import Image

app = Flask(__name__)

transform = transforms.Compose([  \
        transforms.ToTensor(),   \
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        
transform_1d = transforms.Compose([ \
        transforms.ToTensor(), \
        transforms.Normalize((0.5,), (0.5,))])

def changearm(old_label):
    label=old_label
    arm1=torch.FloatTensor((old_label.cpu().numpy()==11).astype(np.int))
    arm2=torch.FloatTensor((old_label.cpu().numpy()==13).astype(np.int))
    noise=torch.FloatTensor((old_label.cpu().numpy()==7).astype(np.int))
    label=label*(1-arm1)+arm1*4
    label=label*(1-arm2)+arm2*4
    label=label*(1-noise)+noise*4
    return label

def pipeline_acgpn(img, cloth, cloth_mask, opt): 
    img_array = np.array(img)
    params = get_params(opt, img.size)
    transform_A = get_transform(opt, params, method=Image.NEAREST, normalize=False)
    transform_B = get_transform(opt, params)
    img = transform_B(img)
    

    hp = HumanpartPredictor("./checkpoint/densepose_rcnn_R_50_FPN_s1x.pkl")
    label = hp.predict(img)
    A_tensor = transform_A(label) * 255.0

    mask_clothes = torch.FloatTensor((label.cpu().numpy() == 4).astype(np.int))
    mask_fore = torch.FloatTensor((label.cpu().numpy() > 0).astype(np.int))
    img_fore = img * mask_fore
    img_fore_wc = img_fore * mask_fore
    all_clothes_label = changearm(label)

    key_model = KeyPointPredictor()
    pose_map, _ = key_model.predict(img_array)

    pose_map_corect = torch.zeros((18, 256, 192)) - 1 
    pose_map_corect[0] = pose_map[0]
    pose_map_corect[1] = pose_map[17]
    pose_map_corect[2] = pose_map[6]
    pose_map_corect[3] = pose_map[8]
    pose_map_corect[4] = pose_map[10]
    pose_map_corect[5] = pose_map[5]
    pose_map_corect[6] = pose_map[7]
    pose_map_corect[7] = pose_map[9]
    pose_map_corect[8] = pose_map[12]
    pose_map_corect[11] = pose_map[11]
    pose_map_corect[14] = pose_map[2]
    pose_map_corect[15] = pose_map[1]
    pose_map_corect[16] = pose_map[4]
    pose_map_corect[17] = pose_map[3]
    pose_map = torch.tensor(pose_map_corect).type(torch.float32)

    model_acgpn = create_model(opt)
    res = model_acgpn(
        Variable(label.cuda()),
        Variable(cloth_mask.cuda()),
        Variable(img_fore.cuda()),
        Variable(mask_clothes.cuda()),
        Variable(cloth.cuda()),
        Variable(all_clothes_label.cuda()),
        Variable(img_array.cuda()),
        Variable(pose_map.cuda()),
        Variable(img_array.cuda()),
        Variable(mask_fore.cuda())
        )
    return res

def pipeline(img, cloth, cloth_mask): 
    img_array = np.array(img)
    img_array = convert_image(img_array)
    img = Image.fromarray(convert_image(np.array(img)))
    img = transform(img)
    im_255 = cv2.resize(img_array, (255, 255), interpolation = cv2.INTER_AREA)
    dp = DensePosePredictor("./detectron2_repo/projects/DensePose/configs/densepose_rcnn_R_50_FPN_s1x.yaml", 
                            "./checkpoint/densepose_rcnn_R_50_FPN_s1x.pkl")
    phead, body = dp.predict(im_255)
    body = cv2.resize(body, (192, 256), interpolation = cv2.INTER_AREA)
    body = Image.fromarray((body*255).astype(np.uint8))
    body = body.resize((192//16, 256//16), Image.BILINEAR)
    body = body.resize((192, 256), Image.BILINEAR)
    phead = cv2.resize(phead, (192, 256), interpolation = cv2.INTER_AREA)
    phead = torch.from_numpy(phead[np.newaxis,:,:])
    head = img * phead - (1 - phead)
    shape = transform_1d(body)

    key_model = KeyPointPredictor()
    pose_map, _ = key_model.predict(img_array)

    pose_map_corect = torch.zeros((18, 256, 192)) - 1 
    pose_map_corect[0] = pose_map[0]
    pose_map_corect[1] = pose_map[17]
    pose_map_corect[2] = pose_map[6]
    pose_map_corect[3] = pose_map[8]
    pose_map_corect[4] = pose_map[10]
    pose_map_corect[5] = pose_map[5]
    pose_map_corect[6] = pose_map[7]
    pose_map_corect[7] = pose_map[9]
    pose_map_corect[8] = pose_map[12]
    pose_map_corect[11] = pose_map[11]
    pose_map_corect[14] = pose_map[2]
    pose_map_corect[15] = pose_map[1]
    pose_map_corect[16] = pose_map[4]
    pose_map_corect[17] = pose_map[3]
    pose_map = torch.tensor(pose_map_corect).type(torch.float32)

    viton = Viton(gmm_checkpoint_path='./checkpoint/gmm_train_new/gmm_final.pth', 
                tom_checkpoint_path='./checkpoint/tom_train_new/tom_final.pth', use_cuda=False)
    # shape = Image.open('./datasets/data/train/cloth_mask/000003_1.jpg') # path to cloth
    # shape = transform_1d(shape)
    #
    # cloth == color
    # cloth_mask == color_mask
    # 
    res = viton.run_viton(head, pose_map, shape, cloth, cloth_mask)
    return res

def decode(byte64_str):
    bytes = BytesIO(base64.b64decode(byte64_str))
    pil_img = PIL.Image.open(bytes)
    return pil_img


def encode(img):
    pil_img = Image.fromarray(img.astype('uint8'))
    pil_img = pil_img.convert('RGB')
    buff = BytesIO()
    pil_img.save(buff, format="JPEG")
    return base64.b64encode(buff.getvalue()).decode("utf-8")

@app.route('/style/api/v1.0/segmentate', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.values['file']
        cloth_id = int(request.values['cloth'])
        img = decode(file)
        if cloth_id == 0:
            c = Image.open('./datasets/data/train/cloth/000003_1.jpg') # path to cloth
            c = transform(c)
            cm = Image.open('./datasets/data/train/cloth-mask/000003_1.jpg') # path to cloth-mask
            cm_array = np.array(cm)
            cm_array = (cm_array >= 128).astype(np.float32)
            cm = torch.from_numpy(cm_array)
            cm.unsqueeze_(0)

        res = pipeline(img, c, cm)
    return encode(np.array(res[0].cpu().detach()).transpose(1,2,0) * 255)

if __name__ == '__main__':
    app.run(debug=True)