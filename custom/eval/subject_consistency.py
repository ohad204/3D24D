import numpy as np
from PIL import Image
from tqdm import tqdm
import os

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

from custom.eval.I2V.utils import load_video, load_i2v_dimension_info, dino_transform_internet, dino_transform_Image_internet
# from vbench2_beta_long.utils import dreamsim_transform, dreamsim_transform_Image, dinov2_transform, dinov2_transform_Image

def subject_consistency_dinov2(model, video_list, device='cuda', read_frame=False):
    sim = 0.0
    cnt = 0
    video_results = []
    if read_frame:
        image_transform = dinov2_transform_Image(224)
    else:
        image_transform = dinov2_transform(224)
    for video_path in tqdm(video_list):
        video_sim = 0.0
        if read_frame:
            video_path = video_path[:-4].replace('videos', 'frames').replace(' ', '_')
            tmp_paths = [os.path.join(video_path, f) for f in sorted(os.listdir(video_path))]
            images = []
            for tmp_path in tmp_paths:
                images.append(image_transform(Image.open(tmp_path)))
        else:
            images = load_video(video_path)
            images = image_transform(images)
        for i in range(len(images)):
            with torch.no_grad():
                image = images[i].unsqueeze(0)
                image = image.to(device)
                image_features = model(image)
                image_features = F.normalize(image_features, dim=-1, p=2)
                if i == 0:
                    first_image_features = image_features
                else:
                    sim_pre = max(0.0, F.cosine_similarity(former_image_features, image_features).item())
                    sim_fir = max(0.0, F.cosine_similarity(first_image_features, image_features).item())
                    cur_sim = (sim_pre + sim_fir) / 2
                    video_sim += cur_sim
                    cnt += 1
            former_image_features = image_features
        sim_per_images = video_sim / (len(images) - 1)
        sim += video_sim
        video_results.append({'video_path': video_path, 'video_results': sim_per_images})
    # sim_per_video = sim / (len(video_list) - 1)
    sim_per_frame = sim / cnt if cnt != 0 else None
    return sim_per_frame, video_results

def dinov2_transform(n_px):
    t = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(n_px),
        transforms.Lambda(lambda x: x.float().div(255.0)),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    return t


def dinov2_transform_Image(n_px):
    t = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(n_px),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    return t


def compute_i2v_subject(json_dir, device, submodules_list, **kwargs):
    dinov2_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
    all_results, video_results = subject_consistency_dinov2(dinov2_model, [], 'cuda')
    return all_results, video_results

if __name__=="__main__":
    compute_i2v_subject('', device='cuda', submodules_list=None)