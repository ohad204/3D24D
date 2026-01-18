import os
import numpy as np

import torch
from tqdm import tqdm
from PIL import Image, ImageSequence
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, ToPILImage
from torchvision import transforms
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
    BILINEAR = InterpolationMode.BILINEAR
except ImportError:
    BICUBIC = Image.BICUBIC
    BILINEAR = Image.BILINEAR
from custom.eval.VICLIP.viclip import ViCLIP
from custom.eval.VICLIP.simple_tokenizer import SimpleTokenizer

CACHE_DIR = os.path.join(os.path.expanduser('~'), '.cache', 'vbench')

def clip_transform(n_px):
    return Compose([
        Resize(n_px, interpolation=BICUBIC),
        CenterCrop(n_px),
        transforms.Lambda(lambda x: x.float()),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

def get_text_features(model, input_text, tokenizer, text_feature_dict={}):
    if input_text in text_feature_dict:
        return text_feature_dict[input_text]
    text_template= f"{input_text}"
    with torch.no_grad():
        text_features = model.encode_text(text_template).float()
        text_features /= text_features.norm(dim=-1, keepdim=True)
        text_feature_dict[input_text] = text_features
    return text_features

def get_vid_features(model, input_frames):
    with torch.no_grad():
        clip_feat = model.encode_vision(input_frames,test=True).float()
        clip_feat /= clip_feat.norm(dim=-1, keepdim=True)
    return clip_feat

def get_predict_label(clip_feature, text_feats_tensor, top=5):
    label_probs = (100.0 * clip_feature @ text_feats_tensor.T).softmax(dim=-1)
    top_probs, top_labels = label_probs.cpu().topk(top, dim=-1)
    return top_probs, top_labels

def temporal_style(video, query, device='cuda'):
    tokenizer = SimpleTokenizer('custom/eval/VICLIP/bpe_simple_vocab_16e6.txt.gz')
    pretrain_path = '/home/lab/rahamiob/ohadr/models/VICLIP/ViClip-InternVid-10M-FLT.pth'
    clip_model = ViCLIP(tokenizer=tokenizer, pretrain=pretrain_path).to(device)

    image_transform = clip_transform(224)
    with torch.no_grad():
        # images = load_video(video_path, num_frames=8)
        images = image_transform(video[::8])
        clip_feat = get_vid_features(clip_model, images.unsqueeze(0))
        text_feat = get_text_features(clip_model, query, tokenizer)
        logit_per_text =  clip_feat @ text_feat.T
        score =  float(logit_per_text[0][0].cpu())
    return score

# def compute_temporal_style(json_dir, device, submodules_list, **kwargs):
#     tokenizer = SimpleTokenizer(os.path.join(CACHE_DIR, "ViCLIP/bpe_simple_vocab_16e6.txt.gz"))
#     viclip = ViCLIP(tokenizer= tokenizer, **submodules_list).to(device)
#     _, video_dict = load_dimension_info(json_dir, dimension='temporal_style', lang='en')
#     video_dict = distribute_list_to_rank(video_dict)
#     all_results, video_results = temporal_style(viclip, video_dict, tokenizer, device)
#     if get_world_size() > 1:
#         video_results = gather_list_of_dict(video_results)
#         all_results = sum([d['video_results'] for d in video_results]) / len(video_results)
#     return all_results, video_results