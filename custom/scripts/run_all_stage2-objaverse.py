import os
import subprocess
from multiprocessing import Process
from numpy import random
random.seed(0)
import json
import glob

import argparse
import yaml


def divide_list(objects, num_batches):
    # Calculate the base size of each batch
    base_size = len(objects) // num_batches
    # Calculate the remainder to distribute among the batches
    remainder = len(objects) % num_batches

    # Create the batches
    batches = []
    start = 0
    for i in range(num_batches):
        # Distribute the extra element if the remainder is greater than the index
        extra = 1 if i < remainder else 0
        end = start + base_size + extra
        batches.append(objects[start:end])
        start = end

    return batches


def run(objects, gpu):
    print(f'gpu: {gpu}')
    outpus_dir = '/home/lab/rahamiob/ohadr/projects/object_animation/outputs-objaverse/animate124-stage1'
    object_paths = glob.glob(os.path.join(outpus_dir, "**", "*.ckpt"), recursive=True)
    for object in objects:
        object_dir = object[0]
        rotation = object[1]
        prompt = object[2]
        name = object[3]
        camera = 3.0
        object_paths_ = [path for path in object_paths if name in path]

        for ckpt in object_paths_:
            parts = ckpt.split('/')
            parts = parts[-3].split('_')
            location = parts[-2]
            file = parts[2].split('@')[0]
            object_path = os.path.join(object_dir, 'glbs', parts[1], file + '.glb')
            line = ('--config custom/configs/animate124-stage2-objaverse-ms.yaml --train --gpu 1 ' +
                    f'data.image.image_path={object_path} system.prompt_processor.prompt="{prompt}" ' +
                    f'data.image.rotate=4 ' +
                    f'data.single_view.camera_sampling=incremental ' +
                    f'system.loggers.wandb.enable=True ' +
                    f'system.loggers.wandb.project=object-objaverse ' +
                    f'system.loggers.wandb.name=objaverse-stage2/{name}_{location}_{file} ' +
                    f'data.image.random_camera.camera_distance_range=[{camera},{camera}] ' +
                    f'data.image.random_camera.eval_camera_distance={camera} ' +
                    f'data.single_view.camera_distance_range=[{camera},{camera}] ' +
                    f'data.single_view.eval_camera_distance={camera} ' +
                    # f'system.guidance.loss_attention_masking=True ' +
                    f'system.guidance.faces_noise=True ' +
                    f'data.image.noise_reduction=1 ' +
                    f'data.image.sphere_scale=1 ' +
                    f'data.image.sphere_level=5 ' +
                    f'data.single_view.round=True ' +
                    f'system.weights={ckpt} '
                    f'exp_root_dir=outputs-objaverse'
                    )
            print(line)
            cmd = (f'export HF_HOME="/cortex/hf_cache"; export CUDA_VISIBLE_DEVICES={gpu}; MKL_THREADING_LAYER=GNU; ' +
                   f'/cortex/users/ohadr/envs/envs/animate_h100/bin/python launch.py {line}')
            os.system(cmd)

def main_batch(index, gpu):
    file_path = 'custom/scripts/objects.yaml' # 'custom/scripts/objects.yaml'
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)

    num_gpus = 7  # Change this to the number of available GPUs
    objects = data['stage2-objaverse']

    # batch_size = len(objects) // num_gpus  # Calculate the size of each batch
    # batches = [objects[i:i + batch_size] for i in range(0, len(objects), batch_size)]
    batches = divide_list(objects, num_gpus)

    run(objects=batches[index], gpu=gpu)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Your script description here")
    parser.add_argument('--index', type=int, help='Index argument')
    parser.add_argument('--gpu', type=int, help='GPU argument')
    args = parser.parse_args()

    if args.index is None or args.gpu is None:
        parser.error("You must provide both --index and --gpu arguments.")
    else:
        main_batch(index=args.index, gpu=args.gpu)