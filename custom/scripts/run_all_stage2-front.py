import os
import subprocess
from multiprocessing import Process
from numpy import random
random.seed(0)
import json

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
    for object in objects:
        object_path = object[0]
        rotation = object[1]
        prompt = object[2]
        name = object[3]
        ckpt = object[4]
        if len(object) == 6:
            idx = object[5]
        else:
            idx = 0
        line = ('--config custom/configs/animate124-stage2-front-ms.yaml --train --gpu 1 ' +
                f'data.image.image_path={object_path} system.prompt_processor.prompt="{prompt}" ' +
                f'data.image.rotate={rotation} ' +
                f'system.prompt_processor.obj_token_clip_idx={idx} ' +
                f'data.single_view.camera_sampling=incremental ' +
                f'data.single_view.front_only=True ' +
                f'system.loggers.wandb.enable=True ' +
                f'system.loggers.wandb.project=object-animation-evolution ' +
                f'system.loggers.wandb.name=front-scale_10/{name} ' +
                f'system.weights={ckpt} ' +
                f'system.guidance.guidance_scale=10 ' +
                # f'data.single_view.sample_rand_frames=else ' +
                # f'system.guidance.faces_noise=True ' +
                # f'system.guidance.first_clean=True ' +
                f'exp_root_dir=output-front-object'
                )
        print(line)
        cmd = (f'export CUDA_VISIBLE_DEVICES={gpu}; MKL_THREADING_LAYER=GNU; ' +
               f'/cortex/users/ohadr/envs/envs/animate_h100/bin/python launch.py {line}')
        os.system(cmd)

def main_batch(index, gpu):
    file_path = 'custom/scripts/objects.yaml'
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)

    num_gpus = 2  # Change this to the number of available GPUs
    objects = data['stage2-evolution']

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