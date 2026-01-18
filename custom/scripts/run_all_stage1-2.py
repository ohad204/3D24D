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
        line = ('--config custom/configs/animate124-stage1.yaml --train --gpu 1 ' +
                f'data.image.image_path={object_path} system.prompt_processor.prompt="{prompt}" ' +
                f'data.image.rotate={rotation} ' +
                f'system.loggers.wandb.enable=False ' +
                f'system.loggers.wandb.name=stage1_{name} ' +
                f'exp_root_dir=outputs-stage1'
                )
        print(line)
        cmd = f'export CUDA_VISIBLE_DEVICES={gpu}; MKL_THREADING_LAYER=GNU; /cortex/users/ohadr/envs/envs/animate/bin/python launch.py {line}'
        os.system(cmd)

def main_batch(index, gpu):
    file_path = 'custom/scripts/objects.yaml' #'objects.yaml'
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)

    num_gpus = 1  # Change this to the number of available GPUs
    objects = data['stage1-ablation']

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