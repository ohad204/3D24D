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
    for object in objects:
        object_dir = object[0]
        object_paths = glob.glob(os.path.join(object_dir, "**", "*.glb"), recursive=True)
        name = object[3]
        for object_path in object_paths:
            file_parts = object_path.split('/')
            file_name = f'{file_parts[-4]}_{file_parts[-2]}_{file_parts[-1][:-4]}'
            line = ('--config custom/configs/animate124-stage1-objaverse.yaml --train --gpu 1 ' +
                    f'data.image.image_path={object_path} system.prompt_processor.prompt="{file_name}" ' +
                    f'system.loggers.wandb.enable=True ' +
                    f'system.loggers.wandb.project=object-objaverse ' +
                    f'system.loggers.wandb.name=stage1_{name} ' +
                    f'exp_root_dir=outputs-objaverse-15 ' +
                    f'data.image.rotate=0'
                    )
            print(line)
            cmd = f'export CUDA_VISIBLE_DEVICES={gpu}; MKL_THREADING_LAYER=GNU; /cortex/users/ohadr/envs/envs/animate_h100/bin/python launch.py {line}'
            os.system(cmd)

def main_batch(index, gpu):
    file_path = 'custom/scripts/objects.yaml' # 'custom/scripts/objects.yaml'
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)

    num_gpus = 1  # Change this to the number of available GPUs
    objects = data['stage2-objaverse-snowman']

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