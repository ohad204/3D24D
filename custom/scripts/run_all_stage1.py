import os
import subprocess
from multiprocessing import Process
from numpy import random
random.seed(0)
import json

import argparse
import yaml


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
                'system.loggers.wandb.enable=True ' +
                f'system.loggers.wandb.name=stage1_{name}' +
                f'system.loggers.wandb.project=object-objaverse-publish ' +
                f'exp_root_dir=output-publish'
                )
        print(line)
        cmd = f'export CUDA_VISIBLE_DEVICES={gpu}; MKL_THREADING_LAYER=GNU; /cortex/users/ohadr/envs/envs/3to4d/bin/python launch.py {line}'
        os.system(cmd)

def main_batch(index, gpu):
    file_path = 'custom/scripts/objects.yaml' #'objects.yaml'
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)

    num_gpus = 1  # Change this to the number of available GPUs
    objects = data['stage2-objaverse-test']

    batch_size = len(objects) // num_gpus  # Calculate the size of each batch
    batches = [objects[i:i + batch_size] for i in range(0, len(objects), batch_size)]

    for index, gpu in zip(range(len(batches)), range(num_gpus)):
        run(objects=batches[index], gpu=5)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Your script description here")
    parser.add_argument('--index', type=int, help='Index argument')
    parser.add_argument('--gpu', type=int, help='GPU argument')
    args = parser.parse_args()

    if args.index is None or args.gpu is None:
        parser.error("You must provide both --index and --gpu arguments.")
    else:
        main_batch(index=args.index, gpu=args.gpu)