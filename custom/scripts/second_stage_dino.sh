#!/bin/bash

# Default GPU number
gpu=5
experiment_name="default_experiment"

# Parse command-line options
while getopts ":g:e:" opt; do
  case ${opt} in
    g )
      gpu=$OPTARG
      ;;
    e )
      experiment_name=$OPTARG
      ;;
    \? )
      echo "Usage: $(basename $0) [-g GPU_NUMBER] [-e EXPERIMENT_NAME]" >&2
      exit 1
      ;;
    : )
      echo "Invalid option: $OPTARG requires an argument" >&2
      exit 1
      ;;
  esac
done
shift $((OPTIND -1))

# --------- Stage 2 (Coarse dynamic Stage) --------- #
export CUDA_VISIBLE_DEVICES=2; /cortex/users/ohadr/envs/envs/animate/bin/python launch.py --config custom/configs/animate124-stage2-ms.yaml --train --gpu 2 \
data.image.image_path=custom/load/dinosaur/model.obj \
system.prompt_processor.prompt="dinosaur wagging its tail" \
system.weights=/home/lab/rahamiob/ohadr/projects/object_animation/outputs/animate124-stage1/a@20240531-004832/ckpts/last.ckpt \
system.loggers.wandb.enable=true \
system.loggers.wandb.name="${experiment_name}_dinosaur_wagging" &

export CUDA_VISIBLE_DEVICES=3; /cortex/users/ohadr/envs/envs/animate/bin/python launch.py --config custom/configs/animate124-stage2-ms.yaml --train --gpu 3 \
data.image.image_path=custom/load/dinosaur/model.obj \
system.prompt_processor.prompt="roaring dinosaur" \
system.weights=/home/lab/rahamiob/ohadr/projects/object_animation/outputs/animate124-stage1/a@20240531-004832/ckpts/last.ckpt \
system.loggers.wandb.enable=true \
system.loggers.wandb.name="${experiment_name}_dinosaur_roaring" &

export CUDA_VISIBLE_DEVICES=5; /cortex/users/ohadr/envs/envs/animate/bin/python launch.py --config custom/configs/animate124-stage2-ms.yaml --train --gpu 5 \
data.image.image_path=custom/load/dinosaur/model.obj \
system.prompt_processor.prompt="head shaking dinosaur" \
system.weights=/home/lab/rahamiob/ohadr/projects/object_animation/outputs/animate124-stage1/a@20240531-004832/ckpts/last.ckpt \
system.loggers.wandb.enable=true \
system.loggers.wandb.name="${experiment_name}_dinosaur_shaking" &

export CUDA_VISIBLE_DEVICES=6; /cortex/users/ohadr/envs/envs/animate/bin/python launch.py --config custom/configs/animate124-stage2-ms.yaml --train --gpu 6 \
data.image.image_path=custom/load/dinosaur/model.obj \
system.prompt_processor.prompt="bending dinosaur" \
system.weights=/home/lab/rahamiob/ohadr/projects/object_animation/outputs/animate124-stage1/a@20240531-004832/ckpts/last.ckpt \
system.loggers.wandb.enable=true \
system.loggers.wandb.name="${experiment_name}_dinosaur_bending" &

# Wait for all background jobs to complete
wait
