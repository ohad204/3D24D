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
/cortex/users/ohadr/envs/envs/animate/bin/python launch.py --config custom/configs/animate124-stage2-ms.yaml --train --gpu $gpu \
data.image.image_path=custom/load/dinosaur/model.obj \
system.prompt_processor.prompt="walking dinosaur" \
system.weights=/home/lab/rahamiob/ohadr/projects/object_animation/outputs/animate124-stage1/a@20240531-004832/ckpts/last.ckpt \
system.loggers.wandb.enable=true \
system.loggers.wandb.name="${experiment_name}_dinosaur"

/cortex/users/ohadr/envs/envs/animate/bin/python launch.py --config custom/configs/animate124-stage2-ms.yaml --train --gpu $gpu \
data.image.image_path=custom/load/eagle/model.obj \
system.prompt_processor.prompt="flying eagle" \
system.weights=/home/lab/rahamiob/ohadr/projects/object_animation/outputs/animate124-stage1/a@20240531-005419/ckpts/last.ckpt \
system.loggers.wandb.enable=true \
system.loggers.wandb.name="${experiment_name}_eagle"

/cortex/users/ohadr/envs/envs/animate/bin/python launch.py --config custom/configs/animate124-stage2-ms.yaml --train --gpu $gpu \
data.image.image_path=custom/load/shark/model.obj \
system.prompt_processor.prompt="swimming shark" \
system.weights=/home/lab/rahamiob/ohadr/projects/object_animation/outputs/animate124-stage1/a@20240531-010730/ckpts/last.ckpt \
system.loggers.wandb.enable=true \
system.loggers.wandb.name="${experiment_name}_shark"
