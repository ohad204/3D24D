#!/bin/bash

# Default GPU number
gpu=5

# Parse command-line options
while getopts ":g:" opt; do
  case ${opt} in
    g )
      gpu=$OPTARG
      ;;
    \? )
      echo "Usage: $(basename $0) [-g GPU_NUMBER]" >&2
      exit 1
      ;;
    : )
      echo "Invalid option: $OPTARG requires an argument" >&2
      exit 1
      ;;
  esac
done
shift $((OPTIND -1))

# --------- Stage 1 (Static Stage) --------- #
/cortex/users/ohadr/envs/envs/animate/bin/python launch.py --config custom/configs/animate124-stage1.yaml --train --gpu $gpu \
data.image.image_path=custom/load/dinosaur/model.obj \
 system.prompt_processor.prompt="dinosaur" \
 system.loggers.wandb.enable=true \
 system.loggers.wandb.name=first_attempt

/cortex/users/ohadr/envs/envs/animate/bin/python launch.py --config custom/configs/animate124-stage1.yaml --train --gpu $gpu \
data.image.image_path=custom/load/eagle/model.obj \
 system.prompt_processor.prompt="eagle" \
 system.loggers.wandb.enable=true \
 system.loggers.wandb.name=first_attempt

/cortex/users/ohadr/envs/envs/animate/bin/python launch.py --config custom/configs/animate124-stage1.yaml --train --gpu $gpu \
data.image.image_path=custom/load/shark/model.obj \
 system.prompt_processor.prompt="shark" \
 system.loggers.wandb.enable=true \
 system.loggers.wandb.name=first_attempt

/cortex/users/ohadr/envs/envs/animate/bin/python launch.py --config custom/configs/animate124-stage1.yaml --train --gpu $gpu \
data.image.image_path=custom/load/panda_po/Po.obj \
system.prompt_processor.prompt="panda_po" \
system.loggers.wandb.enable=true \
system.loggers.wandb.name=first_attempt