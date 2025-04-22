#!/bin/bash

# For running on University of Bath's GPU cluster - Hex, using hare.

SCRIPT_PATH=""
GPU_DEVICE="device=6" # default to GPU 6
DETACHED_MODE=false

while [[ "$#" -gt 0 ]]; do
  case $1 in
    --script) SCRIPT_PATH="$2"; shift ;;
    --gpus) GPU_DEVICE="$2"; shift ;;
    -d) DETACHED_MODE=true ;;
    *) echo "Unknown parameter passed: $1"; exit 1 ;;
  esac
  shift
done

if [ -z "$SCRIPT_PATH" ]; then
    echo "Error: No script specified. Use --script to specify the Python script."
    exit 1
fi

if [ ! -z "$2" ]; then
  GPU_DEVICE="$2"
fi

if [ "$DETACHED_MODE" = true ]; then
  HARE_RUN_FLAGS="-dit"
else
  HARE_RUN_FLAGS="-it"
fi

TOKEN=$(cat $(pwd)/.token)

hare reserve /mnt/faster0/as4387
hare run --rm $HARE_RUN_FLAGS --gpus $GPU_DEVICE -v $(pwd):/app -u $(id -u):$(id -g) \
        -e HF_HOME=/app/.cache \
        -e HF_TOKEN="$TOKEN" \
        -e MPLCONFIGDIR="/app/.config/matplotlib" \
        as4387/pt-plus-libs python3 $SCRIPT_PATH 

hare release /mnt/faster0/as4387