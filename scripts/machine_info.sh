#!/bin/bash
set -eu
echo "nproc: $(nproc)"
cat /proc/cpuinfo | grep 'model name' | uniq
lscpu | grep -e MHz -e "per socket" -e "per core" -e "Socket"
nvidia-smi --list-gpus
