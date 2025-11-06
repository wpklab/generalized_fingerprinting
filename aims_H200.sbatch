#!/bin/bash
#SBATCH --time=40:00:00                  # Job run time (hh:mm:ss)
#SBATCH --account=wpk                   # Replace "account_name" with an account available to you
#SBATCH --nodes=3                       # Number of nodes - one pc
#SBATCH --ntasks-per-node=8            # Number of task (cores/ppn) per node - AMD 7763
#SBATCH --cpus-per-task=16              # Number of CPU cores per task
#SBATCH --job-name=aims # Name of batch job
#SBATCH --partition=wpk                  # Partition (queue)
#SBATCH --output=aims_test.o%j        # Name of batch job output file
#SBATCH --error=aims_test.e%j
#SBATCH --gres=gpu:H200:8
#SBATCH --mem=1000GB
#SBATCH --array=1-500

nvidia-smi

nvidia-smi nvlink -s
nvidia-smi topo -m

nvcc --version

ifconfig

ip addr

ibv_devinfo


export NCCL_IB_DISABLE=0
export NCCL_P2P_LEVEL=NVL  # Prefer NVLink
export NCCL_NET_GDR_LEVEL=5
export NCCL_TOPO_DUMP_FILE=nccl_topo.xml


export MASTER_PORT=$(python -c "import socket; s = socket.socket(socket.AF_INET, socket.SOCK_STREAM); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
export WORLD_SIZE=$((SLURM_NNODES * SLURM_NTASKS_PER_NODE))
echo "WORLD_SIZE="$WORLD_SIZE

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

dcgmi profile --trace gpuinfo,nvlink -f nvlink_usage.csv &

srun python AIMS_fingerprint_original_model.py

if [ $? -ne 0 ]; then
    echo "srun command failed, exiting job."
    scancel $SLURM_JOB_ID
    exit 1 
fi