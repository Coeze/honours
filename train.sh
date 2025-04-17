#!/bin/bash
#SBATCH --job-name=etcaps_train        # Name of the job
#SBATCH --output=logs/etcaps_%j.out    # Standard output file (%j is job ID)
#SBATCH --error=logs/etcaps_%j.err     # Standard error file
#SBATCH --ntasks=1                     # Run a single task
#SBATCH --cpus-per-task=4              # Number of CPU cores per task 
#SBATCH --gres=gpu:4                   # Request 4 GPUs (adjust as needed)
#SBATCH --mem=32G                      # Memory per node
#SBATCH --time=24:00:00                # Time limit (24 hours)
#SBATCH --partition=gpu                # GPU partition/queue

# Create log directory if it doesn't exist
mkdir -p logs

# Print some information about the job
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_JOB_NODELIST"
echo "Number of GPUs: $(nvidia-smi --list-gpus | wc -l)"
echo "Start time: $(date)"

# Activate virtual environment (adjust for your environment)
source ./venv/bin/activate

# Run the Python script with distributed flag to use all available GPUs
python main.py \
    --data_dir ./data \
    --dataset svhn \
    --epochs 100 \
    --batch_size 128 \
    --distributed \
    --et

echo "End time: $(date)"
