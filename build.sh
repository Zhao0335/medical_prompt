#!/bin/bash
#SBATCH --job-name=medical_prompt
#SBATCH --partition=gre
#SBATCH --nodes=1                 # 单节点运行
#SBATCH --ntasks=1                # 1个主任务
#SBATCH --cpus-per-task=16        # 申请 16 个 CPU 核心处理数据
#SBATCH --gres=gpu:8              # 【关键】申请 8 块 GPU！
#SBATCH --mem=168G                # 【关键】申请 120GB 系统内存，防止加载权重时OOM
#SBATCH --time=02:00:00           # 预计运行时间 2 小时
#SBATCH --output=run_32b_%j.log   # 正常输出日志
#SBATCH --error=run_32b_%j.err    # 报错日志


module load cuda/12.4
module load conda

# Source conda activation script directly
source /persist_data/apps/miniconda3/etc/profile.d/conda.sh
conda activate /persist_data/home/chenxuzhao/.conda/envs~/medical_prompt

export OMP_NUM_THREADS=16


echo "Job Start: $(date)"
python /persist_data/home/chenxuzhao/medical_prompt/main.py
echo "Job End: $(date)"
