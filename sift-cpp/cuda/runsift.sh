#!/usr/bin/env bash
#SBATCH --job-name=task1
#SBATCH -p instruction
#SBATCH --ntasks=1 --cpus-per-task=2
#SBATCH --time=0-00:10:00
#SBATCH --output="run.out"
#SBATCH --error="run.err"
#SBATCH --gres=gpu:1

./cudasift ../imgs/book_rotated.jpg ../imgs/book_in_scene.jpg  
