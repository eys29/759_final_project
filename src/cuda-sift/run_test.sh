#!/usr/bin/env bash
#SBATCH --job-name=task1
#SBATCH -p instruction
#SBATCH --ntasks=1 --cpus-per-task=4
#SBATCH --time=0-00:10:00
#SBATCH --output="run.out"
#SBATCH --error="run.err"
#SBATCH --gres=gpu:1

# Remove logs from previous runs
rm -f small.txt
rm -f medium.txt
rm -f large.txt

# Run test for small scale images
for i in {1..5}; do
    ./cudasift ../../dataset/small/img1.png ../../dataset/small/img2.png >> small.txt 2>&1
done

# Run test for medium scale images
for i in {1..5}; do
    ./cudasift ../../dataset/medium/img1.jpg ../../dataset/medium/img2.jpg >> medium.txt 2>&1
done

# Run test for large scale images
for i in {1..5}; do
    ./cudasift ../../dataset/large/img1.jpg ../../dataset/large/img2.jpg >> large.txt 2>&1
done

rm -f Matches.png