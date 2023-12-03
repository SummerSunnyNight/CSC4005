#!/bin/bash
#SBATCH -o ./Project4-Results.txt
#SBATCH -p Project
#SBATCH -J Project4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --gres=gpu:1

# Get the current directory
CURRENT_DIR=$(pwd)
echo "Current directory: ${CURRENT_DIR}"

TRAIN_X=./dataset/training/train-images.idx3-ubyte
TRAIN_Y=./dataset/training/train-labels.idx1-ubyte
TEST_X=./dataset/testing/t10k-images.idx3-ubyte
TEST_Y=./dataset/testing/t10k-labels.idx1-ubyte

# Softmax
# echo "Softmax Sequential"
# # srun -n 1 --cpus-per-task 1 valgrind --leak-check=full --track-origins=yes ${CURRENT_DIR}/build/softmax $TRAIN_X $TRAIN_Y $TEST_X $TEST_Y
# srun -n 1 --cpus-per-task 1 ${CURRENT_DIR}/build/softmax $TRAIN_X $TRAIN_Y $TEST_X $TEST_Y
# #可以以后在自己的主机上学，没必要在集群上学
# echo ""

file="openacc_Result-new.qdrep"

# 检查文件是否存在
if [ -e "$file" ]; then
    # 如果文件存在，则删除
    rm "$file"
    echo "文件 $file 已删除"
else
    echo "文件 $file 不存在"
fi

file="openacc_Result-new.sqlite"

# 检查文件是否存在
if [ -e "$file" ]; then
    # 如果文件存在，则删除
    rm "$file"
    echo "文件 $file 已删除"
else
    echo "文件 $file 不存在"
fi


echo "Softmax OpenACC"

#  srun -n 1 --gpus 1 nsys profile -t cuda,nvtx,osrt,openacc -o ./openacc_Result-new.qdrep ${CURRENT_DIR}/build/softmax_openacc $TRAIN_X $TRAIN_Y $TEST_X $TEST_Y

srun -n 1 --gpus 1 ${CURRENT_DIR}/build/softmax_openacc $TRAIN_X $TRAIN_Y $TEST_X $TEST_Y
echo ""

# # NN
# echo "NN Sequential"
# srun -n 1 --cpus-per-task 1 ${CURRENT_DIR}/build/nn $TRAIN_X $TRAIN_Y $TEST_X $TEST_Y
# echo ""

# echo "NN OpenACC"
# srun -n 1 --gpus 1 ${CURRENT_DIR}/build/nn_openacc $TRAIN_X $TRAIN_Y $TEST_X $TEST_Y
# echo ""