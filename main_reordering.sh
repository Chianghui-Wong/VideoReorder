#! /bin/zsh

timestamp=`date "+%m-%d-%H%M%S"`
output_dir="outputs/${timestamp}/"
mkdir -p ${output_dir}

OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 main_reordering.py --outputs_dir ${output_dir} --config './configs/frame_reorder.yaml' > ${output_dir}nohup