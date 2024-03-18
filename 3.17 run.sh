#!/bin/bash

# 定义变量
rounds=5 epochs=300
batch_size=1
learning_rate=0.0004 decay=0.0005
program_test=1
experiment=1 record=1
dimensions="32"
datasets="cpu gpu"
densities="0.01 0.02 0.03 0.04 0.05"
#py_files="run_mlp_llm run_embed run_embed_LLM run_gcn_llm run_embed_GCN"
py_files="run_embed_GCN"
for py_file in $py_files
do
    for dim in $dimensions
    do
        for dataset in $datasets
        do
            for density in $densities
            do
                python ./$py_file.py \
                      --rounds $rounds \
                      --density $density \
                      --dataset $dataset \
                      --epochs $epochs \
                      --bs $batch_size \
                      --lr $learning_rate \
                      --decay $decay \
                      --program_test $program_test \
                      --dimension $dim \
                      --experiment $experiment \
                      --record $record
            done
        done
        # 在命令序列之间添加一个空行
        echo ""
    done
done