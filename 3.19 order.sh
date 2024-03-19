#!/bin/bash

rounds=5 epochs=300
batch_size=1
learning_rate=0.0004 decay=0.0005
program_test=1 experiment=0 record=0
dimensions="32"
datasets="gpu"
densities="0.05"
orders="1 2 3 4 5"
py_files="run_experiment"
for py_file in $py_files
do
    for dim in $dimensions
    do
        for dataset in $datasets
        do
            for density in $densities
            do
                for order in $orders
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
                          --order $order \
                          --experiment $experiment \
                          --record $record
                    echo ""
                done
            done
        done
    done
done
