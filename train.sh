#!/bin/bash

# for ((i=1; i<=5; i++))

# do
#     echo "train $i, seed = `expr 45 + ${i}`"
#     python -u main.py -act t e -seed `expr 45 + ${i}` -task TPpred_tw_60_${i} -pth trained_models/TPpred_tw_60_${i}.pth >logs/TPpred_tw_60_${i}.txt

# done



# !/bin/bash
# 
echo "training"
python -u main.py -act t e -seed 49 \
-task TPpred_tw_60_1 \
-pth trained_models/TPpred_tw_60_1.pth \
>logs/TPpred_tw_60_1.txt
