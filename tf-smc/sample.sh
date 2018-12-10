#!/bin/bash
#export OMP_NUM_THREADS=$1
#python3 smc.py | tee ./output/$OMP_NUM_THREADS-`date +%Y-%m-%d-%H%M%S.smc.out`
source activate edward_py2
CUDA_VISIBLE_DEVICES=0 python reg_bayesian_nn_7_10.py | tee ./edout/aq-`date +%Y-%m-%d-%H%M%S.ed`
source deactivate
