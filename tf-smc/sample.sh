#!/bin/bash
#export OMP_NUM_THREADS=$1
#python3 smc.py | tee ./output/$OMP_NUM_THREADS-`date +%Y-%m-%d-%H%M%S.smc.out`
source activate edward_py2
python bayesian_nn.py | tee ./edout/`date +%Y-%m-%d-%H%M%S.ed`
source deactivate
