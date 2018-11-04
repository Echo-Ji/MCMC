#!/bin/bash
export OMP_NUM_THREADS=$1
#python3 smc.py | tee ./output/$OMP_NUM_THREADS-`date +%Y-%m-%d-%H%M%S.smc.out`
python smc-m.py | tee ./smc3-output/$OMP_NUM_THREADS-`date +%Y-%m-%d-%H%M%S.smc.out`
