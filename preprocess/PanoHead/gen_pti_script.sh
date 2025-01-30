#!/usr/bin/env bash

#models=("easy-khair-180-gpc0.8-trans10-025000.pkl"\
#  "ablation-trigridD-1-025000.pkl")
model=("easy-khair-180-gpc0.8-trans10-025000.pkl")

in="models"
out="../dataset/head_img_new_inversion"

for i in $(seq 0 10)

do 
    # perform the pti and save w
    python projector_withseg.py --outdir=${out} --target_img=../dataset/head_img_new_crop --network ${in}/${model} --idx ${i}
done
