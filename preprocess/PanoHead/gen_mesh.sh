#!/bin/bash

base_dir="../dataset/head_img_new_inversion"    # gan inversion dataset

for i in $(find $base_dir -mindepth 1 -maxdepth 1 -type d -exec basename {} \;); do
   echo $i
   python pre_gen_multiview.py --outdir=preprocess_data --trunc=0.7 --shapes=True \
           --network "$base_dir/${i}/fintuned_generator.pkl" \
           --latent "$base_dir/${i}/projected_w.npz" \
           --shape-format '.ply' \
           --outdir "../../preprocess_data/${i}"  # preprocess data
done

