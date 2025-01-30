#!/bin/bash

# 指定文件夹路径
folder="../3DDFA_V2-master/test/head_img_new/"

# 遍历文件夹下所有PNG文件
for file in "$folder"*.png; do
  # 获取文件名
  filename=$(basename "$file")
  # 执行命令
  python pre_align_3Dlm.py -source_dir ../dataset/head_img_new -mesh_dir ../../preprocess_data -img_name "$filename"
done