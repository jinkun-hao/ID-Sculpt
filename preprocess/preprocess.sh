cd 3DDFA_V2-master

python dlib_kps.py -input_dir ../dataset/head_img_new -output_path data.pkl

python recrop_images.py -i data.pkl -j dataset.json --out_dir ../dataset/head_img_new_crop

cd ../PanoHead/face_parsing

python evaluate.py   -input_dir ../../dataset/head_img_new_crop -output_dir ../../../preprocess_data

cd ../../
python get_facenormal.py -root_dir ../preprocess_data
python get_facedepth.py -root_dir ../preprocess_data

cd PanoHead
bash gen_pti_script.sh	# 修改sh文件中的out（就是eg3d模型的保存路径），还有--target_img_new（crop后并且做了背景去除的图片路径）

bash gen_mesh.sh
bash gen_ldmk.sh
