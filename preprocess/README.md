# Preprocessing Pipeline
## Environment
Environment setup refers to configurations in [Panohead](https://github.com/SizheAn/PanoHead/tree/main) and [3DDFA_V2](https://github.com/cleardusk/3DDFA_V2)

## One-click Processing Script
```
bash preprocess.sh
```


## Processing Pipeline
1. Face alignment (working directory is under 3DDFA_V2-master folder, images in input folder should be .png)
```
cd 3DDFA_V2-master
python dlib_kps.py -input_dir $DIR -output_path $PATH
# DIR = test/head_img_new, PATH = data.pkl
```

2. Calculate 3DMM and camera parameters (dataset.json: image names, camera parameters; face_data.pkl: coordinates of seven facial keypoints)
```
cd 3DDFA_V2-master
python recrop_images.py -i data.pkl -j dataset.json --out_dir '../PanoHead-main/dataset/head_img_new'
```


3. get rgba image
```
cd Panohead/face_parsing
python evaluate.py  -input_dir $DIR -output_path $DIR
```

4. run diffusion inference to get depth and normal
```
cd preprocess
python get_facedepth.py
python get_facenormal.py
```

dataset：/home/haojinkun/3DAvatar/PanoHead-main/dataset/head_img_new

5. run panohead inversion, and get triplane weight
```
cd Panohead
bash gen_pti_script.sh
```


6. Export multi-view normal maps and generate mesh .ply file
```
bash gen_mesh.sh
```

7. Align landmark
```
python pre_align_3Dlm.py
```
