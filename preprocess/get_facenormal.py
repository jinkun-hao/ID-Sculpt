import cv2
from insightface.app import FaceAnalysis
import torch
import torch
from diffusers import StableDiffusionControlNetPipeline, DDIMScheduler, AutoencoderKL, ControlNetModel
from diffusers import StableDiffusionPipeline, DDIMScheduler, AutoencoderKL
from diffusers.pipelines.controlnet.multicontrolnet import MultiControlNetModel
from PIL import Image, ImageChops
from diffusers.utils import load_image
import numpy as np
import os
import argparse
import sys
import json
sys.path.append('../IP_Adapter')
# from ip_adapter.ip_adapter_faceid import IPAdapterFaceID
from ip_adapter.ip_adapter_faceid_separate import IPAdapterFaceID
from diffusers import UniPCMultistepScheduler


app = FaceAnalysis(name="buffalo_l", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],
                    root='../sd_model/insightface')
app.prepare(ctx_id=0, det_size=(224, 224))

base_model_path = "../../../sd_model/models--xanderhuang--normal-adapted-sd1.5/snapshots/a44f486d748a4bd0b46fc90113b5612afc58e0c4"
vae_model_path = "../../../sd_model/models--stabilityai--sd-vae-ft-mse/snapshots/31f26fdeee1355a5c34592e401dd41e45d25a493"
ip_ckpt = "../IP_Adapter/models/ip-adapter-faceid-portrait-v11_sd15.bin"
device = "cuda"

noise_scheduler = DDIMScheduler.from_pretrained(
    '../../../sd_model/models--runwayml--stable-diffusion-v1-5/snapshots/1d0c4ebf6ff58a5caecab40fa1406526bca4b5b9',
    subfolder="scheduler",
    torch_dtype=torch.float16,
)

vae = AutoencoderKL.from_pretrained(vae_model_path).to(dtype=torch.float16)

#
controlnet_model_path_canny = "../../../sd_model/models--lllyasviel--control_v11p_sd15_canny/snapshots/115a470d547982438f70198e353a921996e2e819"
controlnet_canny = ControlNetModel.from_pretrained(controlnet_model_path_canny, torch_dtype=torch.float16)

pipe = StableDiffusionControlNetPipeline.from_pretrained(base_model_path, controlnet=controlnet_canny,
                                                         # scheduler=noise_scheduler,
                                                         # vae=vae,
                                                         feature_extractor=None,
                                                         torch_dtype=torch.float16, safety_checker=None)

ip_model = IPAdapterFaceID(pipe, ip_ckpt, device, num_tokens=16)

def main(id, caption, root_dir):
    img_path = os.path.join(root_dir, id, 'img_rgba.png')
    image = cv2.imread(img_path)
    faces = app.get(image)
    if len(faces) == 0:
        print(f"No face detected in {id}")
        return

    faceid_embeds = torch.from_numpy(faces[0].normed_embedding).unsqueeze(0)
    n_cond = faceid_embeds.shape[1]

    prompt = caption

    negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality, blurry"
    # laion_img = load_image("output_laion_norm.png")

    image = Image.open(img_path)
    image = np.array(image)
    low_threshold = 100
    high_threshold = 200

    image = cv2.Canny(image, low_threshold, high_threshold)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    canny_image = Image.fromarray(image)
    canny_image.save(os.path.join(root_dir, id, 'img_canny.png'))

    # control_image = ImageChops.add(laion_img.resize(canny_image.size), canny_image)

    images = ip_model.generate(
        prompt=prompt + 'black background, normal map, head only', negative_prompt=None, faceid_embeds=faceid_embeds,
        #   image=[laion_img.resize(canny_image.size), canny_image],
        image=canny_image, scale=0.5,
        num_samples=1, width=512, height=512, num_inference_steps=30, seed=2024
    )

    def image_grid(imgs, rows, cols):
        assert len(imgs) == rows*cols

        w, h = imgs[0].size
        grid = Image.new('RGB', size=(cols*w, rows*h))
        grid_w, grid_h = grid.size
        
        for i, img in enumerate(imgs):
            grid.paste(img, box=(i%cols*w, i//cols*h))
        return grid


    grid = image_grid(images, 1, 1)
    grid.save(os.path.join(root_dir, id, 'img_normal.png'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Get normal and depth')
    parser.add_argument('-root_dir', type=str, default="/home/haojinkun/3DAvatar/FFFHQ_sub/dataset_single/sub1")
    args = parser.parse_args()

    id_list = os.listdir(args.root_dir)

    caption_file = {
        'man': "a man with short hair",
        'hinton2': "an old man with short hair, shirt", 
        'liuyifei': "a woman with Long, tangled hair", 
        'taylor': "a woman with Long, loose hair", 
        'woman': "a woman with Long, loose hair", 
        'Trump': "an old man with short hair", 
        'liudehua': "a man", 
        'changzeyamei': "a woman with Long, loose hair", 
        'img_new': "a man, shirt", 
        'hinton1': "an old man with short hair",
    }
    for id in id_list:
        # if id != '00810':
        #     continue
        # id_filename = id
        caption = caption_file[id]
        if os.path.exists(f"{args.root_dir}/{id}/img_rgba.png"):
            main(id, caption, args.root_dir)

