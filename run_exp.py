import json
import subprocess
import shlex

prompt="a head potrait of the man, black hair , white shirt"
negative_prompt="(face painting, artifact, flag, decorative design,deformed iris, dark, white hair, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime:1.4), worst quality, low quality, jpeg artifacts, bad anatomy"

image_path = f"preprocess_data/liudehua"
tag = 'liudehua'
prompt = "a DSLR portrait of " + prompt + "black hair"

# subprocess.run('export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python')

subprocess.run(f'python launch.py --config configs/liudehua_geo.yaml \
                                    --train system.prompt_processor.prompt="{prompt}, black background, normal map" \
                                            data.image_path={image_path} \
                                            system.geometry.shape_init="mesh:{image_path}/init_mesh.ply" \
                                            system.guidance.pil_image_path="{image_path}/img.png" \
                                            tag={tag}', shell=True)

cmd = f'python launch.py --config configs/liudehua_tex.yaml \
                        --train system.prompt_processor.prompt={shlex.quote(prompt)} \
                                system.prompt_processor.negative_prompt={shlex.quote(negative_prompt)} \
                                data.image_path={image_path} \
                                system.geometry_convert_from="outputs/IDsculpt-geometry/{tag}@LAST/ckpts/last.ckpt" \
                                system.guidance.pil_image_path="{image_path}/img.png" \
                                tag={tag}'


subprocess.run(shlex.split(cmd))