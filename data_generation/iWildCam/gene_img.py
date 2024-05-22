import os
import sys

sys.path.append(os.getcwd())

from model import StableDiffusionImg2ImgPipeline, StableDiffusionXLImg2ImgPipeline
import torch
from PIL import Image
import pickle
from tqdm import tqdm
import numpy as np
import pandas as pd
import random
import argparse

device = "cuda" if torch.cuda.is_available() else "cpu"
model_id_or_path = "stabilityai/stable-diffusion-xl-refiner-1.0"


def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)  # print(f"Random seed set as {seed}")


def create_folder_generate(img, prompt: str, strength: float, guid_scale: float, out_folder: str, out_name: str,
                           seed=42):

    out_folder_n = os.path.join(out_folder, f'Strength{int(strength * 100)}_guidance{int(guid_scale * 10)}_seed{seed}')
    os.makedirs(out_folder_n, exist_ok=True)

    generate_img(img=img, prompt=prompt, strength=strength, guid_scale=guid_scale, out_folder=out_folder_n,
                 out_name=out_name, seed=seed)
    return


def generate_img(img, prompt: str, strength: float, guid_scale: float, out_folder: str, out_name: str, seed=42):
    """Generate denoised image based on given prompt

    :param out_name:
    :param out_folder:
    :param seed:
    :param _type_ img: _description_
    :param str prompt: _description_
    :param float strength: _description_
    :param float guid_scale: _description_
    """
    combined_seed = hash(prompt) ^ seed
    set_seed(seed=combined_seed)
    results = pipe(prompt=prompt, image=img, strength=strength, guidance_scale=guid_scale)
    images = results.images
    output_f = f"{out_folder}/{out_name}.jpg"
    while not os.path.exists(output_f):
        images[0].save(output_f)
    print(f"save to {output_f}")

    return


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_csv", type=str, default='', help="Path including original image path / species names")
    parser.add_argument("--output_path", type=str, default='data/train_new', help="Output path for the synthetic data")
    parser.add_argument("--part", type=int, default=1, help="For distributed generating.")
    parser.add_argument("--total_parts", type=int, default=4, help="For distributed generating.")
    args = parser.parse_args()

    # generation settings
    noises = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    text_guidance = [10]
    random_seed = [10, 20, 30, 40, 42, 50, 60, 70]

    # creating folders
    os.makedirs(args.output_path, exist_ok=True)

    # Dict of common name vs species name
    with open(f'assets/metadata/wilds_common_names.pkl', 'rb') as f:
        dict_name = pickle.load(f)

    # loading stable diffusion pipeline
    pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(model_id_or_path, torch_dtype=torch.float16, variant="fp16",
                                                            use_safetensors=True).to(device)

    df_generate = pd.read_csv(args.data_csv)
    if 'Unnamed: 0' in df_generate.columns:
        del df_generate['Unnamed: 0']
    df_generate.drop_duplicates(inplace=True)

    dict_byspname = dict()
    for row_id, row in df_generate.iterrows():
        sp_name = row[1]
        img_name = row[-1]
        filepath = row[2]
        if sp_name not in dict_byspname:
            dict_byspname[sp_name] = []
        dict_byspname[sp_name].append([img_name, filepath])

    ########################################
    # generate from beginning
    list_all = list(dict_byspname.keys())
    list_all = sorted(list_all, reverse=False)
    list_index = list(split(range(len(list_all)), args.total_parts))
    list_cur_parts = list_index[args.part - 1]
    list_folder = [list_all[i] for i in list_cur_parts]
    print(list_folder)

    for sp_name in tqdm(list_folder):
        if sp_name == 'empty':
            continue
        sp_name_underscore = sp_name.replace(' ', '_')
        if sp_name_underscore not in dict_name:
            continue

        cmm_name = dict_name[sp_name_underscore]
        print(f'generate image for {sp_name}: {cmm_name}')

        output_folder = f'{args.output_path}/{sp_name_underscore}/'
        os.makedirs(output_folder, exist_ok=True)
        all_generated_img = set()
        guid_folders = os.listdir(output_folder)
        dict_img_seed = dict()
        for sub_folder in guid_folders:
            cur_seed = int(sub_folder.split('seed')[-1])
            tmp_path = os.path.join(output_folder, sub_folder)
            list_img = os.listdir(tmp_path)
            list_img = [item.replace('.jpg', '') for item in list_img if 'jpg' in item]
            for img_name in list_img:
                if img_name not in dict_img_seed:
                    dict_img_seed[img_name] = []
                if cur_seed not in dict_img_seed[img_name]:
                    dict_img_seed[img_name].append(cur_seed)

        for img_name, list_seed in dict_img_seed.items():
            if len(list_seed) >= 2:
                all_generated_img.add(img_name)

        del dict_img_seed

        img_list = dict_byspname[sp_name]
        for img_pair in img_list:
            img_name = img_pair[0]
            iwildcamp_img_path = img_pair[1]
            iwildcamp_img_path = iwildcamp_img_path.replace('../', '')
            # check if the image is generated 
            if img_name in all_generated_img:
                # generated, skip
                continue

            # directly use original img to generate new images
            init_image = Image.open(iwildcamp_img_path).convert("RGB")
            init_image = init_image.resize((480, 270))

            cur_prompt = f"a photo of {cmm_name} in the wild"
            for ranseed in random_seed:
                for noise_strength in noises:
                    create_folder_generate(img=init_image, prompt=cur_prompt, strength=noise_strength,
                                           guid_scale=text_guidance[0], out_folder=output_folder,
                                           out_name=img_name.replace('.jpg', ''), seed=ranseed)
