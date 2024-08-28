import os
import sys

sys.path.append(os.getcwd())
from data_generation.model import StableDiffusionXLImg2ImgPipeline

from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer
from sklearn.metrics.pairwise import cosine_similarity
import torch
from PIL import Image
from tqdm import tqdm
import numpy as np
import pandas as pd
import random
import argparse
import json

device = "cuda" if torch.cuda.is_available() else "cpu"
model_id_or_path = "stabilityai/stable-diffusion-xl-refiner-1.0"

# Define the model ID
clip_path = "openai/clip-vit-base-patch32"


def get_model_info(model_ID, device):
    # Save the model to device
    model = CLIPModel.from_pretrained(model_ID).to(device)
    # Get the processor
    processor = CLIPProcessor.from_pretrained(model_ID)
    # Get the tokenizer
    tokenizer = CLIPTokenizer.from_pretrained(model_ID)
    # Return model, processor & tokenizer
    return model, processor, tokenizer


# Get model, processor & tokenizer
clip_model, processor, tokenizer = get_model_info(clip_path, device)


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


def get_single_text_embedding(tokenizer, model, text):
    inputs = tokenizer(text, return_tensors="pt").to(device)
    text_embeddings = model.get_text_features(**inputs)
    # convert the embeddings to numpy array
    embedding_as_np = text_embeddings.cpu().detach().numpy()
    return embedding_as_np


def get_single_image_embedding(processor, model, my_image, do_rescale=True):
    image = processor(text=None, images=my_image, return_tensors="pt", do_rescale=do_rescale)["pixel_values"].to(device)
    embedding = model.get_image_features(image)
    # convert the embeddings to numpy array
    embedding_as_np = embedding.cpu().detach().numpy()
    return embedding_as_np


def create_folder_generate(img, cls_name: str, strength: float, guid_scale: float, out_folder: str, out_name: str,
                           seed=42, real_text_embs=None):

    cur_cate = f'Strength{int(strength * 100)}_guidance{int(guid_scale * 10)}_seed{seed}'
    out_folder_n = os.path.join(out_folder, cur_cate)
    os.makedirs(out_folder_n, exist_ok=True)

    text_sim, img_sim = generate_img(img=img, cls_name=cls_name, strength=strength, guid_scale=guid_scale,
                                     out_folder=out_folder_n, out_name=out_name, seed=seed,
                                     real_text_embs=real_text_embs)
    return cur_cate, text_sim, img_sim


def generate_img(img, cls_name: str, strength: float, guid_scale: float, out_folder: str, out_name: str, seed=42,
                 real_text_embs=None):
    """Generate denoised image based on given prompt

    :param out_folder:
    :param real_text_embs:
    :param seed:
    :param out_name:
    :param cls_name:
    :param _type_ img: _description_
    :param float strength: _description_
    :param float guid_scale: _description_
    """
    set_seed(seed=seed)
    prompt = generate_text_prompt(cls_name, dict_result[cls_name], seed=seed, ver='v2')
    results = pipe(prompt=prompt, image=img, strength=strength, guidance_scale=guid_scale)
    images = results.images
    synth_image = images[0]
    text_sim, img_sim = None, None
    if real_text_embs is not None:
        real_emb, text_emb = real_text_embs
        img_emb = get_single_image_embedding(processor, clip_model, synth_image)
        text_sim = cosine_similarity(img_emb, text_emb)[0][0]
        img_sim = cosine_similarity(img_emb, real_emb)[0][0]
        list_generated_img = [(synth_image, text_sim, img_sim)]
        while text_sim <= 0.3 and len(list_generated_img) <= 5:
            prompt = generate_text_prompt(cls_name, dict_result[cls_name], seed=seed ** len(list_generated_img),
                                          ver='v2')
            results = pipe(prompt=prompt, image=img, strength=strength, guidance_scale=guid_scale)
            images = results.images
            synth_image = images[0]
            img_emb = get_single_image_embedding(processor, clip_model, synth_image)
            img_sim = cosine_similarity(img_emb, real_emb)[0][0]
            text_sim = cosine_similarity(img_emb, text_emb)[0][0]
            list_generated_img.append((synth_image, text_sim, img_sim))

        # use the image with largest text similarity
        list_generated_img = sorted(list_generated_img, key=lambda x: x[1], reverse=True)
        synth_image = list_generated_img[0][0]
        text_sim = list_generated_img[0][1]
        img_sim = list_generated_img[0][2]

    output_f = f"{out_folder}/{out_name}"
    while not os.path.exists(output_f):
        synth_image.save(output_f)
    return text_sim, img_sim


def gene_imgs(output_folder):
    all_generated_img = set()
    guid_folders = os.listdir(output_folder)
    dict_img_seed = dict()
    for sub_folder in guid_folders:
        cur_seed = int(sub_folder.split('seed')[-1])
        tmp_path = os.path.join(output_folder, sub_folder)
        list_img = os.listdir(tmp_path)
        list_img = [item.replace('.JEPG', '') for item in list_img if 'JEPG' in item]
        for img_name in list_img:
            if img_name not in dict_img_seed:
                dict_img_seed[img_name] = []
            if cur_seed not in dict_img_seed[img_name]:
                dict_img_seed[img_name].append(cur_seed)

    for img_name, list_seed in dict_img_seed.items():
        if len(list_seed) >= 2:
            all_generated_img.add(img_name)

    del dict_img_seed

    return all_generated_img


def generate_text_prompt(class_name, dict_properties, seed, ver='v2'):
    combined_seed = hash(class_name) ^ seed
    random.seed(combined_seed)
    template = lambda prompt: (f"a 8k real photo: {prompt}")
    prompt = random.choice(dict_properties)
    text_prompt = template(prompt=prompt, )

    return text_prompt


def sd_generate(init_image, cls_name, random_seed, noises, text_guidance, output_folder, img_name, real_text_embs):
    list_res = []
    for ranseed in random_seed:
        for noise_strength in noises:
            for text_guid in text_guidance:
                cur_cate, text_sim, img_sim = create_folder_generate(img=init_image, cls_name=cls_name,
                                                                     strength=noise_strength, guid_scale=text_guid,
                                                                     out_folder=output_folder,
                                                                     out_name=img_name.replace('.JEPG', ''),
                                                                     seed=ranseed, real_text_embs=real_text_embs)
                list_res.append((cur_cate, img_name.replace('.JEPG', ''), text_sim, img_sim))
    return list_res


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_csv", type=str, default='', help="Path including original image path / species names")
    parser.add_argument("--prompt_json", type=str, default='', help="Path including text prompts for each classes")
    parser.add_argument("--output_path", type=str, default='data/train_new', help="Output path for the synthetic data")
    parser.add_argument("--part", type=int, default=1, help="For distributed training.")
    parser.add_argument("--total_parts", type=int, default=4, help="For distributed training.")
    args = parser.parse_args()

    # generation settings
    noises = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    text_guidance = [10]
    random_seed = [10, 20, 30, 40, 42, 50, 60, 70]

    # loading stable diffusion pipeline
    pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(model_id_or_path, torch_dtype=torch.float16, variant="fp16",
                                                            use_safetensors=True).to(device)

    property_file = args.prompt_json
    with open(property_file, 'r') as f:
        dict_result = json.load(f)

    df_generate = pd.read_csv(args.data_csv)
    if 'Unnamed: 0' in df_generate.columns:
        del df_generate['Unnamed: 0']
    df_generate.drop_duplicates(inplace=True)

    dict_byclassname = dict()
    for row_id, row in df_generate.iterrows():
        filepath = row[0]
        img_name = row[1]
        class_name = row[3]
        if class_name not in dict_byclassname:
            dict_byclassname[class_name] = []
        dict_byclassname[class_name].append([img_name, filepath])

    ########################################
    # generate from beginning
    list_all = list(dict_byclassname.keys())
    list_all = sorted(list_all, reverse=False)
    list_index = list(split(range(len(list_all)), args.total_parts))
    list_cur_parts = list_index[args.part - 1]
    list_folder = [list_all[i] for i in list_cur_parts]
    print(list_folder)

    for cls_name in tqdm(list_folder):
        underscore = cls_name.replace(' ', '_')
        print(f'generate image for {cls_name}')

        output_folder = f'{args.output_path}/{underscore}/'
        os.makedirs(output_folder, exist_ok=True)

        all_generated_img = gene_imgs(output_folder)
        img_list = dict_byclassname[cls_name]
        for img_pair in img_list:
            img_name = img_pair[0]
            img_path = img_pair[1]
            img_path = img_path.replace('../', '')
            # check if the image is generated 
            if img_name in all_generated_img:
                # generated, skip
                continue

            # directly use original img to generate new images
            init_image = Image.open(img_path).convert("RGB")

            # check clip score
            text_prompt = f"a photo of {cls_name}"
            text_embedding = get_single_text_embedding(tokenizer, clip_model, text_prompt)
            real_img_embedding = get_single_image_embedding(processor, clip_model, init_image)

            cur_prompt = f"{cls_name}"
            list_res = sd_generate(init_image, cls_name, random_seed, noises, text_guidance, output_folder, img_name,
                                   (real_img_embedding, text_embedding))
    print(f"finish generation!")
