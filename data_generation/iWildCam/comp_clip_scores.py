import os
import torch
import pickle
from PIL import Image
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import argparse

# Set the device
device = "cuda" if torch.cuda.is_available() else "cpu"
# Define the model ID
model_id_or_path = "openai/clip-vit-base-patch32"


def get_model_info(model_ID, device):
    # Save the model to device
    model = CLIPModel.from_pretrained(model_ID).to(device)
    # Get the processor
    processor = CLIPProcessor.from_pretrained(model_ID)
    # Get the tokenizer
    tokenizer = CLIPTokenizer.from_pretrained(model_ID)
    # Return model, processor & tokenizer
    return model, processor, tokenizer


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


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--syn_path", type=str, default='data/train_new', help="Output path for the synthetic data")
    parser.add_argument("--real_path", type=str, default='data/iwildcam/iwildcam_v2.0/train',
                        help="Output path for the synthetic data")
    parser.add_argument("--threshold", type=float, default=0.25, help="CLIPScore threshold")

    args = parser.parse_args()

    with open(f'assets/metadata/wilds_common_names.pkl', 'rb') as f:
        dict_name = pickle.load(f)

    # Get model, processor & tokenizer
    model, processor, tokenizer = get_model_info(model_id_or_path, device)

    dict_clip_res = dict()
    root_folder = args.syn_path

    list_folder = os.listdir(root_folder)

    for sp_name in tqdm(list_folder):
        sp_name_under = sp_name.replace(' ', '_')
        if sp_name == 'empty':
            continue
        if sp_name_under not in dict_name:
            continue

        fp_path = str(os.path.join(root_folder, sp_name))
        cmm_name = dict_name[sp_name_under]
        prompt = f"a photo of {cmm_name} in the wild"

        list_subfolder = os.listdir(fp_path)
        # sub categories
        for sub_f in list_subfolder:
            dict_pkl = dict()
            sub_fp = os.path.join(fp_path, sub_f)
            img_f = os.listdir(sub_fp)
            img_f = [item for item in img_f if '.pkl' not in item and 'concat' not in item]
            for img_id in img_f:
                img_fp = os.path.join(sub_fp, img_id)

                real_img_fp = f'{args.real_path}/{img_id}'

                init_image = Image.open(img_fp).convert("RGB")
                init_image = init_image.resize((480, 270))
                real_image = Image.open(real_img_fp).convert("RGB")
                init_image = init_image.resize((480, 270))

                img_embedding = get_single_image_embedding(processor, model, init_image)
                real_img_embedding = get_single_image_embedding(processor, model, real_image)
                text_embedding = get_single_text_embedding(tokenizer, model, prompt)

                text_sim = cosine_similarity(img_embedding, text_embedding)[0][0]
                img_sim = cosine_similarity(img_embedding, real_img_embedding)[0][0]

                cur_name = f"{sp_name}={sub_f}={img_id.replace('.jpg', '')}"
                dict_clip_res[cur_name] = [text_sim, img_sim]
                dict_pkl[img_id.replace('.jpg', '')] = img_embedding

            with open(f'{sub_fp}/clip_emb.pkl', 'wb') as f:
                pickle.dump(dict_pkl, f)

    with open(f'{args.syn_path}/clip_score.pkl', 'wb') as f:
        pickle.dump(dict_clip_res, f)

    # filtered synthetic data
    list_filtered = list(dict_clip_res.items())
    list_filtered = [[item[0].split('='), item[1], item[2]] for item in list_filtered]
    list_filtered = [item for item in list_filtered if item[1] >= args.threshold]

    with open(f'{args.syn_path}/filtered_results.pkl', 'wb') as f:
        pickle.dump(list_filtered, f)
