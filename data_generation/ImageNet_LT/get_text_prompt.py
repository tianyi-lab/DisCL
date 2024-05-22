import openai
import pandas as pd
import json
import os
from tqdm import tqdm
import regex as re
import argparse

api_key = 'YOUR_API_KEY_HERE'
client = openai.OpenAI(api_key=api_key, )

template = lambda \
        x: f"Please provide 10 language descriptions for random scenes that contain only the class '{x}' from the ImageNet Longtaill dataset. Each description should be different and contain a minimum of 15 words. These descriptions will serve as a guide for Stable Diffusion in generating images. answer in format of: prompt1\nprompt2\n"


def query_chatgpt(class_name):

    message = [{"role": "user", "content": template(x=class_name)}]

    response = client.chat.completions.create(model="gpt-3.5-turbo-0125", messages=message)
    return response


def process_prompt_v1(resp_list):
    cur_dict = {}
    for prop in resp_list:
        if ': ' in prop:
            prop = prop.split(': ')
        else:
            prop = prop.split(':')

        prop_name = prop[0]
        props = prop[1].replace('[', '').replace(']', '').split(', ')
        cur_dict[prop_name.lower()] = props
    return cur_dict


def process_prompt_v2(resp_list):
    cur_res = []
    for prop in resp_list:
        prop = prop.strip()
        prop = re.sub(r'^\d+\.\s+', '', prop)
        cur_res.append(prop)
    return cur_res


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_csv", type=str, default='', help="Path including original image path / species names")
    parser.add_argument("--prompt_json", type=str, default='', help="Path including text prompts for each classes")
    args = parser.parse_args()

    df_generate = pd.read_csv(args.data_csv)
    list_labels = df_generate['classname'].unique().tolist()

    output_file = args.prompt_json
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            dict_result = json.load(f)

        list_labels = list(set(list_labels) - set(dict_result.keys()))

    dict_result = dict()
    with open(output_file.replace('json', 'txt'), 'w') as f:
        for class_name in tqdm(list_labels):
            response = query_chatgpt(class_name=class_name)
            return_resp = response.choices[0].message.content
            resp_list = return_resp.split('\n')

            while len(resp_list) <= 8:
                response = query_chatgpt(class_name=class_name)
                return_resp = response.choices[0].message.content
                resp_list = return_resp.split('\n')

            cur_res = process_prompt_v2(resp_list)
            dict_result[class_name] = cur_res

        f.write(return_resp + '\n')

    with open(output_file, 'w') as f:
        json.dump(dict_result, f, indent=4)
