import os
import json
import pickle
import base64
import urllib3
import requests
import argparse
import traceback
import pandas as pd

from tqdm import tqdm
from PIL import Image
from time import sleep
from urllib3.exceptions import InsecureRequestWarning

from prompt.prompt_sys_en import *


urllib3.disable_warnings(InsecureRequestWarning)
def append_to_pkl(file_path, new_data):
    with open(file_path, 'wb') as f:
        pickle.dump(new_data, f)

def save_inference_results(all_infered_datas, save_path
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, 'w') as f:
        json.dump(all_infered_datas, f, ensure_ascii=False,indent=4)
    print(f"inference results saved at {save_path}")

def get_eval_data(data_path):
    file_path = os.path.join(data_path,'VisuRiddles_source.json')
    data_list = json.load(open(file_path))
    for idx,item in enumerate(data_list):
        item['imgs'] = [os.path.join(data_path,img_item) for img_item in item['imgs']]

    return data_list

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def parse_gemini(response):
    try:
        ans = response['choices'][0]['message']['content']
    except:
        import pdb;pdb.set_trace()
        ans = 'answer failure'
    return ans

def model_call(model_name, text_input, icon_path=None):
    URL = "https://yunwu.ai/v1/chat/completions
    HEADERS = {
            "Content-Type": "application/json",
            "Authorization": "your token",
        }
    if icon_path is not None:
        if isinstance(icon_path,str):
            base64_image = encode_image(icon_path)
            body = {
                "model": model_name,
                "max_tokens": 32768,
                "messages": [
                    {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": text_input
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                        ]
                    }
                ],
                "stream": False,
                "temperature": 0.0
            }
        elif isinstance(icon_path,list):
            base64_images = [encode_image(img) for img in icon_path]
            body = {
                "model": model_name,
                "max_tokens": 32768,
                "messages": [
                    {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": text_input
                        },
                        ] + [
                            {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        } for base64_image in base64_images
                        ]
                    }
                ],
                "stream": False,
                "temperature": 0.0
            }
    else:
        body = {
            "model": model_name,
            "max_tokens": 4096,
            "messages": [
                {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": text_input
                    }
                    ]
                }
            ],
            "stream": False,
            "temperature": 0.0
        }
    response = requests.post(URL, json=body, headers=HEADERS, verify=False).json()
    ans = parse_gemini(response)
    return ans

def inference_online(dataset,save_name,args):
    infered_datas = []
    rank = 0
    world_size=1
    sheet_indices = list(range(rank, len(dataset), world_size))
    lt = len(sheet_indices)
    temp_path = os.path.join(args.save_dir,f'{save_name}_{rank}.pkl')
    if os.path.exists(temp_path):
        infered_datas = pickle.load(open(temp_path,'rb'))
    else:
        infered_datas = []
    processed_id = [item['id'] for item in infered_datas]
    for i in tqdm(range(lt)):
        data = dataset[sheet_indices[i]]
        if data['id'] in processed_id:
            continue
        query,option,answer,images,id = data['question'],data['option'],data['gold_answer'],data['imgs'],data['id']
        if data['class']=='sudoku':
            if args.use_cot:
                prompt = prompt_sudoku_cot
            else:
                prompt = prompt_sudoku_direct
        elif data['class']=='raven':
            if args.use_cot:
                prompt = prompt_raven_cot
            else:
                prompt = prompt_raven_direct
        else:
            if args.use_cot:
                prompt = prompt_choice_cot
            else:
                prompt = prompt_choice_direct
        text_input = prompt + '\n' + query + '\n' + '\n'.join(['<image>']*len(images)) + '\n' + option
        while True:
            try:
                out = model_call(model_name=args.model, text_input=text_input, icon_path=images)
            except Exception as e:
                error_message = traceback.format_exc()
                print("Error:\n", error_message)
                continue
            if out != 'answer failure':
                break
        print(f"rank:{rank}-query:{query}-res:{out}",flush=True)
        data['id']=id
        data['pred']=out
        infered_datas.append(data)
        append_to_pkl(temp_path, infered_datas)
        

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default='gemini-2.0-flash-thinking-exp-01-21', type=str)    
    parser.add_argument("--dataset_path", default='/path/to/dataset_abstract',type=str)
    parser.add_argument("--save_dir", default='output', type=str)
    parser.add_argument("--use_cot",default=False,type=bool)
    parser.add_argument("--rank", default=0, type=int)
    parser.add_argument("--local_rank", default=0, type=int)
    args = parser.parse_args()
    save_name = f"{args.model.split('/')[-1]}_abstract_graph"
    save_path = os.path.join(args.save_dir,f'{save_name}.json')
    rank=0
    if rank==0:
        temp_all_data = []
        if os.path.exists(save_path):
            temp_all_data += json.load(open(save_path,'r',encoding='utf-8'))
        for i in range(8):
            temp_file = os.path.join(save_path,f'{save_name}_{i}.pkl')
            if os.path.exists(temp_file):
                with open(temp_file, 'rb') as f:
                    temp_all_data.extend(pickle.load(f))
        temp_all_data = sorted(temp_all_data,key = lambda item:item['id'])
        unique_data = {}
        for item in temp_all_data:
            if item['id'] not in unique_data:
                unique_data[item['id']] = item
    all_infered_datas = []
    if os.path.exists(save_path):
        all_infered_datas += json.load(open(save_path,'r',encoding='utf-8'))
    infered_ids = [d['id'] for d in all_infered_datas]

    dataset = get_eval_data(args.dataset_path)
    inference_online(dataset,save_name,args)
    if rank == 0:
        for i in range(8):
            temp_file = os.path.join(args.save_dir,f'{save_name}_{i}.pkl')
            if os.path.exists(temp_file):
                with open(temp_file, 'rb') as f:
                    all_infered_datas.extend(pickle.load(f))
                os.remove(temp_file)  
            
        all_infered_datas = sorted(all_infered_datas,key = lambda item:item['id'])
        save_inference_results(all_infered_datas,save_path)
