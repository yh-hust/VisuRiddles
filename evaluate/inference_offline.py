import json
import requests
import time
import os
import io
import re
import json
import glob
import pickle
import pandas as pd
import argparse
import numpy as np

from PIL import Image
from glob import glob
from tqdm import tqdm
from PIL import Image
from tqdm import tqdm

import torch
import torch.distributed as dist
from transformers import AutoProcessor,AutoTokenizer

from prompt_sys_en import *


def get_rank_and_world_size():
    rank = dist.get_rank()  
    world_size = dist.get_world_size() 
    return rank, world_size

def get_eval_data(data_path):
    file_path = os.path.join(data_path,'VisuRiddles_source.json')
    data_list = json.load(open(file_path))
    for idx,item in enumerate(data_list):
        item['imgs'] = [os.path.join(data_path,img_item) for img_item in item['imgs']]
    return data_list

def save_inference_results(all_infered_datas, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, 'w') as f:
        json.dump(all_infered_datas, f, ensure_ascii=False,indent=4)
    print(f"inference results saved at {save_path}")

def append_to_pkl(file_path, new_data):
    with open(file_path, 'wb') as f:
        pickle.dump(new_data, f)

def load_qwen2_5_vl_model(args):
    min_pixels = 256*28*28
    max_pixels = 32768*28*28
    mm_processor_kwargs = {"min_pixel":min_pixels,"max_pixels":max_pixels}
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tp,
        max_model_len=32768,
        mm_processor_kwargs=mm_processor_kwargs,
        limit_mm_per_prompt={"image": 2},
    )
    processor = AutoProcessor.from_pretrained(args.model, min_pixels=min_pixels, max_pixels=max_pixels)
    return llm,processor

def load_internvl2_5_model(args):
    llm = LLM(model=args.model,
        tensor_parallel_size=args.tp,
        trust_remote_code=True,
        max_model_len=8192,
        limit_mm_per_prompt={"image": 2},
        mm_processor_kwargs={"max_dynamic_patch": 1})
    tokenizer = AutoTokenizer.from_pretrained(args.model,
                                              trust_remote_code=True)
    return llm,tokenizer

def load_deepseek_vl2(args):
    from transformers import AutoModelForCausalLM
    from deepseek_vl2.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM
    llm = AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=True).to(torch.bfloat16).cuda().eval()
    processor = DeepseekVLV2Processor.from_pretrained(args.model)


    return llm,processor

def load_mincpmv(args):
    from transformers import AutoModel, AutoTokenizer
    model = AutoModel.from_pretrained(
        args.model, trust_remote_code=True,attn_implementation='flash_attention_2', torch_dtype=torch.bfloat16
    ).eval().cuda()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    return model,tokenizer

def load_qwq(args):
    from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
    llm = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model, torch_dtype="auto", device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(args.model,trust_remote_code=True)
    return llm,processor

def load_kimivl(args):
    from transformers import AutoModelForCausalLM
    llm = AutoModelForCausalLM.from_pretrained(
    args.model,
    torch_dtype="auto",
    device_map="auto",
    trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(args.model,trust_remote_code=True)
    return llm,processor

def load_model(args):
    if 'qwen' in args.model.lower():
        return load_qwen2_5_vl_model(args)
    elif 'internvl' in args.model.lower():
        return load_internvl2_5_model(args)
    elif 'deepseek' in args.model.lower():
        return load_deepseek_vl2(args)
    elif 'minicpm' in args.model.lower():
        return load_mincpmv(args)
    elif 'qwq' in args.model.lower():
        return load_qwq(args)
    elif 'kimi' in args.model.lower():
        return load_kimivl(args)
    elif 'qvq' in args.model.lower():
        return load_qwq(args)
    else:
        # qwq  kimi-vl
        return None,None


def qwen_preprocess(query,option,images,processor,prompt):
    messages = [
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": query+'\n'+option},
                ] + [
                    {
                        "type": "image",
                        "image": image,
                        "min_pixels": 256 * 28 * 28,
                        "max_pixels": 32768 * 28 * 28,
                    } for image in images
                ],
            },
        ]
    text = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
    image_inputs, video_inputs = process_vision_info(messages)
    mm_data = {}
    if image_inputs is not None:
        mm_data["image"] = image_inputs
    llm_inputs = {
            "prompt": text,
            "multi_modal_data": mm_data,
            }
    sampling_params = SamplingParams(
    temperature=0.0,
    top_p=0.001,
    max_tokens=8192,
    repetition_penalty=1.05
    )
    return llm_inputs,sampling_params

def intern_preprocess(query,option,images,tokenizer,prompt):
    question = query+'\n'+option

    placeholders = "\n".join(f"Image-{i}: <image>\n"
                             for i, _ in enumerate(images))
    messages = [
            {"role": "system", "content": prompt},
            {
                "role":'user','content':f"{placeholders}\n{question}"
            }
    ]
    text = tokenizer.apply_chat_template(messages,
                                           tokenize=False,
                                           add_generation_prompt=True)
    stop_tokens = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|end|>"]
    stop_token_ids = [tokenizer.convert_tokens_to_ids(i) for i in stop_tokens]

    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=0.001,
        max_tokens=8192,
        repetition_penalty=1.05,
    )
    mm_data = {"image":[Image.open(image).convert("RGB") for image in images]}
    llm_inputs = {
            "prompt": text,
            "multi_modal_data": mm_data,
            }
    return llm_inputs,sampling_params

def deepseekvl_preprocess(query,option,images,tokenizer,prompt):
    question = prompt+'\n' + query+'\n'+option
    placeholder = "".join(f"image_{i}:<image>\n"
                          for i, _ in enumerate(images))
    text = f"<|User|>: {question}\n{placeholder}\n<|Assistant|>:"
    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=0.001,
        max_tokens=32,
        repetition_penalty=1.05,
    )
    mm_data = {"image":[Image.open(image).convert("RGB") for image in images]}
    llm_inputs = {
            "prompt": text,
            "multi_modal_data": mm_data,
            }
    return llm_inputs,sampling_params

def minicpm_preprocess(query,option,images,tokenizer,prompt):
    questions = [prompt+'\n' + query+'\n'+option]
    modality = "image"
    stop_tokens = ['<|im_end|>', '<|endoftext|>']
    stop_token_ids = [tokenizer.convert_tokens_to_ids(i) for i in stop_tokens]

    modality_placeholder = {
        "image": "(<image>./</image>)",
        "video": "(<video>./</video>)",
    }

    prompts = [
        tokenizer.apply_chat_template(
            [{
                'role': 'user',
                'content': f"{modality_placeholder[modality]}\n{question}"
            }],
            tokenize=False,
            add_generation_prompt=True) for question in questions
    ]
    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=0.001,
        max_tokens=1024,
        repetition_penalty=1.05,
    )
    mm_data = {"image":[Image.open(image).convert("RGB") for image in images]}
    llm_inputs = {
            "prompt": prompts[0],
            "multi_modal_data": mm_data,
            }
    return llm_inputs,sampling_params

def kimivl_preprocess(query,option,images,processor,prompt):
    messages = [
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": query+'\n'+option},
                    {"type": "image","image":images}
                ],
            },
        ]
    prompt = processor.apply_chat_template(messages,
                                           tokenize=False,
                                           add_generation_prompt=True)
    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=0.001,
        max_tokens=64,
        repetition_penalty=1.05
    )
    mm_data = {"image":Image.open(images[0]).convert("RGB")}
    llm_inputs = {
            "prompt": prompt,
            "multi_modal_data": mm_data,
            }
    return llm_inputs,sampling_params

def qwq_preprocess(query,option,images,processor,prompt):
    messages = [
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": query+'\n'+option},
                ] + [
                    {
                        "type": "image",
                        "image": image,
                        "min_pixels": 256 * 28 * 28,
                        "max_pixels": 32768 * 28 * 28,
                    } for image in images
                ],
            },
        ]
    text = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
    image_inputs, video_inputs = process_vision_info(messages)
    mm_data = {}
    if image_inputs is not None:
        mm_data["image"] = image_inputs
    llm_inputs = {
            "prompt": text,
            "multi_modal_data": mm_data,
            }
    sampling_params = SamplingParams(
    temperature=0.0,
    top_p=0.001,
    max_tokens=1024,
    repetition_penalty=1.05
    )
    return llm_inputs,sampling_params

def kimivl_qvq_transformers_inference(prompt,query,option,images,processor,model):
    messages = [
            {
                "role": "user", 
                "content": [ 
                    {
                        "type": "text", "text": prompt+'\n' + query+'\n'+option
                    },
                    {
                        "type": "image", "image": images[0]
                    },
                    ]
            }
    ]
    text = processor.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
    inputs = processor(images=Image.open(images[0]), text=text, return_tensors="pt", padding=True, truncation=True).to(model.device)
    generated_ids = model.generate(**inputs, max_new_tokens=4096,repetition_penalty=1.5)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    out = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    return out

def deepseek_transformers_inference(prompt,query,option,images,processor,model):
    from deepseek_vl2.utils.io import load_pil_images
    question = prompt+'\n' + query+'\n'+option
    tokenizer = processor.tokenizer
    image_places = ''.join([f'image_{i+1}' for i in range(len(images))])
    conversation = [
    {
        "role": "<|User|>",
        "content": f"{image_places}\n<|ref|>{question}<|/ref|>.",
        "images": images,
    },
    {"role": "<|Assistant|>", "content": ""},
    ]
    pil_images = load_pil_images(conversation)
    prepare_inputs = processor(
        conversations=conversation,
        images=pil_images,
        force_batchify=True,
        system_prompt=""
    ).to(model.device)
    inputs_embeds = model.prepare_inputs_embeds(**prepare_inputs)

    # run the model to get the response
    outputs = model.language.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=prepare_inputs.attention_mask,
        pad_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=512,
        do_sample=False,
        use_cache=True
    )

    answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=False)
    print(f"{prepare_inputs['sft_format'][0]}", answer)
    return answer

def minicpm_transformers_inference(prompt,query,option,images,processor,model):
    question = prompt+'\n' + ' '.join([f'image {i+1}' for i in range(len(images))])+ '\n' + query + '\n' + option

    messages = [
            {
                "role": "user", 
                "content": [Image.open(image).convert('RGB') for image in images] + [question]
            }
    ]
    out = model.chat(
        image= None,
        msgs = messages,
        tokenizer=processor
    )
    return out

def inference_transformers(model, processor, dataset, save_name, args):
    assert 'kimi' in args.model.lower() or 'qvq' in args.model.lower() or 'deepseek' in args.model.lower() or 'minicpm' in args.model.lower() 
    infered_datas = []
    buffer = []
    buffer_num=20
    rank=0
    world_size
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
        if 'kimi' in args.model.lower():
            prompt += '\nDo not use other tools.'
        if 'kimi' in args.model.lower() or 'qvq' in args.model.lower():
            out = kimivl_qvq_transformers_inference(prompt,query,option,images,processor,model)
        elif 'deepseek' in args.model.lower():
            out = deepseek_transformers_inference(prompt,query,option,images,processor,model)
        elif 'minicpm' in args.model.lower():
            out = minicpm_transformers_inference(prompt,query,option,images,processor,model)
        else:
            out = None
        data['pred']=out
        print(f"rank:{rank}-query:{query}-res:{out}")
        torch.cuda.empty_cache()
        sample = {
            'id': id,
            'Q': query,
            'answer': answer,
            'pred': out,
        }
        infered_datas.append(data)
        append_to_pkl(temp_path, infered_datas)

def inference(llm, processor, dataset, save_name, args):
    infered_datas = []
    buffer = []
    buffer_num=20
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

        if 'qwen' in args.model.lower():
            llm_inputs,sampling_params =  qwen_preprocess(query,option,images,processor,prompt)
        elif 'internvl' in args.model.lower():
            llm_inputs,sampling_params = intern_preprocess(query,option,images,processor,prompt)
        elif 'deepseek' in args.model.lower():
            llm_inputs,sampling_params = deepseekvl_preprocess(query,option,images,processor,prompt)
        elif 'minicpm' in args.model.lower():
            llm_inputs,sampling_params = minicpm_preprocess(query,option,images,processor,prompt)
        elif 'kimi' in args.model.lower():
            llm_inputs,sampling_params = kimivl_preprocess(query,option,images,processor,prompt)
        elif 'qwq' in args.model.lower():
            llm_inputs,sampling_params = qwq_preprocess(query,option,images,processo,prompt)
        else:
            llm_inputs = {
                "prompt":prompt + '\n' + query + '\n' + '<image>\n' + option,
                "multi_modal_data": {"image":Image.open(images[0])},
            }
        outputs = llm.generate([llm_inputs], sampling_params=sampling_params)
        out = outputs[0].outputs[0].text
        print(f"rank:{rank}-query:{query}-res:{out}")
        torch.cuda.empty_cache()
        data['id']=id
        data['pred']=out
        infered_datas.append(data)
        append_to_pkl(temp_path, infered_datas)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default='', type=str)    
    parser.add_argument("--dataset_path", default='',type=str)
    parser.add_argument("--tp",default=1,type=int)
    parser.add_argument("--save_dir", default='output', type=str)
    parser.add_argument("--use_cot",default=False,type=bool)
    parser.add_argument("--rank", default=0, type=int)
    parser.add_argument("--local_rank", default=0, type=int)
    args = parser.parse_args()
    world_size = int(os.environ.get("WORLD_SIZE",1))
    save_name = f"{args.model.split('/')[-1]}_abstract_graph"
    save_path = os.path.join(args.save_dir,f'{save_name}.json')
    if world_size>1:
        dist.init_process_group(backend='nccl')
        local_rank = os.environ.get('LOCAL_RANK', 0)
        local_rank = dist.get_rank() % torch.cuda.device_count()
        torch.cuda.set_device(local_rank)
    else:
        pass
    torch.cuda.synchronize()
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
                #os.remove(temp_file)
        temp_all_data = sorted(temp_all_data,key = lambda item:item['id'])
        unique_data = {}
        for item in temp_all_data:
            if item['id'] not in unique_data:
                unique_data[item['id']] = item
    #dist.barrier()
    all_infered_datas = []
    if os.path.exists(save_path):
        all_infered_datas += json.load(open(save_path,'r',encoding='utf-8'))
    infered_ids = [d['id'] for d in all_infered_datas]
    if world_size > 1:
        if rank==0:
            dataset = get_eval_data(args.dataset_path)
        else:
            dataset = None
    else:
        dataset = get_eval_data(args.dataset_path)
    if world_size > 1:
        dataset_list = [dataset]
        dataset = dataset_list[0]
    if not ('kimi' in args.model.lower() or 'qvq' in args.model.lower() or 'deepseek' in args.model.lower() or 'minicpm' in args.model.lower()):
        from qwen_vl_utils import process_vision_info
        from vllm import LLM, SamplingParams
    llm,processor=load_model(args
    if 'kimi' in args.model.lower() or 'qvq' in args.model.lower() or 'deepseek' in args.model.lower() or 'minicpm' in args.model.lower():
        inference_transformers(llm, processor, dataset, save_name, args)
    else:
        inference(llm,processor, dataset,save_name,args)
    if world_size > 1:
        dist.barrier()     
    if rank == 0:
        for i in range(world_size):
            temp_file = os.path.join(args.save_dir,f'{save_name}_{i}.pkl')
            if os.path.exists(temp_file):
                with open(temp_file, 'rb') as f:
                    all_infered_datas.extend(pickle.load(f))
                os.remove(temp_file)  
            
        all_infered_datas = sorted(all_infered_datas,key = lambda item:item['id'])
        save_inference_results(all_infered_datas,save_path)
