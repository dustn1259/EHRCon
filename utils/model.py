import torch
import os
from transformers import pipeline
from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, AutoConfig
from transformers import AutoTokenizer, AutoModelForTokenClassification
from time import time, sleep
import openai
import re

def chatgpt_completion(prompt, model,api_key,api_base):
    max_retry = 3
    retry = 0
    while True:
        try:
            openai.api_key = api_key
            openai.api_type = 'azure'
            openai.api_base = api_base
            openai.api_version = "2023-09-01-preview"
            response = openai.ChatCompletion.create(
                engine=model,
                temperature = 0,
                messages=[
                {"role": "user", "content": prompt}]            )
            text = response['choices'][0]['message']['content'].strip()
            text = re.sub('[\r\n]+', '\n', text)
            text = re.sub('[\t ]+', ' ', text)
            filename = '%s_ChatGPT.txt' % time()
            if not os.path.exists('ChatGPT_logs'):
                os.makedirs('ChatGPT_logs')
            return text
        except Exception as oops:
            retry += 1
            if retry >= max_retry:
                return "ChatGPT error: %s" % oops
            print('Error communicating with OpenAI:', oops)
            sleep(1)
            
        
def load_llama3_70b(chat:bool=False, quantization:str="no"):
    """
    Loads Llama2-70B model. We can only use this by using 4-bit quantization if we were to use 1 GPU.
    Parameters:
        chat (bool): Either to use instruction-tuned model or not. Default: True
        quantization (bool): Either to quantize the model or not. Default: True
    Returns:
        model: model
        tokenizer: tokenizer
        config: configurations
    """
    tokenizer = AutoTokenizer.from_pretrained("unsloth/llama-3-70b-Instruct-bnb-4bit")
    config = AutoConfig.from_pretrained("unsloth/llama-3-70b-Instruct-bnb-4bit")

    pipeline = transformers.pipeline(
        "text-generation",
        model="unsloth/llama-3-70b-Instruct-bnb-4bit",
        model_kwargs={"torch_dtype": torch.bfloat16, 
                    "cache_dir":"/nfs_data_storage/huggingface/", 
                    'quantization_config': {'load_in_4bit': True}, 
                    'low_cpu_mem_usage': True},
    )

    return pipeline, tokenizer, config


def load_mixtral(it:bool=False, quantization:str="no",num_threads:int=None):
    """
    Loads Mixtral-8x7B model.
    Supports only instruction-tuned version at the moment.

    Parameters:
        quantization (bool): Either to quantize the model or not. Default: True

    Returns:  
        model: model
        tokenizer: tokenizer
        config: configurations
    """
    if num_threads is not None:
        torch.set_num_threads(num_threads)
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    model_dir = 'mistralai/Mixtral-8x7B-Instruct-v0.1'
    if it:
        if quantization != "no":
            model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto", quantization_config=bnb_config)
        else:
            model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        config = AutoConfig.from_pretrained(model_dir)
    else:
        if quantization != "no":
            model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto", quantization_config=bnb_config)
        else:
            model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(model_dir) 
        config = AutoConfig.from_pretrained(model_dir)
    return model, tokenizer, config


def load_tulu2_70b(quantization:str="no",num_threads:int=None):
    """
    Loads Llama2-70B model. We can only use this by using 4-bit quantization if we were to use 1 GPU.

    Parameters:
        chat (bool): Either to use instruction-tuned model or not. Default: True
        quantization (bool): Either to quantize the model or not. Default: True

    Returns:  
        model: model
        tokenizer: tokenizer
        config: configurations
    """
    if num_threads is not None:
        torch.set_num_threads(num_threads)
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    model_dir = "allenai/tulu-2-dpo-70b"
    
    if quantization != "no":
        model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto", quantization_config=bnb_config)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_dir) 
    config = AutoConfig.from_pretrained(model_dir)
    return model, tokenizer, config

def load_model(model_name: str):
    if model_name in ["tulu2"]:
        model, tokenizer, config = load_tulu2_70b(quantization="4bits")
        return model, tokenizer,config
    elif model_name in ["mixtral"]:
        model, tokenizer, config = load_mixtral(quantization="4bits")
        return model, tokenizer,config
    elif model_name in ["llama3"]:
        model, tokenizer, config = load_llama3_70b(quantization="4bits")
        return model,tokenizer,config
    else:
        raise Exception('Invalid model.')
        
def open_source_model_inference_llama(message, pipeline, tokenizer, config):
    messages = [
        {"role": "system", "content": "You are a logical and helpful medical assistant."},
        {"role": "user", "content": message},
    ]
    prompt = pipeline.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
    )

    terminators = [
        pipeline.tokenizer.eos_token_id,
        pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    outputs = pipeline(
        prompt,
        max_new_tokens=1024,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.2,
        top_p=0.1,
        )

    return outputs[0]["generated_text"][len(prompt):]

def open_source_model_inference(message, model, tokenizer, config):
    """
    Inference for open-source models.
    Tantamount to `gpt35_inference`.
    Parameters:
        message (str): prompt for the model
        model: model
        tokenizer: tokenizer
        config: configuration
    Returns:
        generated_text: generated text of the model. We disregard the input prompt from the output of the model.
    """
    input_ids = None
    try:
        input_ids = tokenizer(message, return_tensors="pt", max_length=config.max_position_embeddings, truncation=True).to("cuda").input_ids
    except:
        input_ids = tokenizer(message, return_tensors="pt", max_length=4096, truncation=True).to("cuda").input_ids
    input_len = len(input_ids[0])
    generated_ids = model.generate(input_ids, max_length=input_len+1024, temperature=0.2, top_p=0.1)
    generated_text = tokenizer.decode(generated_ids[0][input_len:], skip_special_tokens=True)
    return generated_text
