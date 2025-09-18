# -*- coding: UTF-8 -*-
'''
@Project: RationAnomaly
@File   : api.py
@IDE    :
@Author : XuSong
@Date   : 2025/3/16 14:50
'''
import gc
# api_call_example.py
import json
import os
import pickle
import time
import sys
import traceback
import re

import pandas as pd
import torch
import multiprocessing as mp
from vllm import LLM, SamplingParams

import argparse

# 1. 创建 ArgumentParser 对象
parser = argparse.ArgumentParser()

# 2. 添加参数
# add_argument() 方法用于添加参数
# '--input' 是可选参数，-i 是其简写
parser.add_argument('--model', '-m', type=int, required=True)

# '--output' 是可选参数，有默认值
parser.add_argument('--testset', '-t', type=int, required=True)

# '--output' 是可选参数，有默认值
parser.add_argument('--conversation', '-c', type=str, required=True)

# 3. 解析参数
args = parser.parse_args()


from llama2_chat_templater import PromptTemplate as PT

def logRL_CoT_template(prompt):
    pt = PT(system_prompt="The user asks a question, and the Assistant solves it.The assistant first thinks about the reasoning process in the mind and then provides the user with the final answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>. Now the user asks you to solve a log analysing reasoning problem. After thinking, when you finally reach a conclusion, clearly provide the answer within <answer> </answer> tags.")
    pt.add_user_message(prompt)
    prefix = pt.build_prompt()
    return prefix

def logRL_template(prompt):
    pt = PT(system_prompt="You are a log anomaly detection expert, specifically responsible for classifying system log entries as 'normal' or 'abnormal'. Please classify according to the following rules:1. If the log indicates expected system operations, status reports, or known security warnings, classify it as 'normal'.2. If the log indicates hardware failures, security breaches, configuration errors, or unknown severe incidents, classify it as 'abnormal'.")
    pt.add_user_message(prompt)
    prefix = pt.build_prompt()
    return prefix

def llama_2_template(prompt):
    pt = PT()
    pt.add_user_message(prompt)
    prompt = pt.build_prompt()
    return prompt

def vllm_infer(model_path, dataset_path, output_file, template_use):
    # 与 llamafactory 一致
    sampling_params = SamplingParams(temperature=0.1, top_p = 0.7, max_tokens=1024)

    prompts = []
    results = []

    llm = LLM(
        model=model_path,
        max_model_len=4096,# llama3.1 支持上下文长度为128k
        trust_remote_code=True,
        tokenizer=model_path,
        tokenizer_mode='auto',
        gpu_memory_utilization=0.9,
        tensor_parallel_size=4
    )

    with open(dataset_path, 'r', encoding='utf-8') as file:
        items = json.load(file)
        print(len(items))
        for item in items:
            # 修改 LogEval instruction
            # if 'LogEval' in dataset_path:
            #     item["instruction"] = 'Convert the following logs into standardized templates by identifying and replacing the variable parts with a <*>.'
            prompt = item["instruction"] + "\n" + item["input"]

            # 切换对话模板
            if template_use == 'default':
                prompts.append(llama_2_template(prompt))
            if template_use == 'log':
                prompts.append(logRL_template(prompt))
            if template_use == 'cot':
                prompts.append(logRL_CoT_template(prompt))

    try:
        outputs = llm.generate(prompts, sampling_params)
        for i,output in enumerate(outputs):
            instruction = items[i]["instruction"]
            input = items[i]["input"]
            predict = output.outputs[0].text
            results.append({"instruction": instruction,"input": input, "output": predict})
            # Save periodically every 100 entries
            if len(results) % 100 == 0:
                with open(output_file, 'w', encoding='utf-8') as json_file:
                    json.dump(results, json_file, ensure_ascii=False, indent=4)
        json.dump(results, open(output_file, 'w', encoding="utf-8"), ensure_ascii=False, indent=4)
    except Exception as e:
        print(f"An error occurred: {e}")
        print(parse_exception_traceback(e))
    finally:
        # 清理逻辑
        if llm is not None:
            from vllm.distributed.parallel_state import destroy_model_parallel
            ...
            destroy_model_parallel()
            # Isn't necessary for releasing memory, but why not
            # del llm.llm_engine.model_executor.driver_worker
            # del llm  
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.empty_cache()  # 释放CUDA缓存
            print("Clean finished!")


def eval_infer(model_id, testset_id, template_use):

    # 切换测试集后需要切换对话模板
    # 读取JSON文件
    with open("model_testset_list.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    models = data['models']
    testsets = data['testsets']
    model_paths = data['model_paths']
    testset_paths = data['testset_paths']


    model = models[model_id]
    testset = testsets[testset_id]

    model_path = model_paths[model]
    dataset_path = testset_paths[testset]
    output_file = f'../datasets/results/original_vllm_output/{testset}_by_{model}.json'


    print(f'Running inference by {model} on {testset}...')
    # 此处推理，会使用模型的chat_template
    vllm_infer(model_path, dataset_path, output_file, template_use)
    print("Inference finished!")

# 打印异常完整信息
def parse_exception_traceback(exception):
    exc_type, exc_value, exc_tb = sys.exc_info()
    exception_str = traceback.format_exception(exc_type, exc_value, exc_tb)
    return (f'Exception Type: {type(exception)}\n'
            f'  Message: {exception}\n'
            f'TraceBack:\n\n' + ''.join(exception_str))

if __name__ == '__main__':
    # single_test()
    os.environ['VLLM_WORKER_MULTIPROC_METHOD']='spawn'
    mp.set_start_method('spawn')
    # 禁用资源跟踪器
    # os.environ['PYTHONWARNINGS'] = 'ignore::UserWarning'
    model_id = args.model
    testset_id = args.testset
    template = args.conversation
    eval_infer(model_id, testset_id, template)
