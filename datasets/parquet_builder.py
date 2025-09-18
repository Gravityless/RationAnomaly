""" Preprocess dataset for RationAnomaly task """

import os
from datasets import Dataset, load_dataset
import argparse
import json
from llama2_chat_templater import PromptTemplate as PT
import pandas as pd

def make_prompt(dp, template_type):

    messages = []
    if template_type == 'base':
        messages = f"""
        The user asks a question, and the Assistant solves it.The assistant first thinks about the reasoning process \
in the mind and then provides the user with the final answer. The reasoning process and answer are enclosed within \
<think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> \
<answer> answer here </answer>. Now the user asks you to solve a log parsing reasoning problem. After thinking, \
when you finally reach a conclusion, clearly state the identity of each character within <answer> </answer> tags. \
A log template only retains static parts in a log and replaces dynamic variables with a symbol of <*>. \
for example, <answer>CE sym <*>, at <*>, mask <*></answer>.\n\nUser:{dp['instruction']} {dp['input']}\nAssistant: <think>"""

    elif template_type == 'old-verl-llama-2-chat-RationAnomaly':
        # 构建系统消息内容
        system_content = """You are a log anomaly detection expert, specifically responsible for classifying system log entries as 'normal' or 'abnormal'. \
Please classify according to the following rules:\
1. If the log indicates expected system operations, status reports, or known security warnings, classify it as 'normal'.\
2. If the log indicates hardware failures, security breaches, configuration errors, or unknown severe incidents, classify it as 'abnormal'."""

        # 构建用户消息内容，合并instruction和input
        user_content = f"{dp['instruction']} {dp['input']}"

        pt = PT(system_prompt=system_content)
        pt.add_user_message(user_content)
        messages = pt.build_prompt()

    elif template_type == 'old-verl-llama-2-chat-RationAnomaly-CoT':
        # 构建系统消息内容
        system_content = """The user asks a question, and the Assistant solves it.The assistant first thinks about the reasoning process in \
the mind and then provides the user with the final answer. The reasoning process and answer are enclosed within \
<think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>. \
Now the user asks you to solve a log analysing reasoning problem. After thinking, when you finally reach a conclusion, \
clearly provide the answer within <answer> </answer> tags."""

        # 构建用户消息内容，合并instruction和input
        user_content = f"{dp['instruction']} {dp['input']}"

        pt = PT(system_prompt=system_content)
        pt.add_user_message(user_content)
        messages = pt.build_prompt()

    elif template_type == 'verl-llama-2-chat-RationAnomaly':
        # 构建系统消息内容
        system_content = """You are a log anomaly detection expert, specifically responsible for classifying system log entries as 'normal' or 'abnormal'. \
Please classify according to the following rules:\
1. If the log indicates expected system operations, status reports, or known security warnings, classify it as 'normal'.\
2. If the log indicates hardware failures, security breaches, configuration errors, or unknown severe incidents, classify it as 'abnormal'."""

        # 构建用户消息内容，合并instruction和input
        user_content = f"{dp['instruction']} {dp['input']}"

        # 构建消息列表
        messages = [
            {'content': system_content, 'role': 'system'},
            {'content': user_content, 'role': 'user'}
        ]

    elif template_type == 'verl-llama-2-chat-RationAnomaly-CoT':
        # 构建系统消息内容
        system_content = """The user asks a question, and the Assistant solves it.The assistant first thinks about the reasoning process in \
the mind and then provides the user with the final answer. The reasoning process and answer are enclosed within \
<think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>. \
Now the user asks you to solve a log analysing reasoning problem. After thinking, when you finally reach a conclusion, \
clearly provide the answer within <answer> </answer> tags."""

        # 构建用户消息内容，合并instruction和input
        user_content = f"{dp['instruction']} {dp['input']}"

        # 构建消息列表
        messages = [
            {'content': system_content, 'role': 'system'},
            {'content': user_content, 'role': 'user'}
        ]

    elif template_type == 'new-verl-llama-2-chat-RationAnomaly':
        user_content = """<<SYS>>You are a log anomaly detection expert, specifically responsible for classifying system log entries as 'normal' or 'abnormal'. \
Please classify according to the following rules:\
1. If the log indicates expected system operations, status reports, or known security warnings, classify it as 'normal'.\
2. If the log indicates hardware failures, security breaches, configuration errors, or unknown severe incidents, classify it as 'abnormal'.<</SYS>>"""
        user_content += f"{dp['instruction']} {dp['input']}"
        messages = [
            {'content': user_content, 'role': 'user'}
        ]
    elif template_type == 'new-verl-llama-2-chat-RationAnomaly-CoT':
        user_content = """<<SYS>>The user asks a question, and the Assistant solves it.The assistant first thinks about the reasoning process in \
the mind and then provides the user with the final answer. The reasoning process and answer are enclosed within \
<think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>. \
Now the user asks you to solve a log analysing reasoning problem. After thinking, when you finally reach a conclusion, \
clearly provide the answer within <answer> </answer> tags.<</SYS>>"""
        user_content += f"{dp['instruction']} {dp['input']}"
        messages = [
            {'content': user_content, 'role': 'user'}
        ]

    return messages

def make_system_prompt(template_type):
    if template_type == 'verl-llama-2-chat-RationAnomaly':
        system_content = "You are a log anomaly detection expert, specifically responsible for classifying system log entries as 'normal' or 'abnormal'. Please classify according to the following rules: 1. If the log indicates expected system operations, status reports, or known security warnings, classify it as 'normal'. 2. If the log indicates hardware failures, security breaches, configuration errors, or unknown severe incidents, classify it as 'abnormal'."

    elif template_type == 'verl-llama-2-chat-RationAnomaly-CoT':
        system_content = "The user asks a question, and the Assistant solves it.The assistant first thinks about the reasoning process in the mind and then provides the user with the final answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>. Now the user asks you to solve a log analysing reasoning problem. After thinking, when you finally reach a conclusion, clearly provide the answer within <answer> </answer> tags."

    return system_content

def make_user_prompt(dp):
    user_content = f"{dp['instruction']} {dp['input']}"
    return user_content

def preprocess_dataset(
    input_file: str,
    output_dir: str,
    data_source: str,
    template_type: str = "verl-llama-2-chat-logIRS",
    hdfs_dir: str = None,
    split: str = "train"
):
    """
    预处理数据集并保存为指定格式
    
    参数:
    input_file -- 输入JSON文件路径
    output_dir -- 输出目录路径
    data_source -- 数据源标识符
    template_type -- 提示模板类型 (默认: verl-llama-2-chat-logIRS)
    hdfs_dir -- HDFS输出目录 (可选)
    split -- 数据集类型 [train|test] (默认: train)
    """
    
    # 创建本地输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 数据加载器
    def gen_from_json(path):
        with open(path, encoding='utf-8') as f:
            dataset = json.load(f)
        print(f'Loaded dataset: {len(dataset)} samples')
        # print(json.dumps(dataset, indent=4, ensure_ascii=False))
        for item in dataset:
            yield item

    # 数据处理映射函数
    def make_map_fn():
        def process_fn(example, idx):
            system_prompt = make_system_prompt(template_type)
            question = make_user_prompt(example)
            prompt = system_prompt + question
            if len(prompt) >= 2048:
                print("Long question detected:", prompt)

            solution = example['label']
            return {
                "data_source": data_source,
                "prompt": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question}
                ],
                "ability": "log",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": solution
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                }
            }
        return process_fn

    # 生成数据集
    base_name = os.path.basename(input_file).split('.')[0]
    dataset = Dataset.from_generator(
        gen_from_json, 
        gen_kwargs={'path': input_file}
    )

    # 处理数据集
    processed_ds = dataset.map(
        function=make_map_fn(),
        with_indices=True
    )

    # 保存Parquet格式
    parquet_path = os.path.join(output_dir, f'{base_name}_{template_type}.parquet')
    processed_ds.to_parquet(parquet_path)

    # 转换为CSV
    # convert parquet to csv
    import pyarrow.parquet as pq

    csv_path = os.path.join(output_dir, f'{base_name}_{template_type}.csv')
    df = pq.read_table(parquet_path).to_pandas()
    df.to_csv(csv_path, index=False)

if __name__ == '__main__':
    
    # RationAnomaly BGL Train Dataset
    train_file_path = 'trainsets/BGL.json'
    parquet_dir = 'trainsets'
    data_source = 'RationAnomaly'
    template_type = 'verl-llama-2-chat-RationAnomaly'

    preprocess_dataset(
        input_file=train_file_path,
        output_dir=parquet_dir,
        data_source=data_source,
        template_type=template_type,
        split='train'
    )