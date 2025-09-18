import re
from typing import Dict, Tuple, Optional
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge
import json
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from typing import Dict, Optional

import logging
# 过滤transformers库中关于loss_type的警告
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

def calculate_perplexity(
    chain_text: str,
    related_info: Optional[str] = None,
    model_name_or_path: str = "gpt2",  # 可替换为领域相关模型
    device: str = "auto"
) -> Dict[str, float]:
    """
    计算思维链文本的困惑度（Perplexity）
    
    参数:
        chain_text: 思维链文本（必填）
        related_info: 相关背景信息（可选，会拼接在思维链前作为上下文）
        model_name_or_path: 预训练语言模型名称或路径（默认使用gpt2，可选领域模型如"microsoft/phi-2"）
        device: 计算设备（"auto"自动选择GPU/CPU，"cpu"强制用CPU）
    
    返回:
        包含困惑度及相关指标的字典
    """
    # 处理设备
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    
    # 拼接上下文（如果有相关信息）
    full_text = f"{related_info}\n{chain_text}" if related_info else chain_text
    
    # 加载预训练模型和分词器
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path).to(device)
    except Exception as e:
        raise ValueError(f"模型加载失败：{str(e)}，请检查model_name_or_path是否正确")
    
    # 确保分词器有pad_token（部分模型默认没有）
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 文本编码
    inputs = tokenizer(
        full_text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=1024  # 根据模型最大序列长度调整
    ).to(device)
    
    # 计算损失
    with torch.no_grad():  # 关闭梯度计算，节省内存
        outputs = model(** inputs, labels=inputs["input_ids"])
        loss = outputs.loss  # 平均交叉熵损失
    
    # 计算困惑度（perplexity = exp(loss)）
    perplexity = torch.exp(loss).item()
    
    # 计算token数量（用于参考）
    token_count = inputs["input_ids"].shape[1]
    
    return {
        "perplexity": perplexity,  # 困惑度（核心指标，值越低越好）
        "loss": loss.item(),       # 交叉熵损失（辅助指标）
        "token_count": token_count,  # 文本token数量（用于解释结果）
        "model_used": model_name_or_path
    }

def extract_solution(solution_str):
    """Extracts the final answer from the model's response string.
    
    Args:
        solution_str: Raw response string from the language model
        
    Returns:
        Tuple containing (extracted_answer, processed_string)
    """

    # Split response to isolate assistant output
    # if "[/INST]" in solution_str:
    #     raw_log = solution_str.split("[/INST]", 1)[0]
    #     processed_str = solution_str.split("[/INST]", 1)[1]
        
    # Extract final answer using XML-style tags
    answer_pattern = r'<answer>(.*?)</answer>'
    matches = list(re.finditer(answer_pattern, solution_str, re.DOTALL))
    
    if not matches:
        # print("[Error] No valid answer tags found")
        return None, None
        
    final_answer = matches[-1].group(1).strip()

    think_pattern = r'<think>(.*?)</think>'
    think_matches = list(re.finditer(think_pattern, solution_str, re.DOTALL))

    if not think_matches:
        # print("[Error] No valid think tags found")
        return None, None
        
    final_think = think_matches[-1].group(1).strip()


    return final_think, final_answer

def extract_ground_truth(ground_truth):
    """Extracts the final answer from the model's response string.
    
    Args:
        ground_truth: Ground truth string from the dataset
        
    Returns:
        Tuple containing (extracted_answer, processed_string)
    """
    # Extract final answer using XML-style tags
    answer_pattern = r'<answer>(.*?)</answer>'
    matches = list(re.finditer(answer_pattern, ground_truth, re.DOTALL))
    final_answer = matches[-1].group(1).strip()

    think_pattern = r'<think>(.*?)</think>'
    think_matches = list(re.finditer(think_pattern, ground_truth, re.DOTALL))       
    final_think = think_matches[-1].group(1).strip()


    return final_think, final_answer

def validate_response_structure(processed_str: str) -> bool:
    """Performs comprehensive validation of response structure.
    
    Args:
        processed_str: Processed response string from the model
        
    Returns:
        Boolean indicating whether all formatting requirements are met
    """
    # print("\n[Structure Validation]")
    validation_passed = True

    # Check required tags
    tags = {
        'think_start': ('<think>', 1),
        'think_end': ('</think>', 1),
        'answer_start': ('<answer>', 1),
        'answer_end': ('</answer>', 1)
    }

    positions = {}
    for tag_name, (tag_str, expected_count) in tags.items():
        count = processed_str.count(tag_str)
        positions[tag_name] = pos = processed_str.find(tag_str)
        
        # print(f"  {tag_str}: count={count}, position={pos}")
        
        if count != expected_count:
            # print(f"  [Error] {tag_str} appears {count} times (expected {expected_count})")
            validation_passed = False

    # Verify tag order
    if (positions['think_start'] > positions['think_end'] or
        positions['think_end'] > positions['answer_start'] or
        positions['answer_start'] > positions['answer_end']):
        # print("  [Error] Incorrect tag order: Expected <think>...</think><answer>...</answer>")
        validation_passed = False
    else:
        # print("  Tag sequence validation passed")
        pass

    return validation_passed

def get_answer_score(ground_truth, answer_text):

    # 全小写转换
    ground_truth = ground_truth.lower().strip()
    answer_text = answer_text.lower().strip()

    if ground_truth == answer_text:
        answer_score = 1.0
        # print("ANSWER: FULL MATCH")
    else:
        answer_score = -1.0
        # print("ANSWER: MISMATCH")

    # 异常召回率修正
    if ground_truth == 'abnormal' and answer_text == 'normal':
        answer_score = answer_score / 0.15
    
    # 异常准确率修正
    if ground_truth == 'normal' and answer_text == 'abnormal':
        answer_score = answer_score / 0.15

    return answer_score

def get_think_scores(ref,pred):
    # 计算BLEU-4得分
    bleu_score = sentence_bleu([ref], pred, weights=(0.25, 0.25, 0.25, 0.25))
    BLEU=100* bleu_score
    # print(f"BLEU-4 Score: {100* bleu_score}")

    # 准备数据用于ROUGE计算
    rouge = Rouge()
    scores = rouge.get_scores(pred, ref, avg=True)
    ROUGE_1=100*scores["rouge-1"]['f']
    ROUGE_2=100*scores["rouge-2"]['f']
    ROUGE_l=100*scores["rouge-l"]['f']
    # print("ROUGE-1:", 100*scores["rouge-1"]['f'])
    # print("ROUGE-2:", 100*scores["rouge-2"]['f'])
    # print("ROUGE-L:", 100*scores["rouge-l"]['f'])
    semantic_score = sum([BLEU,ROUGE_1,ROUGE_2,ROUGE_l]) / 800 - 0.5
    

    # 思维链长度检验
    len_variation = 2 * abs(len(pred) - len(ref)) / (len(ref) + len(pred))
    if len(pred) > len(ref) and len_variation < 1:
        len_score = len_variation
    else:
        len_score = - len_variation

    

    # 困惑度检验
    # 相关信息（可选，例如日志分析的领域知识）
    related_info = "This is a log analysis task to determine if a log entry is normal or abnomal."

    # 计算困惑度（使用轻量模型distilgpt2加快速度）
    ppl_score = 0

    result = calculate_perplexity(
        chain_text=pred,
        related_info=related_info,
        model_name_or_path="/data/xwx1397120/distilgpt2",  # 可选："gpt2", "microsoft/phi-2"（需更大显存）
        device="cpu"  # 若有GPU可改为"cuda"
    )
    ppl_score = (100 - result['perplexity']) / 100

    # if result['perplexity'] > 100:
    #     ppl_score = -0.5
    # if result['perplexity'] < 50:
    #     ppl_score = 0.5
    
    think_score = semantic_score + len_score + ppl_score
    return semantic_score, len_score, ppl_score, think_score

def compute_score(solution_str: str, 
                 ground_truth: str,
                 format_reward: int = 0.1,
                 answer_reward: float = 1.0) :
    """Computes comprehensive score for model response.
    
    Args:
        solution_str: Raw model response string
        ground_truth: Dictionary containing ground truth data
        format_reward: Points awarded/deducted for format correctness
        answer_reward: Points awarded/deducted for answer correctness
        
    Returns:
        Total score (sum of format and answer rewards)
    """

    # Extract model answer
    think_text, answer_text = extract_solution(solution_str)
    think_label, answer_label = extract_ground_truth(ground_truth)

    # Validate response structure
    format_correct = validate_response_structure(solution_str)
    format_score = format_reward if format_correct else -abs(format_reward * 10)

    # Validate answer content
    think_score = 0
    answer_score = 0
    
    if format_correct and answer_text:
        
        # A方案：二元匹配(不验证思维链)
        # print(f"  Predicted: {answer_text}")
        # answer_score = get_single_eval(ground_truth, answer_text)

        # B方案：思维链质量评估
        # print(f"Predicted Think: {think_text}")
        # print(f"Predicted Answer: {answer_text}")

        think_semantic, think_len, think_ppl, think_score = get_think_scores(think_label,think_text)

        # 不优化思维链
        # think_semantic, think_len, think_ppl, think_score = (0, 0, 0, 0)
        
        answer_score = get_answer_score(answer_label, answer_text)

            
    else:
        answer_score = -1.0 / 0.15

    if answer_score < 0:
        think_score = 0
    total_score = format_score + answer_score + 0.5 * think_score

    if format_correct and (think_score > -1) and (total_score > 0):
        return total_score
    

    print("\n" + "="*80)
    print(" Processing New Sample ".center(80, '='))
    print(f"[Solution String]")
    print(f"Predict: \n{solution_str}")

    print(f"\n[Format validation] {'PASS' if format_correct else 'FAIL'}")
    if format_correct and answer_text:
        print(f"\n[Content Validation]")
        print(f"Expected: \n{ground_truth}")

        print(f"[Think] Semantic: {think_semantic}, CoT Length: {think_len}, Perplexity: {think_ppl}")
        print(f"[Answer] Expect: {answer_label}, Predict: {answer_text}, Score: {answer_score}")
    else:
        print("\n[Content Validation] Skipped due to format errors or missing answer")

    print("\n" + "-"*80)
    print(f" Final Score ".center(80, '-'))
    print(f"Format: {format_score}")
    print(f"Think: {think_score}")
    print(f"Answer: {answer_score}")
    print(f"Total: {total_score}")
    print("="*80 + "\n")

    return total_score