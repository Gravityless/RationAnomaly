import json
import re
import string
import pandas as pd
import json
# 提取 answer
from prettytable import PrettyTable
from sklearn.metrics import classification_report, confusion_matrix
import argparse

# 1. 创建 ArgumentParser 对象
parser = argparse.ArgumentParser()

# 2. 添加参数
# add_argument() 方法用于添加参数
parser.add_argument('--model', '-m', type=int, required=True)
parser.add_argument('--testset', '-t', type=int, required=True)
parser.add_argument('--session', '-s', type=str, default=' ', required=False)

# 3. 解析参数
args = parser.parse_args()

import pickle
import os
from sklearn.metrics import classification_report

def standardize_template(template):
    # 1. 去除前后空格
    template = template.strip()
    # 2. 统一为小写（若你的数据大小写不统一）
    template = template.lower()
    # 3. 去除多余空格（比如多个空格合并为一个）
    template = re.sub(r'\s+', ' ', template)
    # 4. 若有特殊字符（如括号、引号），按需去除或保留
    # template = re.sub(r'[()"]', '', template)
    return template

    
def load_sessions(data_dir):
    with open(os.path.join(data_dir, "session_train.pkl"), "rb") as fr:
        session_train = pickle.load(fr)
    with open(os.path.join(data_dir, "session_test.pkl"), "rb") as fr:
        session_test = pickle.load(fr)

    # train_labels = [
    #     v["label"] if not isinstance(v["label"], list) else int(sum(v["label"]) > 0)
    #     for _, v in session_train.items()
    # ]
    # test_labels = [
    #     v["label"] if not isinstance(v["label"], list) else int(sum(v["label"]) > 0)
    #     for _, v in session_test.items()
    # ]

    return session_train, session_test
def get_ses_eval(template2pred,domain):
    for key,val in template2pred.items():
        if val=='abnormal':
            template2pred[key]=1
        if val=='normal':
            template2pred[key]=0
        if val=='unknown':
            template2pred[key]=0
    if domain=='bgl':
        session_train, session_test = load_sessions(data_dir='../datasets/bgl_0_tar')
    else:
        session_train, session_test = load_sessions(data_dir='../datasets/spirit_0_tar')        
    session_test_filtered=[]
    for _,dic in session_test.items():
        filtered=False
        dic['label_pred']=[]
        for i,log in enumerate(dic['templates']):
            if log in template2pred:
                dic['label_pred'].append(int(template2pred[log]))
            else:
                #不剔除session，而是剔除log
                # dic['label_pred'].append(0)
                # dic['label'][i]=0
                #直接整个session 剔除
                filtered=True
                break
        if not filtered:
            session_test_filtered.append(dic)
    print('session_test',len(session_test),'session_test_filtered',len(session_test_filtered))
    golden=[]
    for dic in session_test_filtered:
        if sum(dic['label'])>0:
            golden.append(1)
        else:
            golden.append(0)
    pred=[]
    for dic in session_test_filtered:
        if sum(dic['label_pred'])>0:
            pred.append(1)
        else:
            pred.append(0)
    #pd.DataFrame(zip(golden,pred),columns=['golden_session','pred_session']).to_excel('%s.xlsx'%name)
    
    from prettytable import PrettyTable
    results = pd.DataFrame(classification_report(golden,pred,target_names=['normal','abnormal'],output_dict=True))
    table = PrettyTable()
    table.field_names = [""] + list(results.columns)
    for idx, row in results.iterrows():
        table.add_row([idx] + [round(i, 4) for i in row.tolist() if isinstance(i, float)])
    print(table)

def get_single_eval(pred,golden):
    # print("####template-level test###############")
    # eval_result=pd.read_excel(file)
    # pred=eval_result['pred_id'].tolist()
    # golden=eval_result['label_id'].tolist()
    print("Unique values in pred:", set(pred))
    print("Unique values in golden:", set(golden))

    # 检查 pred 中是否存在 'unknown'
    if 'unknown' in set(pred):
        print("Detected 'unknown' in predictions, generating 3-class report...")
        target_names = ['abnormal', 'normal', 'unknown']
    else:
        print("No 'unknown' detected, generating 2-class report...")
        target_names = ['abnormal', 'normal']

    results = pd.DataFrame(classification_report(golden, pred, target_names=target_names, output_dict=True))
    print(results)
    cm = confusion_matrix(golden, pred, labels=['normal', 'abnormal', 'unknown'])
    cm_df = pd.DataFrame(cm, index=['True normal', 'True abnormal', 'True unknown'],
                        columns=['Pred normal', 'Pred abnormal', 'Pred unknown'])
    
    table = PrettyTable()
    table.field_names = [""] + list(results.columns)
    for idx, row in results.iterrows():
        table.add_row([idx] + [round(i, 5) for i in row.tolist() if isinstance(i, float)])
    print(table)
    print(cm_df)
def extract_answer(solution_str: str, tags_to_try: list = ['answer', 'em', 'b']) -> str:
    """
    从模型的响应字符串中提取被指定标签包裹的最终答案。
    
    函数会按照 `tags_to_try` 列表中的顺序依次尝试查找标签。
    一旦找到任何一个标签的匹配项，就会立即返回最后一个匹配的内容。

    Args:
        solution_str: 语言模型的原始响应字符串。
        tags_to_try: 一个包含要查找的 HTML 标签名的列表，按优先级排序。
        
    Returns:
        提取出的答案字符串。如果所有标签都未找到，则返回原始字符串。
    """
    # 确保输入是字符串
    if not isinstance(solution_str, str):
        return "" # 或者可以返回 None 或 raise a TypeError

    # 按照列表中的优先级顺序，依次尝试每个标签
    for tag in tags_to_try:
        # 动态构建正则表达式，以匹配当前循环的标签
        # 例如，当 tag='answer' 时，pattern 会变成 r'<answer>(.*?)</answer>'
        pattern = fr'<{tag}>(.*?)</{tag}>'
        
        # 使用 re.finditer 查找所有匹配项
        matches = list(re.finditer(pattern, solution_str, re.DOTALL))
        
        # 如果找到了匹配项
        if matches:
            # 提取最后一个匹配项的内容，去除首尾空格后返回
            final_answer = matches[-1].group(1).strip()
            return final_answer
            
    # 如果循环结束后仍未找到任何标签的匹配项，打印错误信息并返回原始字符串
    return solution_str

key_words = ["normal", 'abnormal']
normal_description = ['this log entry is normal', 'it is a normal log entry', 'that it is normal', 'the log entry is normal', "the output would be normal", 'this is a normal log entry', "the output is normal", 'the status is normal', 'the entry is normal', 'the entire log entry is normal', 'the log entry can be considered normal', 'the log entry belongs to the normal category', 'the log entry appears to be normal', 'the answer is normal', 'as a normal log entry', 'it is considered normal','log entry is normal','will say normal']
abnormal_description = ['it is an abnormal log entry', 'that it is abnormal', 'this log entry is abnormal', 'the log entry is abnormal', "the output would be abnormal", 'this is an abnormal log entry', "the output is abnormal", 'the status is abnormal', 'the entry is abnormal', 'the entire log entry is abnormal', 'the log entry can be considered abnormal', 'the log entry belongs to the abnormal category', 'the log entry appears to be abnormal', 'the answer is abnormal','it is considered abnormal','log entry is abnormal']
front_words = ["output", "judgement", "observation", 'classification']
def validate_first_line(text):
    # 提取第一行
    first_line = text.split('\n')[0]
    
    # 检查是否包含关键字
    has_normal = 'normal' in first_line.split()
    has_abnormal = 'abnormal' in first_line.split()
    
    # 根据检查结果返回相应的值
    if has_normal and has_abnormal:
        return 'unknown'
    elif has_normal:
        return 'normal'
    elif has_abnormal:
        return 'abnormal'
    else:
        return 'unknown' 
def find_keyword_with_regex(sentence, key_words):

    keyword_pattern = "|".join(re.escape(k) for k in key_words)

    # `as\s+` 匹配 'as' 后面一个或多个空格
    # `(?:a|an)?\s*` 匹配可选的 'a' 或 'an'，后面跟零个或多个空格
    # `(\b(?:{keyword_pattern})\b)` 捕获关键词，`\b` 确保是完整单词匹配
    # `|` 在正则表达式中表示“或”，因此 `as\s*(\b...)\b` 也能匹配 `asnormal` 这种情况

    pattern = rf"as\s+(?:a|an)?\s*(\b(?:{keyword_pattern})\b)|(\b(?:{keyword_pattern})\b)"
    
    match = re.search(pattern, sentence, re.IGNORECASE)

    if match:
        # match.group(1) 对应 `as ...` 模式的捕获组
        # match.group(2) 对应直接匹配关键词的捕获组
        if match.group(1):
            return match.group(1)
        if match.group(2):
            return match.group(2)

    return None
def check_normal(predict):
    predict = predict.lower()

    # 原始提取方法
    if validate_first_line(extract_answer(predict)) in key_words:
        return validate_first_line(extract_answer(predict))
    
    # 删除标点符号
    predict = predict.replace("\n", " ").replace(":", " ").replace("\'", " ").replace("\"", " ").replace(";", " ").replace(",", " ").replace(".", " ").replace("'", " ").strip()
    predict_token = predict.split(' ')
    refresh_tokens = []
    for token in predict_token:
        if token == '':
            continue
        else:
            refresh_tokens.append(token)
    predict = ' '.join(refresh_tokens) # 去掉多余空格
    if predict in key_words:
        return predict # 直接返回
    elif predict.split(" ")[0] in key_words:
        # label + 解释的情况
        return predict.split(" ")[0]
    elif predict.find("log entry") == 0 and predict.split(" ")[2] in key_words:
        return predict.split(" ")[2]
    else:
        for description in normal_description:
            if description in predict:
                return 'normal'
        for description in abnormal_description:
            if description in predict:
                return 'abnormal'
        if ' as ' in predict:       
            if find_keyword_with_regex(predict,key_words):
                return find_keyword_with_regex(predict,key_words)
        
        refresh_tokens = predict.split(" ")
        for front_word in front_words:
            if front_word in refresh_tokens:
                try:
                    key_word = refresh_tokens[refresh_tokens.index(front_word) + 1]
                except IndexError:
                    # 如果front_word是列表的最后一个元素，则没有下一个元素，设置key_word为None或适当处理
                    key_word = None
                if key_word in key_words:
                    return key_word
        found_keywords = list(set([word for word in refresh_tokens if word in key_words]))
        if len(found_keywords) == 1:
            return found_keywords[0]

        return 'unknown'
        # return 'abnormal'

if __name__ == '__main__':

    # 读取JSON文件
    with open("model_testset_list.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    models = data['models']
    testsets = data['testsets']
    model_paths = data['model_paths']
    testset_paths = data['testset_paths']

    model = models[args.model]
    testset = testsets[args.testset]

    test_data_file = testset_paths[testset]
    infer_result_file = f'../datasets/results/original_vllm_output/{testset}_by_{model}.json'
    labelled_infer_result_file = f'../datasets/results/labelled_output/{testset}_by_{model}.json'
    diff_infer_result_file = f'../datasets/results/diff/{testset}_by_{model}.json'

    with open(infer_result_file, 'r', encoding='utf-8') as f:
        infer_result = json.load(f)
    with open(test_data_file, 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    # 1. 建立字典，减少时间复杂度，通过instruction + "" + input 映射到
    label_dict = {}
    for test_unit in test_data:
        # 修改 LogEval instruction
        # if 'LogEval' in test_data_file:
        #     test_unit["instruction"] = 'Convert the following logs into standardized templates by identifying and replacing the variable parts with a <*>.'
        key = test_unit['instruction'] + test_unit['input']
        output = test_unit['label']
        label_dict[key] = output

    # 2. 检索字典
    bio_checker_num = 0
    none_bio_num = 0
    for infer_unit in infer_result:
        # 给推理结果贴label
        key = infer_unit['instruction'] + infer_unit['input']
        infer_unit['label'] = label_dict[key]
        infer_unit['output'] = infer_unit['output'].strip()
    # 3. 针对Anomaly任务使输出结果二元化
        infer_output = infer_unit['output']
    
        # 直接调用二值化函数，并将结果赋给 infer_unit['output']
        infer_unit['output'] = check_normal(infer_output)
        
        # 你的统计和打印逻辑
        if infer_output.lower() != 'normal' and infer_output.lower() != 'abnormal':
            bio_checker_num += 1 
            if infer_unit['output'] == 'unknown':
                none_bio_num += 1
                print("二元化:\n" + "bio_checker_id: " + str(bio_checker_num) + "\nkey: " + key + "\noutput: " + infer_output + "\nbio_result: " + infer_unit['output'])
    # print('Unknown的值的数量：' + str(none_bio_num))

    # 3. 输出差异
    diff_result = []
    for infer_unit in infer_result:
        if infer_unit['label'] != infer_unit['output']:
            diff_result.append(infer_unit)

    # 3. 写出
    with open(labelled_infer_result_file, 'w', encoding='utf-8') as f:
        json.dump(infer_result, f, ensure_ascii=False, indent=4)

    with open(diff_infer_result_file, 'w', encoding='utf-8') as f:
        json.dump(diff_result, f, ensure_ascii=False, indent=4)

    raw_logs=[ele['instruction'] + ele['input'] for ele in infer_result]

    true_labels=[ele['label'].lower() for ele in infer_result]
    pred_results=[ele['output'].lower() for ele in infer_result]
    get_single_eval(pred_results,true_labels)
    print('Unknown的值的数量：' + str(none_bio_num))

    if args.session != ' ':
        print("============================Session Level==========================")
        test_set=json.load(open(labelled_infer_result_file))
        session_testset = testsets[args.testset]
        test_path = testset_paths[session_testset]
        with open(test_path) as f:
            original_templates=json.load(f)
        raw_logs=[ele['instruction'] + ele['input'] for ele in test_set]
        pred_results=[ele['output'].lower() for ele in test_set]
        template2pred={dic['input']:pred for dic,pred in zip(original_templates,pred_results)}
        if args.session == 'b':
            get_ses_eval(template2pred,domain='bgl')
        else:
            get_ses_eval(template2pred,domain='spirit')
    