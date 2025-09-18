import json
import csv
from collections import defaultdict


# 步骤1: 加载标注修正数据并创建映射字典
def load_rectification_mappings(patch_path):
    with open(patch_path, 'r') as f:
        patches = json.load(f)

    # 创建日志模板到修正标签的映射
    # 用defaultdict确保每个模板只存储最后的修正值（避免重复键）
    mapping = {}
    for item in patches:
        template = item["input"].strip()
        rectified = item["rectified"]
        mapping[template] = rectified
    return mapping


# 步骤2: 处理日志数据文件并替换标签
def rectify_labels(log_data_path, output_path, mapping):
    # 读取数据并处理
    updated_rows = []
    template_count = 0

    with open(log_data_path, 'r') as f:
        reader = csv.reader(f)
        headers = next(reader)  # 读取标题行
        updated_rows.append(headers)

        for row in reader:
            # 解析各字段（跳过首列索引）
            _, label, timestamp, event_template = row[:4]
            cleaned_template = event_template.strip()

            # 检查是否需要修正
            if cleaned_template in mapping:
                template_count += 1
                new_label = mapping[cleaned_template]
                # 转换标签格式
                row[1] = 'RECTIFIED' if new_label == 'abnormal' else '-'

            updated_rows.append(row)

    # 写入修正后的数据
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(updated_rows)

    return template_count


# 主执行流程
if __name__ == "__main__":
    # 配置文件路径
    patch_path = "anomaly_patch.json"  # 标注修正文件路径
    log_data_path = "./trainsets/bgl_structured.csv"  # 原始日志文件路径
    output_path = "./trainsets/rectified_bgl_structured.csv"  # 输出文件路径

    # 1. 加载修正映射
    mapping = load_rectification_mappings(patch_path)
    print(f"Loaded {len(mapping)} rectification mappings")

    # 2. 处理日志文件
    matched_count = rectify_labels(log_data_path, output_path, mapping)
    print(f"Rectified {matched_count} log entries")
    print(f"Output saved to: {output_path}")

    # 配置文件路径
    log_data_path = "./trainsets/spirit_structured.csv"  # 原始日志文件路径
    output_path = "./trainsets/rectified_spirit_structured.csv"  # 输出文件路径

    # 1. 加载修正映射
    mapping = load_rectification_mappings(patch_path)
    print(f"Loaded {len(mapping)} rectification mappings")

    # 2. 处理日志文件
    matched_count = rectify_labels(log_data_path, output_path, mapping)
    print(f"Rectified {matched_count} log entries")
    print(f"Output saved to: {output_path}")

