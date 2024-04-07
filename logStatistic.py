# -*- coding: utf-8 -*-
import argparse
import json
import glob
import re
import psutil
import os
import threading
import time
import numpy as np
from decimal import Decimal, ROUND_HALF_UP
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics.pairwise import cosine_similarity
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

def memory_watcher(max_memory, check_interval):
    """
    Monitor the memory usage of the current process in a separate thread.
    :param max_memory: Maximum memory in MB.
    :param check_interval: Time interval in seconds between memory checks.
    """
    process = psutil.Process()
    start_time = time.time()
    while True:
        current_time = time.time()
        elapsed_time = current_time - start_time
        memory_usage = process.memory_info().rss / (1024 * 1024)
        if memory_usage > max_memory:
            print(f"Memory limit exceeded: {memory_usage:.2f} MB used; limit is {max_memory} MB.")
            os._exit(1)
        if elapsed_time > 30:
            # print(f"Run {elapsed_time: .0f}s, Current memory usage: {memory_usage:.2f} MB.")
            pass
        time.sleep(check_interval)

def start_memory_watcher(max_memory, check_interval=1):
    """
    Start the memory watcher in a daemon thread.
    :param max_memory: Maximum memory in MB.
    :param check_interval: Time interval in seconds between memory checks.
    """
    watcher_thread = threading.Thread(target=memory_watcher, args=(max_memory, check_interval), daemon=True)
    watcher_thread.start()

def extract_bracket_contents(log_line, left='[', right=']'):
    parts = []
    stack = []
    temp = ""
    for char in log_line:
        if char == left:
            if stack:
                temp += char
            stack.append(char)
        elif char == right:
            if len(stack) == 1:
                parts.append(temp)
                temp = ""
            else:
                temp += char
            stack.pop()
        else:
            if stack:
                temp += char
    # 如果栈不为空，说明有未闭合的括号
    if stack:
        raise ValueError("Unmatched brackets in the string")
    return parts

def parse_timestamp(log):
    # 2024-04-01 20:54:53.415 或 2024-04-02T20:46:23.415+0800
    for fmt in ('%Y-%m-%d %H:%M:%S.%f', '%Y-%m-%dT%H:%M:%S.%f%z'):
        try:
            return datetime.strptime(log, fmt)
        except ValueError:
            continue
    # 如果都不匹配，返回一个特定的默认时间戳
    return datetime(1970, 1, 1)

def parse_significant_digits(log, significant_digits):
    # 四舍五入到指定有效位
    pattern = re.compile(r'([-+]?\d*\.\d+|[-+]?\d+)')
    match = pattern.search(log)
    if match:
        number_str = match.group(0)
        number_decimal = Decimal(number_str)
        # 移动小数点使得只有有效位数的数字在小数点前
        shift = number_decimal.adjusted() - significant_digits + 1
        # 四舍五入到有效位数
        rounded_number = number_decimal.scaleb(-shift).quantize(Decimal('1'), rounding=ROUND_HALF_UP).scaleb(shift)
        start, end = match.span(0)
        log = log[:start] + str(rounded_number) + log[end:]
    return log

def parse_decimal_places(log, decimal_places):
    # 四舍五入到指定小数位
    pattern = re.compile(r'([-+]?\d*\.\d+|[-+]?\d+)')
    match = pattern.search(log)
    if match:
        number_str = match.group(0)
        rounded_number = Decimal(number_str).quantize(Decimal('1.' + '0' * decimal_places), rounding=ROUND_HALF_UP)
        start, end = match.span(0)
        log = log[:start] + str(rounded_number) + log[end:]
    return log

# 预处理函数
def preprocess_log(log_line, field_mapping_def):
    log_parts = {}
    if field_mapping_def['seperator'] == 'none':
        # WHOLE 定义，整个逻辑行是一个字段
        log_parts = {'MESSAGE': log_line.strip()}
    elif field_mapping_def['seperator'] == 'brackets':
        if not log_line.startswith('[') or not log_line.strip().endswith(']'):
            parts = []
        else:
            parts = extract_bracket_contents(log_line)
            # parts = re.findall(r'\[(?:[^\[\]]|\[[^\[\]]*\])*\]', log_line)
            # parts = re.findall(r'\[([^[\]]+(?:\[[^\]]*\])?[^[\]]*)\]', log_line)
        for key, index in field_mapping_def['mapping'].items():
            try:
                log_parts[key] = parts[index - 1]
            except IndexError:
                log_parts[key] = "null"
    elif field_mapping_def['seperator'] == 'json':
        try:
            log_json = json.loads(log_line)
            for key, field in field_mapping_def['mapping'].items():
                log_parts[key] = log_json.get(field, "null")
        except json.JSONDecodeError:
            for key in field_mapping_def['mapping'].keys():
                log_parts[key] = "null"
    else:
        parts = log_line.strip().split(field_mapping_def['seperator'])
        for key, index in field_mapping_def['mapping'].items():
            try:
                log_parts[key] = parts[index - 1]
            except IndexError:
                log_parts[key] = "null"
    return log_parts

# 聚类函数
def group_log(logs, merge_def):
    groups = []
    group_info = defaultdict(list)
    group_indexes = defaultdict(list)
    if not logs:
        return groups, group_info, group_indexes
    if merge_def == 'DBSCAN':
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(logs)
        dbscan = DBSCAN(eps=0.5, min_samples=5)
        dbscan.fit(X)
        groups = dbscan.labels_
        # 为DBSCAN生成簇信息和索引
        for idx, label in enumerate(groups):
            group_indexes[label].append(idx)
            if label != -1:
                group_info[label].append(logs[idx])
        for label, log_list in group_info.items():
            if label != -1:
                group_info[label] = log_list[:1]  # top1
    elif merge_def.startswith('K_MEANS'):
        num_groups = int(merge_def.split('-')[1])
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(logs)
        num_groups = min(num_groups, X.shape[0])
        kmeans = KMeans(n_clusters=num_groups, random_state=42)
        kmeans.fit(X)
        groups = kmeans.labels_
        # 为K_MEANS生成簇信息和索引
        for i in range(num_groups):
            group_indexes[i] = list(np.where(groups == i)[0])
            centroid_vector = kmeans.cluster_centers_[i]
            log_vectors = X[groups == i]
            if log_vectors.shape[0] > 0:
                similarities = cosine_similarity(centroid_vector.reshape(1, -1), log_vectors)
                top_index = similarities.argsort()[0][-1:]  # top1
                for index in top_index:
                    original_log_index = group_indexes[i][index]
                    group_info[i].append(logs[original_log_index])
    elif merge_def.startswith('TIME'):
        time_interval = int(merge_def.split('-')[1])
        timestamps = [parse_timestamp(log) for log in logs]
        if timestamps:
            min_time = min(timestamps)
            min_time_rounded = min_time.replace(minute=0, second=0, microsecond=0)
            groups = [int((timestamp - min_time_rounded).total_seconds() // time_interval) for timestamp in timestamps]
            # 为TIME生成簇信息和索引
            for idx, group_id in enumerate(groups):
                group_indexes[group_id].append(idx)
                start_time = min_time_rounded + timedelta(seconds=group_id * time_interval)
                end_time = start_time + timedelta(seconds=time_interval)
                group_info[group_id] = {
                    'start': start_time.strftime('%Y-%m-%d %H:%M:%S.%f'),
                    'end': end_time.strftime('%Y-%m-%d %H:%M:%S.%f')
                }
    elif merge_def.startswith('ROUND_DECIMAL'):
        decimal_places = int(merge_def.split('-')[1])
        values = [parse_decimal_places(log, decimal_places) for log in logs]
        if values:
            unique_logs = list(set(values))
            groups = [unique_logs.index(log) for log in values]
            # 为ROUND_DECIMAL生成簇信息和索引
            for idx, log in enumerate(values):
                group_id = unique_logs.index(log)
                group_indexes[group_id].append(idx)
            group_info = {i: log for i, log in enumerate(unique_logs)}
    elif merge_def.startswith('ROUND_SIGNIFICANT'):
        decimal_places = int(merge_def.split('-')[1])
        values = [parse_significant_digits(log, decimal_places) for log in logs]
        if values:
            unique_logs = list(set(values))
            groups = [unique_logs.index(log) for log in values]
            # 为ROUND_SIGNIFICANT生成簇信息和索引
            for idx, log in enumerate(values):
                group_id = unique_logs.index(log)
                group_indexes[group_id].append(idx)
            group_info = {i: log for i, log in enumerate(unique_logs)}
    elif merge_def == 'EQUAL':
        unique_logs = list(set(logs))
        groups = [unique_logs.index(log) for log in logs]
        # 为EQUAL生成簇信息和索引
        for idx, log in enumerate(logs):
            group_id = unique_logs.index(log)
            group_indexes[group_id].append(idx)
        group_info = {i: log for i, log in enumerate(unique_logs)}
    else:
        raise ValueError(f"Unsupported merge definition: {merge_def}")
    # 返回聚类结果和每个簇的信息和索引
    return groups, group_info, group_indexes

# 主处理函数
def process_log(input_config, output_config, define):
    if output_config['merge_file']:
        preprocessed_logs = []
        for file_key in input_config['file_filter'].keys():
            process_input(preprocessed_logs, input_config, define, file_key)
        process_output(preprocessed_logs, output_config, define, "MERGED")
    else:
        for file_key in input_config['file_filter'].keys():
            preprocessed_logs = []
            process_input(preprocessed_logs, input_config, define, file_key)
            process_output(preprocessed_logs, output_config, define, file_key)

def process_input(preprocessed_logs, input_config, define, file_key, use_multithreading=False):
    file_paths = []
    for path in input_config['file_filter'][file_key]:
        matched_files = glob.glob(path)
        matched_files = [f for f in matched_files if not f.endswith('.gz')]
        matched_files = [f for f in matched_files if not f.endswith('.tgz')]
        if matched_files:
            file_paths.extend(matched_files)
            print(f"Include: {matched_files}")
        else:
            print(f"Warning: No files found for pattern {path}")

    field_mapping_def = define["field_mapping_def"][input_config['field_mapping']]

    # 内部函数，用于处理单个文件
    def process_single_file(log_file, field_mapping_def, line_filter_pattern, field_filters):
        preprocessed_data = []
        with open(log_file, 'r', encoding='utf-8') as file:
            file_content = file.read()
            # 使用正则表达式匹配逻辑行
            logical_lines = re.findall(line_filter_pattern, file_content, re.MULTILINE | re.DOTALL)
            for log in logical_lines:
                log_data = preprocess_log(log, field_mapping_def)
                if all(re.search(f['match'], log_data.get(f['field'], '')) for f in field_filters):
                    preprocessed_data.append(log_data)
            print(f"file:{log_file}, logical_lines:{len(logical_lines)}, matched_lines:{len(preprocessed_data)}")
        return preprocessed_data

    line_filter_pattern = input_config.get('line_filter', r'^(.*?)(?=\n|$)')  #默认行为类似:readline
    field_filters = input_config.get('field_filters', [])  #默认[], 不过滤field

    if use_multithreading:
        # 使用多线程处理日志文件
        with ThreadPoolExecutor() as executor:
            future_to_file_data = {executor.submit(process_single_file, file_path, field_mapping_def, line_filter_pattern, field_filters): file_path for file_path in file_paths}
            for future in as_completed(future_to_file_data):
                file_path = future_to_file_data[future]
                try:
                    data = future.result()
                    preprocessed_logs.extend(data)
                except Exception as exc:
                    print(f'File {file_path} generated an exception: {exc}')
    else:
        # 不使用多线程，直接顺序处理
        for file_path in file_paths:
            try:
                data = process_single_file(file_path, field_mapping_def, line_filter_pattern, field_filters)
                preprocessed_logs.extend(data)
            except Exception as exc:
                print(f'File {file_path} generated an exception: {exc}')

def process_output(preprocessed_logs, output_config, define, file_key, use_multithreading=False):
    print(f"Processing {file_key}, total lines {len(preprocessed_logs)} ...")

    def process_toplist_item(toplist_item):
        field = toplist_item['field']
        count = toplist_item['count']
        merge = toplist_item['merge']

        field_logs = [log[field] for log in preprocessed_logs]
        return group_log(field_logs, merge), (file_key, count, merge, field)

    def print_results(results, file_key, count, merge, field):
        groups, group_info, group_indexes = results
        group_counter = Counter(groups)
        total_count = sum(group_counter.values())
        top_groups = group_counter.most_common(count)

        print(f'{file_key} Top {count} by {merge} for field {field}:')
        for group, group_count in top_groups:
            percentage = (group_count / total_count) * 100
            print(f'Count: {group_count}\t| Percentage: {percentage:.2f}%\t| Content: {group_info.get(group, "N/A")}'.encode('utf-8').decode('utf-8'))
            # 如果 output_config['print_line_num'] 有定义，则打印每个组的前几行日志
            if 'print_line_num' in output_config:
                print_line_num = min(output_config['print_line_num'], len(group_indexes[group]))
                for i in group_indexes[group][:print_line_num]:
                    if 'print_line_field' in output_config:
                        log_str = " ".join(
                            f"{print_field}: {preprocessed_logs[i].get(print_field, 'N/A')}"
                            for print_field in output_config['print_line_field']
                        )
                        print("\t", log_str)
                    else:
                        print("\t", preprocessed_logs[i])
        print('-' * 80)

    if use_multithreading:
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(process_toplist_item, toplist_item) for toplist_item in output_config['toplist']]
            for future in as_completed(futures):
                results, info = future.result()
                print_results(results, *info)
    else:
        for toplist_item in output_config['toplist']:
            results, info = process_toplist_item(toplist_item)
            print_results(results, *info)

if __name__ == "__main__":
    # 设置命令行参数解析
    parser = argparse.ArgumentParser(description='Log Statistics Processor')
    parser.add_argument('--conf', default='config.cluster.json', help='Path to the configuration file')
    parser.add_argument('--mem_limit', default=10240, type=int, help='Memory limit in megabytes')
    args = parser.parse_args()

    # 设置内存限制(MB)
    if args.mem_limit:
        start_memory_watcher(args.mem_limit)

    # 读取配置文件路径
    config_file_path = args.conf

    # 读取配置文件
    try:
        with open(config_file_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file '{config_file_path}' not found.")
        exit(1)

    # 读取定义文件
    try:
        with open('define.json', 'r', encoding='utf-8') as f:
            define = json.load(f)
    except FileNotFoundError:
        print("Error: Definition file 'define.json' not found.")
        exit(1)

    # 运行主处理函数
    process_log(config['input'], config['output'], define)