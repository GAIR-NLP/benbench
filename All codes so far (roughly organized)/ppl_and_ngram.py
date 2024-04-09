import torch
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
import numpy as np
import json
import random
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, LlamaForCausalLM
import multiprocessing
from functools import partial


def load_model(model_path, device):
    # 加载模型和 tokenizer
    # if "llama" in model_path:
    #     model = LlamaForCausalLM.from_pretrained(model_path, trust_remote_code=True)
    if ("chatglm-6b" in model_path) or ("chatglm3-6b" in model_path):
        model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
    
    if "Qwen" in model_path:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, pad_token='<|endoftext|>')
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    # 设置设备
    model = model.half().to(device)
    model.eval()
    return model, tokenizer
    

def load_data_from_jsonl(jsonl_file_name, num_samples=3000):
    if ("SVAMP" in jsonl_file_name) or ("MMLU" in jsonl_file_name) or ("/MATH/" in jsonl_file_name) or ("MetaMath" in jsonl_file_name):
        with open(jsonl_file_name, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        with open(jsonl_file_name, "r", encoding="utf-8") as f:
            data = [json.loads(line) for line in f.readlines()]

    # 随机抽取 num_samples 个样本，如果样本数少于 num_samples，则全部选入
    random.seed(666)
    selected_samples = random.sample(data, min(num_samples, len(data)))
    print(len(selected_samples))

    ds = {"question": [], "answer": []}
    for item in selected_samples:
        if ("/GSM8K-" in jsonl_file_name):
            ds['question'].append(item['question'])
            ds['answer'].append(item['answer'])
        if ("AQuA" in jsonl_file_name):
            ds['question'].append(item['question'])
            ds['answer'].append(item['rationale'][15:-12])
        if ("numglue" in jsonl_file_name):
            ds['question'].append(item['question'])
            ds['answer'].append(item['answer'])
        if ("SVAMP" in jsonl_file_name):
            ds['question'].append(item['Body'] + ' ' + item['Question'])
            ds['answer'].append(item['Equation'] + ' = ' + str(item['Answer']))
        if ("/MATH-" in jsonl_file_name):
            ds['question'].append(item['problem'])
            ds['answer'].append(item['solution'])
        if ("asdiv-a" in jsonl_file_name):
            ds['question'].append(item['Body'] + ' ' + item['Question'])
            ds['answer'].append(item['Formula'] + ' ' + item['Answer'])
        if ("mawps-single" in jsonl_file_name):
            ds['question'].append(item['sQuestion'])
            ds['answer'].append('Equations: ' + item['lEquations'][0] + ' Solutions: ' + item['lSolutions'][0])
        if ("MMLU" in jsonl_file_name):
            ds['question'].append(item['question'])
            ds['answer'].append('\nAnswer: ' + item['target'])
        if ("hellaswag" in jsonl_file_name):
            ds['question'].append(item['ctx'])
            label_index = item["label"]
            ds['answer'].append(item["endings"][label_index])
        if ("Truthful" in jsonl_file_name):
            ds['question'].append(item['prompt'])
            ds['answer'].append(item['completion'])
        if ("exercise" in jsonl_file_name):
            ds['question'].append(item['question'])
            ds['answer'].append(item['solution'] + ' ' + item['answer'])
        if ("MetaMath" in jsonl_file_name):
            ds['question'].append(item['query'])
            ds['answer'].append(item['response'])
        if ("gsmplus" in jsonl_file_name):
            ds['question'].append(item['question'])
            ds['answer'].append(item['solution'])
        if ("rewritten" in jsonl_file_name):
            ds['question'].append(item["rewritten_question"])
            ds['answer'].append(item["rewritten_answer"])
            
    return ds


def find_subsequence(sequence, subsequence):
    """ 在序列中查找子序列的起始索引。如果找不到，则返回-1。"""
    for i in range(len(sequence)):
        if sequence[i:i+len(subsequence)] == subsequence:
            return i
    print("Not found\n")
    return -1


def calculate_answer_ppl(datasets, model, tokenizer, device, output_file):
    sep_token = "Answer:"
    ppls = []
    samples_with_ppl = []
    loss_fct = CrossEntropyLoss(reduction="none")

    for question, answer in tqdm(zip(datasets['question'], datasets['answer']), total=len(datasets['question'])):
        combined_text = question + ' ' + sep_token + ' ' + answer
        encoding = tokenizer(combined_text, return_tensors="pt", return_attention_mask=True).to(device)
        
        # if ("chatglm2-6b" in model_path) or ("chatglm3-6b" in model_path):
        #     encoding = tokenizer(combined_text, return_tensors="pt", truncation=True, max_length=model.config.seq_length).to(device)
        # if ("chatglm-6b" in output_file):
        #     encoding = tokenizer(combined_text, return_tensors="pt", truncation=True, max_length=model.config.max_sequence_length).to(device)
        # elif ("Baichuan-13B" in model_path) or ("Baichuan2-13B" in model_path):
        #     encoding = tokenizer(combined_text, return_tensors="pt", truncation=True, max_length=model.config.model_max_length).to(device)
        # else:
        #     encoding = tokenizer(combined_text, return_tensors="pt", truncation=True, max_length=model.config.max_length).to(device)

        # 获取所有sep_token子令牌的ID
        if ("chatglm2-6b" in output_file) or ("chatglm3-6b" in output_file) or ("llama" in output_file) or ("Abel" in output_file) or ("Mistral" in output_file) or ("Orca" in output_file) or ("Baichuan-" in output_file):
            sep_token_ids = tokenizer.encode(sep_token, add_special_tokens=False)
        else:
            sep_token_ids = tokenizer.encode(' '+sep_token, add_special_tokens=False)

        # 在编码文本中找到sep_token的位置
        sep_index = find_subsequence(encoding["input_ids"][0].tolist(), sep_token_ids)

        if sep_index != -1:  # 确保找到了sep_token
            encoded_text = encoding["input_ids"]
            if ("chatglm-6b" in output_file):
                attn_mask = encoded_text != tokenizer.pad_token_id
            else:
                attn_mask = encoding["attention_mask"]

            # 创建新的注意力掩码，仅针对答案部分
            answer_attn_mask = torch.zeros_like(attn_mask)
            answer_attn_mask[:, sep_index + len(sep_token_ids):] = attn_mask[:, sep_index + len(sep_token_ids):]

            # 计算答案部分的困惑度
            with torch.no_grad():
                out_logits = model(encoded_text, attention_mask=attn_mask).logits

            shift_logits = out_logits[..., :-1, :].contiguous()
            shift_labels = encoded_text[..., 1:].contiguous()
            shift_attention_mask = answer_attn_mask[..., 1:].contiguous()

            loss = (loss_fct(shift_logits.transpose(1, 2), shift_labels) * shift_attention_mask).sum(1) / shift_attention_mask.sum(1)
            perplexity = torch.exp(loss).mean().item()
            ppls.append(perplexity)

            samples_with_ppl.append({"text": combined_text, "perplexity": perplexity})
            
        if sep_index == -1:
            print(combined_text)
            print("encoded_text: ", encoding["input_ids"])

    with open(output_file, 'w') as file:
        for item in samples_with_ppl:
            file.write(json.dumps(item) + '\n')
            
    return {"perplexities": ppls, "mean_perplexity": np.mean(ppls)}


def calculate_total_ppl(datasets, model, tokenizer, device, output_file):
    ppls = []
    samples_with_ppl = []
    loss_fct = CrossEntropyLoss()

    for question, answer in tqdm(zip(datasets['question'], datasets['answer']), total=len(datasets['question'])):
        combined_text = question + ' ' + answer
        encoding = tokenizer(combined_text, return_tensors="pt").to(device)

        # Note: This assumes that you no longer need to account for model-specific maximum sequence lengths
        # or to handle different tokenization strategies for different models as was indicated in the commented-out portion of your provided code.
        
        encoded_text = encoding["input_ids"]
        attn_mask = encoding["attention_mask"]
        
        # 不再需要单独为答案部分创建一个注意力掩码，因为我们计算整个文本的PPL
        with torch.no_grad():
            out_logits = model(encoded_text, attention_mask=attn_mask).logits

        # Adjusted shift_logits and shift_labels for the entire sequence
        shift_logits = out_logits[..., :-1, :].contiguous()
        shift_labels = encoded_text[..., 1:].contiguous()
        
        # Calculate loss for the entire sequence
        loss = loss_fct(shift_logits.transpose(1, 2), shift_labels)
        loss = loss.mean()
        perplexity = torch.exp(loss).item()
        ppls.append(perplexity)

        samples_with_ppl.append({"text": combined_text, "perplexity": perplexity})

    # Saving the samples and their perplexities to a file
    with open(output_file, 'w') as file:
        for item in samples_with_ppl:
            file.write(json.dumps(item) + '\n')
            
    return {"perplexities": ppls, "mean_perplexity": np.mean(ppls)}


def calculate_n_gram_accuracy(n, k, dataset, model, tokenizer, device, output_file):
    """
    Calculate n-gram accuracy using a language model with batching.
    :param n: Size of the n-gram to predict.
    :param k: Number of starting points to use for each sample.
    :param datasets: Dataset containing questions and answers.
    :param model: Pre-trained language model.
    :param tokenizer: Tokenizer corresponding to the language model.
    :param device: Device to run the model on.
    :param batch_size: Size of each batch.
    :return: n-gram accuracy.
    """
    # 设置填充令牌和填充方向
    # if not tokenizer.pad_token:
    #     if tokenizer.eos_token:
    #         tokenizer.pad_token = tokenizer.eos_token
    #     else:
    #         print("no special token")
    if ("deepseek" in output_file) or ("llama" in output_file) or ("GPT" in output_file) or ("phi" in output_file) or ("Baichuan-7B" in output_file) or ("Aquila-7B" in output_file) or ("Mistral" in output_file):
        if tokenizer.eos_token:
            tokenizer.pad_token = tokenizer.eos_token
            print("set pad done")
        else:
            print("no special token")
            
    if ("GPT" in output_file):
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.unk_token
            print("set GPT pad done")
        

    tokenizer.padding_side = 'left'
    if ("Aquila" in output_file) or ("phi" in output_file):
        tokenizer.add_prefix_space = True

    accuracies = []  # 存储每个样本的精度

    tokenized_samples = []
    
    for question, answer in zip(dataset['question'], dataset['answer']):
        if ("hellaswag" in output_file) or ("Truthful" in output_file) or ("MMLU" in output_file):
            format_text = f"{question}{answer}"
        else:
            format_text = f"{question} {answer}"
        tokens = tokenizer.tokenize(format_text)
        tokenized_samples.append(tokens)

    detailed_results = []

    for idx in tqdm(range(0, len(dataset['question']))):
        tokens = tokenized_samples[idx]
        len_tokens = len(tokens)
        sample = tokenizer.convert_tokens_to_string(tokens)
        sample_results = {"idx": idx, "sample": sample, "n_gram_results": []}

        if len_tokens - n - 1 <= 0:
            continue
            
        sample_correct_n_grams = 0
        sample_total_n_grams = 0
        if ("chatglm2-6b" in output_file) or ("chatglm3-6b" in output_file):
            starting_points = np.linspace(2, min(len_tokens, model.config.seq_length) - n, num=k, endpoint=True, dtype=int)
        elif ("chatglm-6b" in output_file):
            starting_points = np.linspace(2, min(len_tokens, model.config.max_sequence_length) - n, num=k, endpoint=True, dtype=int)
        elif ("Baichuan-13B" in output_file) or ("Baichuan2-13B" in output_file):
            starting_points = np.linspace(2, min(len_tokens, model.config.model_max_length) - n, num=k, endpoint=True, dtype=int)
        else:
            starting_points = np.linspace(2, min(len_tokens, model.config.max_position_embeddings) - n, num=k, endpoint=True, dtype=int)
        starting_points = torch.tensor(starting_points)

        for start_index in starting_points:
            prefix_tokens = tokens[:start_index]
            
            prefix_string = tokenizer.convert_tokens_to_string(prefix_tokens)
            encoding = tokenizer(
                prefix_string,
                is_split_into_words=False,
                return_tensors="pt",
                padding="longest"
                ).to(device)
        
            encoding['max_new_tokens'] = n
            encoding['do_sample'] = False
            
            if ("Mistral" in output_file) or ("Abel-7B-002" in output_file) or ("deepseek" in output_file) or ("phi-2" in output_file):
                gens = model.generate(**encoding, pad_token_id=tokenizer.eos_token_id)
            else:
                gens = model.generate(**encoding)

            predicted_ids = gens[0, -n:].tolist()
            original_ids = tokenizer.convert_tokens_to_ids(tokens[start_index: start_index + n])

            predicted_text = tokenizer.decode(predicted_ids, skip_special_tokens=True)
            original_text = tokenizer.decode(original_ids, skip_special_tokens=True)

            # Record detailed results
            n_gram_result = {
                "start_index": int(start_index),
                "predicted_text": predicted_text,
                "original_text": original_text
            }
            sample_results["n_gram_results"].append(n_gram_result)
            
            sample_total_n_grams += 1
            if original_ids == predicted_ids:
                sample_correct_n_grams += 1
            
        if sample_total_n_grams > 0:
            sample_accuracy = sample_correct_n_grams / sample_total_n_grams
            accuracies.append(sample_accuracy)

        detailed_results.append(sample_results)

    with open(output_file, 'w') as f:
        json.dump(detailed_results, f, indent=4)
        
    # 计算所有样本精度的平均值
    return {"n_grams": accuracies, "mean_n_grams": np.mean(accuracies)} if accuracies else 0


# # 数据集路径
# dataset_paths_ppl = [
#                  "/data/rjxu/MATH/test.json",
#                  "/data/rjxu/MATH/train.json",
#                  "/data/rjxu/GSM8K/gsm8k_test.jsonl",
#                  "/data/rjxu/GSM8K/gsm8k_train.jsonl"
#                 ]

# # 初始化一个字典来存储结果
# results_ppl_summary = {
#     'GSM8K-train': None,
#     'GSM8K-test': None,
#     'MATH-train': None,
#     'MATH-test': None
# }

# # 加载模型和 tokenizer
# if "llama" in model_path:
#     model = LlamaForCausalLM.from_pretrained(model_path, trust_remote_code=True)
# elif ("chatglm-6b" in model_path) or ("chatglm3-6b" in model_path):
#     model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
# else:
#     model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)

# if "Qwen" in model_path:
#     tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, pad_token='<|endoftext|>')
# else:
#     tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# # 设置设备
# model = model.half().to(device)
# model.eval()

# # 遍历每个数据集并执行测试
# for dataset_path in dataset_paths_ppl:
#     # 加载数据集
#     dataset = load_data_from_jsonl(dataset_path)

#     # 提取数据集名称和类型（train 或 test）
#     if "MATH" in dataset_path:
#         dataset_name = "MATH"
#     elif "GSM8K" in dataset_path:
#         dataset_name = "GSM8K"
#     else:
#         dataset_name = "SVAMP"
        
#     if "train" in dataset_path:
#         dataset_type = "train"
#     elif "info" in dataset_path:
#         dataset_type = "info"
#     elif "truth" in dataset_path:
#         dataset_type = "truth"
#     elif "val" in dataset_path:
#         dataset_type = "val"
#     else:
#         dataset_type = "test"
    
#     # 执行计算
#     results = calculate_answer_ppl(dataset, model, tokenizer, device)
#     print(f"{dataset_name}-{dataset_type} Average_ppl_accuracy: ", results["mean_perplexity"])
        
#     # 存储结果到字典中
#     results_ppl_summary[f'{dataset_name}-{dataset_type}'] = results["mean_perplexity"]

# # 打印结果
# for key, value in results_ppl_summary.items():
#     print(f"{key}: {value}")

# # 输出 Markdown 格式表格
# print("| GSM8K-train | GSM8K-test | MATH-train | MATH-test |")
# print("|-------------|------------|------------|-----------|")

# print(f"| {results_ppl_summary['GSM8K-train']:.2f} | {results_ppl_summary['GSM8K-test']:.2f} | "
#       f"{results_ppl_summary['MATH-train']:.2f} | {results_ppl_summary['MATH-test']:.2f} |")

# # 设置你的参数
# # n = 5  # 你的 n-gram 大小
# k = 5  # 你的 starting points 数量
# multi_processes = False  # 是否使用多进程
# num_processes = 4  # 如果使用多进程，设置进程数量

# n_values = [5, 10]

# for n in n_values:
#     # 数据集路径
#     dataset_paths_ngram = [
#                      "/data/rjxu/MMLU-data/dev(train)-new.json",
#                      "/data/rjxu/MMLU-data/test-new.json"
#                     ]
    
#     # 初始化一个字典来存储结果
#     results_ngrm_summary = {
#         'MMLU-train': None,
#         'MMLU-test': None
#     }
    
#     # 遍历每个数据集并执行测试
#     for dataset_path in dataset_paths_ngram:
#         # 加载数据集
#         dataset = load_data_from_jsonl(dataset_path)
    
#         # 提取数据集名称和类型（train 或 test）
#         if "AQuA" in dataset_path:
#             dataset_name = "AQuA"
#         elif "numglue" in dataset_path:
#             dataset_name = "numglue"
#         elif "MMLU" in dataset_path:
#             dataset_name = "MMLU"
#         elif "hellaswag" in dataset_path:
#             dataset_name = "hellaswag"
#         elif "Truthful" in dataset_path:
#             dataset_name = "TruthfulQA"
#         elif "MATH" in dataset_path:
#             dataset_name = "MATH"
#         elif "GSM8K" in dataset_path:
#             dataset_name = "GSM8K"
#         else:
#             dataset_name = "SVAMP"
            
#         if "train" in dataset_path:
#             dataset_type = "train"
#         elif "info" in dataset_path:
#             dataset_type = "info"
#         elif "truth" in dataset_path:
#             dataset_type = "truth"
#         elif "val" in dataset_path:
#             dataset_type = "val"
#         else:
#             dataset_type = "test"
        
#         results = calculate_n_gram_accuracy(n, k, dataset, model, tokenizer, device)
#         print(f"{dataset_name}-{dataset_type} {n}_gram_accuracy: ", results["mean_n_grams"])
            
#         # 存储结果到字典中
#         results_ngrm_summary[f'{dataset_name}-{dataset_type}'] = results["mean_n_grams"]
    
#     # 打印结果
#     for key, value in results_ngrm_summary.items():
#         print(f"{key}: {value}")
    
#     # 输出 Markdown 格式表格
#     print(f"{n}_gram_accuracy")
#     print("| MMLU-train | MMLU-test |")
#     print("|------------|-----------|")
    
#     print(f"| {results_ngrm_summary['MMLU-train'] * 100:.2f} | {results_ngrm_summary['MMLU-test'] * 100:.2f} |")