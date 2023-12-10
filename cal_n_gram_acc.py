import torch
from tqdm import tqdm
import numpy as np
import json
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
import multiprocessing
from functools import partial
import argparse



def load_data_from_jsonl(jsonl_file_name):

    with open(jsonl_file_name, "r") as f:
        data = [json.loads(line) for line in f.readlines()]
    ds = {"question": [], "answer": []}
    for item in data:
        ds['question'].append(item['question'])
        ds['answer'].append(item['answer'])
    return ds

# def check_model_name(model):
#     if "chatglm3" in model.config._name_or_path:
#         return "chatglm3"
#     elif "chatglm2" in  model.config._name_or_path:
#         return "chatglm2"
#     elif "chatglm" in model.config._name_or_path:
#         return "chatglm"

def calculate_n_gram_accuracy(n, k, dataset, model, tokenizer, device):
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
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token else '[PAD]'

    tokenizer.padding_side = 'left'
    model_name = check_model_name(model)

    accuracies = []  # 存储每个样本的精度

    tokenized_samples = []
    for question, answer in zip(dataset['question'], dataset['answer']):
        format_text = f"{question} {answer}"
        tokens = tokenizer.tokenize(format_text)
        tokenized_samples.append(tokens)

    for idx in tqdm(range(0, len(dataset['question']))):
        # question, answer = dataset['question'][idx], dataset['answer'][idx]
        # format_text = f"{question} {answer}"
        # tokens = tokenizer.tokenize(format_text)
        tokens = tokenized_samples[idx]
        len_tokens = len(tokens)

        if len_tokens - n - 1 <= 0:
            continue
            
        sample_correct_n_grams = 0
        sample_total_n_grams = 0
        starting_points = np.linspace(2, len_tokens - n, num=k, endpoint=True, dtype=int)
        starting_points = torch.tensor(starting_points)

        for start_index in starting_points:
            prefix_tokens = tokens[:start_index]
            encoding = tokenizer(
                prefix_tokens,
                is_split_into_words = True,
                return_tensors = "pt",
                padding="longest"
            ).to(device)
            encoding['max_new_tokens'] = n
            
            encoding['do_sample'] = False
            gens = model.generate(**encoding)
            # print(gens)
            # print(tokenizer.batch_decode(gens))

            predicted_ids = gens[0, -n:].tolist()
            
            original_ids = tokenizer.convert_tokens_to_ids(tokens[start_index: start_index + n])
            if 5 in predicted_ids:
                if 5 in original_ids:
                    print("ok")
                else:
                    print(predicted_ids, original_ids)
            if original_ids == predicted_ids:
                sample_correct_n_grams += 1
                sample_total_n_grams += 1
                print("Pass!")

        if sample_total_n_grams > 0:
            sample_accuracy = sample_correct_n_grams / sample_total_n_grams
            accuracies.append(sample_accuracy)
    
    # 计算所有样本精度的平均值
    return {"n_grams'": accuracies, "mean_n_grams": np.mean(accuracies)} if accuracies else 0


def split_dataset(dataset, num_splits):
    # 计算每个分割应该包含的元素数量
    split_size = len(dataset['question']) // num_splits

    # 分割数据集
    dataset_splits = []
    for i in range(num_splits):
        start = i * split_size
        # 对于最后一个分割，确保包含所有剩余的元素
        end = None if i == num_splits - 1 else start + split_size
        subset = {'question': dataset['question'][start:end], 'answer': dataset['answer'][start:end]}
        dataset_splits.append(subset)

    return dataset_splits

def process_subset(n, k, subset, model, tokenizer, device):
    return calculate_n_gram_accuracy(n, k, subset, model, tokenizer, device)

def parallel_process(n, k, dataset, model, tokenizer, device, num_processes):
    dataset_split =  split_dataset(dataset, num_processes)

    func = partial(process_subset, n, k, model=model, tokenizer=tokenizer, device=device)

    with multiprocessing.Pool(num_processes) as pool:
        results = pool.map(func, dataset_split)

    return results

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Script to process dataset with a model.")

    # Adding arguments
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model file.')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID to use (default: 0).')
    parser.add_argument('--n', type=int, required=True, help='Size of the n-gram.')
    parser.add_argument('--k', type=int, required=True, help='Number of starting points for each sample.')
    parser.add_argument('--multi_processes', action="store_true", help='using multiple processing.')
    parser.add_argument('--num_processes', type=int, help='Number of starting points for each sample.')

    args = parser.parse_args()


    multiprocessing.set_start_method('spawn', force=True)
    
    dataset_test = load_data_from_jsonl("gsm8k_test.jsonl")

    model = AutoModel.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    device = f"cuda:{args.gpu_id}"
    model = model.to(device)
    model.eval()
    if args.multi_processes:
        results = parallel_process(args.n, args.k, dataset_test, model, tokenizer, device, args.num_processes)
        print("test_n_gram_accuracy: ", results["mean_n_grams"])
    else:

        test_n_gram_accuracy = calculate_n_gram_accuracy(args.n, args.k, dataset_test, model, tokenizer, device)
        print("test_n_gram_accuracy: ", test_n_gram_accuracy["mean_n_grams"])
