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

    random.seed(666)
    selected_samples = random.sample(data, min(num_samples, len(data)))
    print(len(selected_samples))

    ds = {"question": [], "answer": []}

    for item in selected_samples:
        if ("rewritten" in jsonl_file_name):
            ds['question'].append(item["rewritten_question"])
            ds['answer'].append(item["rewritten_answer"])
        if ("orgn" in jsonl_file_name) and ("GSM8K" in jsonl_file_name):
            ds['question'].append(item["question"])
            ds['answer'].append(item["answer"])
        if ("orgn" in jsonl_file_name) and ("MATH" in jsonl_file_name):
            # print(jsonl_file_name)
            ds['question'].append(item["problem"])
            ds['answer'].append(item["solution"])
            
    return ds


def find_subsequence(sequence, subsequence):
    """ find subsequence, return -1 if find nothing"""
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
        encoding = tokenizer(combined_text, return_tensors="pt").to(device)
    

        if ("chatglm2-6b" in output_file) or ("chatglm3-6b" in output_file) or ("llama" in output_file and "llama-3" not in output_file) or ("Abel" in output_file) or ("Mistral" in output_file) or ("Orca" in output_file) or ("loss" in output_file) or ("grok" in output_file):
            sep_token_ids = tokenizer.encode(sep_token, add_special_tokens=False)
        else:
            sep_token_ids = tokenizer.encode(' '+sep_token, add_special_tokens=False)

        sep_index = find_subsequence(encoding["input_ids"][0].tolist(), sep_token_ids)

        if sep_index != -1:  
            encoded_text = encoding["input_ids"]
            attn_mask = encoding["attention_mask"]

            answer_attn_mask = torch.zeros_like(attn_mask)
            answer_attn_mask[:, sep_index + len(sep_token_ids):] = attn_mask[:, sep_index + len(sep_token_ids):]

            try:
                with torch.no_grad():
                    out_logits = model(encoded_text, attention_mask=attn_mask).logits

                shift_logits = out_logits[..., :-1, :].contiguous()
                shift_labels = encoded_text[..., 1:].contiguous()
                shift_attention_mask = answer_attn_mask[..., 1:].contiguous()

                loss = (loss_fct(shift_logits.transpose(1, 2), shift_labels) * shift_attention_mask).sum(1) / shift_attention_mask.sum(1)
                perplexity = torch.exp(loss).mean().item()
                ppls.append(perplexity)
            except torch.cuda.OutOfMemoryError as e:
                print("Error calculating perplexity: ", e)
                continue

            samples_with_ppl.append({"text": combined_text, "perplexity": perplexity})
            
        if sep_index == -1:
            print(combined_text)
            print("encoded_text: ", encoding["input_ids"])
            exit

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



def prepare_prompt_for_chat_model(prefix_str, tokenizer):
    messages = [
        {"role": "system", "content": "You are a helpful assistant! Please directly continue my content without extra content such as '...'."},
        {"role": "user", "content": prefix_str}
    ]
    prompt = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
    )
    return prompt


def calculate_n_gram_accuracy(n, k, dataset, model, tokenizer, device, output_file, model_type = "base"):
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
    # if not tokenizer.pad_token:
    #     if tokenizer.eos_token:
    #         tokenizer.pad_token = tokenizer.eos_token
    #     else:
    #         print("no special token")
    if ("deepseek" in output_file) or ("llama" in output_file) or ("GPT" in output_file) or ("phi" in output_file) or ("Baichuan-7B" in output_file) or ("Aquila-7B" in output_file) or ("Mistral" in output_file) or ("loss" in output_file):
        if not tokenizer.pad_token:
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

    accuracies = []  # 

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
            prompt = tokenizer.convert_tokens_to_string(prefix_tokens)
            if model_type == "chat":
                prompt = tokenizer.build_inputs_with_special_tokens(prompt)
            encoding = tokenizer(
                prompt,
                is_split_into_words=False,
                return_tensors="pt",
                padding="longest"
                ).to(device)
        
            encoding['max_new_tokens'] = n
            encoding['do_sample'] = False
            
            if ("Mistral" in output_file) or ("Abel-7B-002" in output_file) or ("deepseek" in output_file) or ("phi-2" in output_file) or ("loss" in output_file) or ("llama-3" in output_file):
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
        
    return {"n_grams": accuracies, "mean_n_grams": np.mean(accuracies)} if accuracies else 0