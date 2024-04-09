import json
import numpy as np
import torch
from tqdm import tqdm
import editdistance
from rouge_score import rouge_scorer

def calculate_similarity_metrics(n, k, dataset, model, tokenizer, device, output_file, edit_threshold=0.1, rouge_l_threshold=0.75):
    """
    Calculate similarity metrics and determine n-gram accuracy based on thresholds for edit distance and Rouge-L.
    :param n: Size of the n-gram to predict.
    :param k: Number of starting points to use for each sample.
    :param datasets: Dataset containing questions and answers.
    :param model: Pre-trained language model.
    :param tokenizer: Tokenizer corresponding to the language model.
    :param device: Device to run the model on.
    :param edit_threshold: Threshold for normalized edit distance (e.g., 0.1 for 10%).
    :param rouge_l_threshold: Threshold for Rouge-L score.
    :return: n-gram accuracy based on edit distance and Rouge-L.
    """
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    accuracies_edit_distance = []  # Store each sample's accuracy
    accuracies_rouge_l = []  # Store each sample's accuracy
    detailed_results = []  # Store detailed comparison results
    accuracies = []

    # 设置填充令牌和填充方向
    if ("deepseek" in output_file) or ("llama" in output_file) or ("GPT" in output_file) or ("phi" in output_file) or ("Baichuan-7B" in output_file) or ("Aquila-7B" in output_file) or ("Mistral" in output_file):
        if tokenizer.pad_token == None:
            if tokenizer.eos_token:
                tokenizer.pad_token = tokenizer.eos_token
                print("set pad done")
            else:
                print("no special token")

    tokenizer.padding_side = 'left'
    if ("Aquila" in output_file) or ("phi" in output_file):
        tokenizer.add_prefix_space = True

    tokenized_samples = []

    for question, answer in zip(dataset['question'], dataset['answer']):
        if ("hellaswag" in output_file) or ("Truthful" in output_file) or ("MMLU" in output_file):
            format_text = f"{question}{answer}"
        else:
            format_text = f"{question} {answer}"
        tokens = tokenizer.tokenize(format_text)
        tokenized_samples.append(tokens)

    for idx in tqdm(range(0, len(dataset['question']))):
        tokens = tokenized_samples[idx]
        len_tokens = len(tokens)

        if len_tokens - n - 1 <= 0:
            continue

        sample = tokenizer.convert_tokens_to_string(tokens)
        sample_results = {"idx": idx, "sample": sample, "n_gram_results": [], "overall": {}}  # Store detailed results for this sample
        sample_correct_n_grams_edit = 0
        sample_correct_n_grams_rouge = 0
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
            
            if ("Mistral" in output_file) or ("Abel-7B-002" in output_file) or ("deepseek" in output_file):
                gens = model.generate(**encoding, pad_token_id=tokenizer.eos_token_id)
            else:
                gens = model.generate(**encoding)

            predicted_ids = gens[0, -n:].tolist()
            original_ids = tokenizer.convert_tokens_to_ids(tokens[start_index: start_index + n])

            predicted_text = tokenizer.decode(predicted_ids, skip_special_tokens=True)
            original_text = tokenizer.decode(original_ids, skip_special_tokens=True)
    
            # Calculate normalized edit distance
            edit_dist = editdistance.eval(predicted_text, original_text)
            max_length = max(len(predicted_text), len(original_text))
            edit_similarity = 1 - (edit_dist / max_length)
            
            # Calculate Rouge-L score
            rouge_score = scorer.score(original_text, predicted_text)['rougeL'].fmeasure

            # Record detailed results
            n_gram_result = {
                "start_index": int(start_index),
                "predicted_text": predicted_text,
                "original_text": original_text,
                "edit_similarity": edit_similarity,
                "rouge_score": rouge_score
            }
            sample_results["n_gram_results"].append(n_gram_result)
            
            # Increment correct n-gram count if conditions are met
            sample_total_n_grams += 1
            if edit_similarity >= (1 - edit_threshold):
                sample_correct_n_grams_edit += 1
            if rouge_score >= rouge_l_threshold:
                sample_correct_n_grams_rouge += 1
            if original_ids == predicted_ids:
                sample_correct_n_grams += 1
                
        # Add to accuracies
        if sample_total_n_grams > 0:
            edit_distance = sample_correct_n_grams_edit / sample_total_n_grams
            accuracies_edit_distance.append(edit_distance)
            rouge_l = sample_correct_n_grams_rouge / sample_total_n_grams
            accuracies_rouge_l.append(rouge_l)
            sample_accuracy = sample_correct_n_grams / sample_total_n_grams
            accuracies.append(sample_accuracy)
            overall = {"accuracies_edit_distance": edit_distance, "accuracies_rouge_l": rouge_l, "n_grams": sample_accuracy}
            sample_results["overall"] = overall

        # Add this sample's results to the overall detailed results
        detailed_results.append(sample_results)
    
    # Calculate mean accuracies
    mean_accuracy_edit_distance = np.mean(accuracies_edit_distance) if accuracies_edit_distance else 0
    mean_accuracy_rouge_l = np.mean(accuracies_rouge_l) if accuracies_rouge_l else 0
    mean_accuracy = np.mean(accuracies) if accuracies else 0

    # Output detailed results to a JSON file
    # output_file = "detailed_similarity_results.json"
    with open(output_file, 'w') as f:
        json.dump(detailed_results, f, indent=4)
        
    return {
        "n_grams_accuracy_edit_distance": accuracies_edit_distance,
        "mean_accuracy_edit_distance": mean_accuracy_edit_distance,
        "n_grams_accuracy_rouge_l": accuracies_rouge_l,
        "mean_accuracy_rouge_l": mean_accuracy_rouge_l,
        "mean_accuracy": mean_accuracy,
        "detailed_results_file": output_file
    }

# Example usage with hypothetical dataset, model, tokenizer, and device
# dataset = ... (Load your dataset)
# model = ... (Load your model)
# tokenizer = ... (Load your tokenizer)
# device = ... (Specify your device)
# n = 5  # Example n-gram size
# k = 5  # Example number of starting points
# edit_threshold = 0.1  # 10% threshold for edit distance
# rouge_l_threshold = 0.75  # Threshold for Rouge-L
# results = calculate_similarity_metrics(n, k, dataset, model, tokenizer, device, edit_threshold, rouge_l_threshold)
# print(f"Mean Accuracy Edit Distance: {results['mean_accuracy_edit_distance']}")
# print(f"Mean Accuracy Rouge-L: {results['mean_accuracy_rouge_l']}")
