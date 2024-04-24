from ppl_and_ngram_utils import *
import argparse

gsm8k_dataset_names = [
    "GSM8K_rewritten-test-1",
    "GSM8K_rewritten-test-2",
    "GSM8K_rewritten-test-3",
    "GSM8K_rewritten-train-1",
    "GSM8K_rewritten-train-2",
    "GSM8K_rewritten-train-3",
    "orgn-GSM8K-test",
    "orgn-GSM8K-train",
    ]
math_dataset_names = [
    "MATH_rewritten-test-1",
    "MATH_rewritten-test-2",
    "MATH_rewritten-test-3",
    "MATH_rewritten-train-1",
    "MATH_rewritten-train-2",
    "MATH_rewritten-train-3",
    "orgn-MATH-train",
    "orgn-MATH-test",
]



if __name__ == "__main__":

    parser = argparse.ArgumentParser('Benchmark Leakage Detection based on PPL', add_help=False)
    parser.add_argument('--dataset_name', type=str, required=True, help='path to config file')
    parser.add_argument('--model_path', type=str, required=True, help='path to model')
    parser.add_argument('--model_name', type=str, required=True, help='model name')
    parser.add_argument('--device', type=str, required=True, help='device')
    args = parser.parse_args()


    model, tokenizer = load_model(args.model_path, args.device)
    if args.dataset_name == "gsm8k":
        dataset_names = gsm8k_dataset_names
    elif args.dataset_name == "math":
        dataset_names = math_dataset_names
    else:
        raise ValueError("Invalid dataset")


    results_ppl_summary = {}

    for dataset_name in dataset_names:
        if "rewritten" in dataset_name:
            dataset_path = f'./data/rewritten/{dataset_name}.jsonl'
        elif "orgn" in dataset_name:
            dataset_path = f'./data/original/{dataset_name}.jsonl'


        dataset = load_data_from_jsonl(dataset_path)
        
        output_file_ppl = f'./outputs/ppl/ppl-{args.model_name}-{dataset_name}.jsonl'
        ppl_results = calculate_answer_ppl(dataset, model, tokenizer, args.device, output_file_ppl)
        print(f"{dataset_name} Average_ppl_accuracy: ", ppl_results["mean_perplexity"])
        results_ppl_summary[f'{dataset_name}'] = ppl_results["mean_perplexity"]
        
    print(f"PPL of {args.model_name}")
    for key, value in results_ppl_summary.items():
        print(f"{key}: {value}")