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
    parser.add_argument('--model_type', type=str, default = "base", help='model type: base or chat')
    parser.add_argument('--device', type=str, required=True, help='device')
    parser.add_argument('--n', type=int, required=True, help='n-gram', default=5)
    args = parser.parse_args()


    model, tokenizer = load_model(args.model_path, args.device)
    if args.dataset_name == "gsm8k":
        dataset_names = gsm8k_dataset_names
    elif args.dataset_name == "math":
        dataset_names = math_dataset_names
    else:
        raise ValueError("Invalid dataset")


    k = 5  # num of starting point.
    results_ngrm_summary = {}

    for dataset_name in dataset_names:
        if "rewritten" in dataset_name:
            dataset_path = f'./data/rewritten/{dataset_name}.jsonl'
        elif "orgn" in dataset_name:
            dataset_path = f'./data/original/{dataset_name}.jsonl'


        dataset = load_data_from_jsonl(dataset_path)
        
        output_file_ngram = f'./outputs/ngram/{args.n}gram-{args.model_name}-{dataset_name}.jsonl'
        ngram_results = calculate_n_gram_accuracy(args.n, k, dataset, model, tokenizer, args.device, output_file_ngram, args.model_type)
        print(f"{dataset_name} {args.n}_gram_accuracy: ", ngram_results["mean_n_grams"])
        results_ngrm_summary[f'{dataset_name}'] = ngram_results["mean_n_grams"]
        
    print(f"ngram acc of {args.model_name}")
    for key, value in results_ngrm_summary.items():
        print(f"{key}: {value}")