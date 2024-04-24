import editdistance
from rouge_score import rouge_scorer
import json
import os

scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

def read_json_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        print("File Not Fount!")
    except json.JSONDecodeError:
        print("Json File Parsing Error!")
    except Exception as e:
        print("Exception: ", e)

def exact_match_score(predicted_text, original_text):
    return int(predicted_text == original_text)

def edit_similarity_score(predicted_text, original_text):
    # Calculate normalized edit distance
    edit_dist = editdistance.eval(predicted_text, original_text)
    max_length = max(len(predicted_text), len(original_text))
    edit_similarity = 1 - (edit_dist / max_length)
    return edit_similarity

def rouge_l_score(predicted_text, original_text):
    # Calculate Rouge-L score
    rouge_score = scorer.score(original_text, predicted_text)['rougeL'].fmeasure
    return rouge_score

def reformat_data(data):
    for item in data:
        ngrams_preds = item['n_gram_results']
        valid_exact_match, valid_edit_similarity, valid_rouge_score = 0, 0, 0
        for ngram in ngrams_preds:
            if "edit_similarity" not in ngram:
                ngram["edit_similarity"] = edit_similarity_score(ngram['predicted_text'], ngram['original_text'])
            if 'rouge_score' not in ngram:
                ngram['rouge_score'] = rouge_l_score(ngram['predicted_text'], ngram['original_text'])
            if "exact_match_score" not in ngram:
                ngram['exact_match_score'] = exact_match_score(ngram['predicted_text'], ngram['original_text'])
            valid_exact_match += ngram['exact_match_score']
            valid_edit_similarity += int(ngram['edit_similarity'] > 0.75)
            valid_rouge_score += int(ngram['rouge_score'] > 0.75)
        item['overall'] = {}
        item['overall']['exact_match_correct_ratio'] = valid_exact_match / len(ngrams_preds)
        item['overall']['edit_similarity_correct_ratio'] = valid_edit_similarity / len(ngrams_preds)
        item['overall']['rouge_score_correct_ratio'] = valid_rouge_score / len(ngrams_preds)

    return data

if __name__ == "__main__":

    original_dir = "./outputs/ngram/"
    all_filenames = os.listdir(original_dir)
    # sort filenames
    all_filenames.sort()
    for filename in all_filenames:
        if "grok" not in filename.lower() :
            continue
        if "GSM8K" not in filename:
            continue
        path = os.path.join(original_dir, filename)
        data = read_json_file(path)
        data = reformat_data(data)
        correct, total = 0, 0
        for item in data:
            for ngram in item['n_gram_results']:
                correct += ngram['exact_match_score']
                total += 1
        print(f"{filename}: {(correct/total)*100}")