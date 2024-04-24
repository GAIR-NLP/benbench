# Benchmarking Benchmark Leakage in Large Langauge Models

This is the official repository for [Benchmarking Benchmark Leakage in Large Langauge Models]()

[**Homepage**](https://gair-nlp.github.io/benbench/) |
[**HF Demo**](https://huggingface.co/spaces/GAIR/BenBench) | 
[**Datasets**](https://huggingface.co/datasets/GAIR/) | 
[**Paper**](https://huggingface.co/papers/) | 
[**Citation**](https://github.com/GAIR-NLP/benbench?tab=readme-ov-file#citation)







## 🚀Introduction

Amid the expanding use of pre-training data, the phenomenon of benchmark dataset leakage has become increasingly prominent, exacerbated by opaque training processes and the often undisclosed inclusion of supervised data in contemporary Large Language Models (LLMs). This issue skews benchmark effectiveness and fosters potentially unfair comparisons, impeding the field's healthy development.  Given that training data and model details are often opaque, and the leakage detection is influenced by various factors such as mode size and training strategies, detecting benchmark leakage is not a trivial task. In this work, we are not pursuing technical contributions in system development; instead, we are attempting to encourage the healthy development of this field, particularly through the lens of *mathematical reasoning* tasks, in the following aspects:



- **Summaries of various pre-training behaviors and challenges for detecting benchmark leakage.**

- **Proposal of a detection pipeline for estimating pre-training behaviors**: We introduce a simple, computationally efficient, and scalable pipeline that leverages two fundamental yet insightful atomic metrics: *Perplexity* and *N-gram Accuracy*. These metrics effectively encapsulate the essence of language modeling, capturing its nuances from continuous and discrete perspectives, respectively. By paraphrasing benchmarks to create varied reference versions, we can detect discrepancies in models' atomic metrics, thereby identifying potential data leakage. This pipeline's validity is supported by thorough meta-experiments.


<figure >
<img src="static/images/detection-pipeline.png"  alt="img21"/>
<figcaption>
<center><p>Overview of detection pipeline</p></center>
</figcaption>            
</figure>

- **Leakage analysis of existing models**: We extend our investigation to analyze existing models (i.e., 31 open-source LLMs), revealing that, in addition to previously identified leaks, many (i.e., approximately half of them), including well-known language models, may have inadvertently leveraged training data to boost their performance on mathematical reasoning tasks, leading to unfair advantages. Moreover, our metric even enables instance-level detection, revealing the possibility of test set leaks in many models. For example, we found that Qwen-1.8B can accurately predict all 5-grams in 223 examples from the GSM8K training set and 67 from the MATH training set, with an additional 25 correct predictions even in the MATH test set.

- **Recommendation for model documentation, benchmark setup and future evaluations**: Based on these findings, we offer suggestions encompassing model documentation, benchmark construction, public access to benchmarks, and evaluation from multiple perspectives. We particularly emphasize the aspect of model documentation; we recommend that models should be accompanied by a document at release, which registers whether benchmark data was used for specific performance enhancement and whether any data augmentation was conducted. To this end, we introduce the *Benchmark Transparency Card* to facilitate this process, hoping that it will be widely adopted to promote transparency and healthy development of LLMs.



## 🏆Leaderboard

<figure >
  <img src="static/images/benbench-leaderboard.png"  alt="img21"/>
  <figcaption>
    <center><p>The relative possibility that various models conduct verbatim training on the training set of a benchmark over test set to enhance capabilities (measured based on PPL and N-gram Accuracy). Models exhibiting near-zero possibilities suggest either the absence of training and test split or the use of both splits in the training process.</p></center>
  </figcaption>           
</figure>


## 📊 N-gram Accuracy Helps Instance-level Leakage Detection



High prediction accuracy for each n-gram of an example's prediction suggests a high probability that the sample was encountered during the training process. To investigate instance-level leakage, we looked closer at n-gram predictions across different models. Additionally, considering that benchmark data may undergo reformatting, paraphrasing, or other modifications when integrated into model training, we leverage lenient metrics, such as ROUGE-L and edit distance similarity, for comparing n-grams. Under this context, an instance is deemed correctly predicted if it achieves an Exact Match (meaning all predictions align perfectly), or if the edit distance similarity of all predictions exceeds 0.9 (indicating substantial similarity), and further, if the ROUGE-L score of all predictions surpasses 0.75.


<figure >
  <img src="static/images/instance-level-leakage.png"  alt="img21"/>
  <figcaption>
    <p>Statistics of suspicious leaked sample, where all 5-grams within a sample are predicted correctly, either strictly (measured by Exact Match) or loosely (measured by ROUGE-L). The y-axis employs an exponential scale based on powers of 3.</p>
  </figcaption>           
</figure>


We can observe that many models can all ngrams of an example from benchmark training set even test set. Surprisingly, Qwen-1.8B can  accurately predict all 5-grams in 223 examples from the GSM8K training set and 67 from the MATH training set, with an additional 25 correct predictions even in the MATH test set. We would like to emphasize that the n-gram accuracy metric can mitigate issues in our detection pipeline, particularly when the training and test datasets are simultaneously leaked and remain undetected. However, this also has its limitations; it can only detect examples that are integrated into the model training in their original format and wording, unless we know the organizational format of the training data used by the model in advance.



## 📚 Case Study

<figure >
  <img src="static/images/case_study.png"  alt="img21"/>
  <figcaption>
    <p>Two cases: one from the GSM8K training set predicted by the Qwen-1.8B model (above), and one from the GSM8K test set by the Aquila2-34B model (below). Both examples are presented with the original question and answer concatenated, separated by a space.</p>
  </figcaption>           
</figure>



In the first case, the Qwen-1.8B model achieves perfect n-gram predictions on a sample from the GSM8K training set, completing all 5-grams accurately. This strongly suggests potential data leakage within the training set of GSM8K. Additionally, we also conducted a case study on the Aquila2-34B model, known to accidentally be exposed to the entire GSM8K test set. It consistently predicts n-grams as  "The answer is" for all instances where the ground truth was represented by a placeholder "####". This observation exactly explains why it is challenging to detect  leakage using our n-gram accuracy metric. To enhance readers' comprehension of model behaviors, we have released an interactive demo for case studies, available at <a href="https://huggingface.co/spaces/GAIR/BenBench">Huggingface Space: BenBench</a>.




## 📃 Recommendation for Model Documentation and Benchmarks Setup

To ensure fairness in the evaluation of large language models moving forward, we propose the following suggestions:

- **Documentation**: For any LLMs to be released, comprehensive documentation should be provided. This documentation at least specifies **whether the model has been trained on the training or test sets of commonly used benchmarks to prevent potentially unfair comparisons**. To this end, we introduce Benchmark Transparency Card, which serves as the supplement of the Data Card and Model Card, aiming to document the utilization of benchmarks (such as whether any benchmark sets are used for training and whether any data augmentation techniques are applied) and benchmark evaluation details. We hope that this card will be widely adopted upon the release of models to foster the healthy development of large language models.
- **Benchmark Construction**: We recommend constructing benchmarks from the latest corpus to minimize the risk of overlap with pre-training corpora. Additionally, evaluation datasets should be regularly updated using dynamic benchmarks to guard against overfitting to static test datasets. 
- **Benchmark Public Access**: To mitigate the risk of *Input-Output Leakage*, we advise against directly uploading original benchmarks online, particularly when they contain paired questions and answers. As suggested by Jacovi et al., 2023, encrypting the test set prior to uploading can enhance security. Alternatively, maintaining a private test set through a leaderboard format is also a viable option. 
- **Evaluation**: We recommend caution in drawing overly optimistic conclusions about a model's capabilities based on its strong performance in specific benchmarks. It may be beneficial to evaluate the model further using a variety of contemporary challenges, such as new exam questions, to provide a more balanced assessment of its abilities. When benchmarking proprietary models, it is important to proceed with caution, especially when submitting benchmark data through APIs. There is a risk that this data could be utilized by the model's provider for further training purposes.



## 🌴How to evaluate a model using our pipeline

Install dependencies

```
pip install -r requirements.txt
```


Run the pipeline (Please specify ckpt path before running the pipeline)

```
cd src
bash ppl.sh
bash ngram_acc.sh
```


## 🥳Citation

If you find our work useful or use MathPile, please cite our paper:

```


```