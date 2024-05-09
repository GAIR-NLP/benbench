# Benchmarking Benchmark Leakage in LLMs

[Ruijie Xu*](https://plms.ai/people/index.html), [Zengzhi Wang*](https://tinyurl.com/zengzhi-homepage), [Run-Ze Fan*](https://RZFan525.github.io), [Pengfei Liu](https://plms.ai/people/index.html)

[**Homepage**](https://gair-nlp.github.io/benbench/) |
[**HF Demo**](https://huggingface.co/spaces/GAIR/BenBench) | 
[**Paper**](https://arxiv.org/abs/2404.18824) | 
[**Citation**](https://github.com/GAIR-NLP/benbench?tab=readme-ov-file#citation)

## 🔥 News
- [2024/05/08]: We updated the Qwen1.5 series (1.8B, 7B, 14B) results on our [Homepage Leaderboard](https://gair-nlp.github.io/benbench/).

## Table of contents

- [Benchmarking Benchmark Leakage in LLMs](#benchmarking-benchmark-leakage-in-llms)
  - [🚀Introduction](#introduction)
  - [🏆Leaderboard](#leaderboard)
  - [📚Examples](#examples)
  - [Detection Pipeline](#detection-pipeline)
  - [🌴Usage](#usage)
  - [🥳Citation](#citation)

## 🚀Introduction

BenBench is designed to **ben**chmark the potential for data leakage in **bench**mark datasets, which can lead to biased and inequitable comparisons.
In this work, we are not pursuing technical contributions in system development; instead, we are attempting to encourage the healthy development of this field, particularly through the lens of mathematical reasoning tasks, in the following aspects:
1. Summaries of various pre-training behaviors and challenges for detecting benchmark leakage; 
2. Proposal of a detection pipeline for estimating pre-training behaviors; 
3. Leakage analysis of existing models; 
4. Recommendation for model documentation (i.e., introducing Benchmark Transparency Card), benchmark setup and future evaluations.

[//]: # (Amid the expanding use of pre-training data, the phenomenon of benchmark dataset leakage has become increasingly prominent, exacerbated by opaque training processes and the often undisclosed inclusion of supervised data in contemporary Large Language Models &#40;LLMs&#41;. This issue skews benchmark effectiveness and fosters potentially unfair comparisons, impeding the field's healthy development.  Given that training data and model details are often opaque, and the leakage detection is influenced by various factors such as mode size and training strategies, detecting benchmark leakage is not a trivial task. In this work, we are not pursuing technical contributions in system development; instead, we are attempting to encourage the healthy development of this field, particularly through the lens of *mathematical reasoning* tasks, in the following aspects: &#40;1&#41; Summaries of various pre-training behaviors and challenges for detecting benchmark leakage; &#40;2&#41; Proposal of a detection pipeline for estimating pre-training behaviors; &#40;3&#41; Leakage analysis of existing models; &#40;4&#41; Recommendation for model documentation &#40;i.e., introducing Benchmark Transparency Card&#41;, benchmark setup and future evaluations. )


Refer to our paper for more details.

## 🏆Leaderboard

The relative possibility that various models conduct verbatim training to enhance their capabilities on the training set of a benchmark over the test set to enhance capabilities (measured based on PPL and N-gram Accuracy). 

Blue indicates the measured method is based on PPL, while pink indicates the measured method is 5-gram Accuracy.

Models exhibiting near-zero possibilities suggest either the absence of training and test split or the use of both splits in the training process.

<figure >
  <img src="static/images/benbench-leaderboard.png"  alt="img21"/>
  <figcaption>
  </figcaption>           
</figure>

## 📚Examples

[//]: # (## 📊Instance-level Leakage Detection)
[//]: # (<span id="instance-level"></span>)

<img src="static/images/ngram_demo.gif"  alt="img21"/>



High prediction accuracy for each n-gram of an example's prediction suggests a high probability that the sample was encountered during the training process. To investigate instance-level leakage, we looked closer at n-gram predictions across different models. Additionally, considering that benchmark data may undergo reformatting, paraphrasing, or other modifications when integrated into model training, we leverage lenient metrics, such as ROUGE-L and edit distance similarity, for comparing n-grams. Under this context, an instance is deemed correctly predicted if it achieves an Exact Match (meaning all predictions align perfectly), or if the edit distance similarity of all predictions exceeds 0.9 (indicating substantial similarity), and further, if the ROUGE-L score of all predictions surpasses 0.75.


<figure >
  <img src="static/images/instance-level-leakage.png"  alt="img21"/>
  <figcaption>
    <p>Statistics of suspicious leaked sample, where all 5-grams within a sample are predicted correctly, either strictly (measured by Exact Match) or loosely (measured by ROUGE-L). The y-axis employs an exponential scale based on powers of 3.</p>
  </figcaption>           
</figure>


We can observe that many models can all ngrams of an example from benchmark training set even test set. 





<figure >
  <img src="static/images/case_study.png"  alt="img21"/>
  <figcaption>
    <p>Two cases: one from the GSM8K training set predicted by the Qwen-1.8B model (above), and one from the GSM8K test set by the Aquila2-34B model (below). Both examples are presented with the original question and answer concatenated, separated by a space.</p>
  </figcaption>           
</figure>



In the first case, the Qwen-1.8B model achieves perfect n-gram predictions on a sample from the GSM8K training set, completing all 5-grams accurately. This strongly suggests potential data leakage within the training set of GSM8K. Additionally, we also conducted a case study on the Aquila2-34B model, known to accidentally be exposed to the entire GSM8K test set. It consistently predicts n-grams as  "The answer is" for all instances where the ground truth was represented by a placeholder "####". This observation exactly explains why it is challenging to detect  leakage using our n-gram accuracy metric. To enhance readers' comprehension of model behaviors, we have released an interactive demo for case studies, available at <a href="https://huggingface.co/spaces/GAIR/BenBench">Huggingface Space: BenBench</a>.

## Detection Pipeline

We introduce a simple, computationally efficient, and scalable pipeline that leverages two fundamental yet insightful atomic metrics: *Perplexity* and *N-gram Accuracy*. These metrics effectively encapsulate the essence of language modeling, capturing its nuances from continuous and discrete perspectives, respectively. By paraphrasing benchmarks to create varied reference versions, we can detect discrepancies in models' atomic metrics, thereby identifying potential data leakage. This pipeline's validity is supported by thorough meta-experiments.


<figure >
<img src="static/images/detection-pipeline.png"  alt="img21"/>
<figcaption>
<center><p>Overview of detection pipeline</p></center>
</figcaption>            
</figure>


## 🌴Usage

Install dependencies

```
pip install -r requirements.txt
```


Run the pipeline (please specify the ckpt path before running the pipeline)

```
cd src
bash ppl.sh
bash ngram_acc.sh
```


## 🥳Citation

If you find our work useful, please cite our paper:

```
@article{xu2024benchmarking,
      title={Benchmarking Benchmark Leakage in Large Language Models}, 
      author={Xu, Ruijie and Wang, Zengzhi and Fan, Run-Ze and Liu, Pengfei},
      year={2024},
      journal={arXiv preprint arXiv:2404.18824},
      url={https://arxiv.org/abs/2404.18824}
}
```
