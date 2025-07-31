# Deep QA (DQA): An Open-Domain Dataset of Deep Questions and Comprehensive Answers

![License](https://img.shields.io/badge/license-MIT-blue.svg)

## 📌 Overview

**Deep QA (DQA)** is the first open-domain QA dataset designed specifically for *deep, analytical questions* with *multi-sentence, comprehensive answers*. Unlike traditional QA datasets that focus on brief factual answers, DQA introduces questions classified by higher-order thinking as per Bloom’s Taxonomy.

> 📊 Total Questions: 12,816  
> 📚 Domains: Computer Science (Wikipedia + Textbooks)  
> 🧠 Question Types: Explain Why, Justify with Example, Describe, What Is, etc.

---

## 🔍 Motivation

Most QA datasets fall short in two dimensions:

1. **Question Depth** – Fact-based, simple recall tasks dominate.
2. **Answer Length** – Answers are short, often a word or phrase.

**DQA bridges this gap** by:
- Requiring cognitive processing at understanding, application, evaluation, and meta-cognitive levels.
- Generating rich, context-heavy, paragraph-level answers.

---

## 📦 Dataset Structure

Each entry in the dataset includes:

- `question`: A deep, analytical question.
- `answer`: Multi-sentence explanatory answer.
- `context`: Extended paragraph(s) from which question and answer are derived.

**Example:**
```json
{
  "question": "Explain why the working-set model optimizes CPU utilization.",
  "answer": "The working-set model ensures that only the actively used memory pages are retained in RAM...",
  "context": "Memory management is critical in OS design. The working-set model identifies a set of pages..."
}
```

---

## 🧠 Question Generation Techniques

DQA uses a hybrid generation approach:

- **Textbook-Based Heuristics**:  
  - "Explain why", "Justify with example" questions are extracted from textbook paragraphs using rule-based prefix transformations and coreference resolution via AllenNLP.

- **Wikipedia + SRL-Based**:
  - Questions like "Describe", "What is", and "List the uses of" are generated via Semantic Role Labeling (SRL) using the Proposition Bank framework.

---

## 📊 Dataset Statistics

| Metric                     | DQA            | SQuAD          |
|---------------------------|----------------|----------------|
| Avg. words per answer     | 121.86         | 3.48           |
| Avg. entities per context | 228.84         | 42.97          |
| Avg. Flesch score (Q)     | 50.56          | 70.15          |
| ARI (Q)                   | 12.42          | 7.32           |

---

## 📉 Baseline Model Evaluation

| Model       | BLEU (DQA fine-tuned) | BLEU (Original) |
|-------------|------------------------|-----------------|
| BERT        | 0.40                   | 0.36            |
| BART        | 0.38                   | 0.34            |
| T5          | 0.35                   | 0.31            |

---

## 🔗 Graph Neural Approach

A preliminary **Graph Attention Network (GAT)** model was implemented to address inter-sentence dependencies.

| Metric            | Score    |
|-------------------|----------|
| ROC-AUC (Weighted)| 74.97    |
| Precision         | 89.47    |
| F1 Score          | 77.43    |
| Cohen’s Kappa     | 24.20    |

---

## 📂 Access the Dataset

📥 [Zenodo DOI](https://doi.org/10.5281/zenodo.7538113)

---

## 🚀 Future Work

- Expand to include more comprehension question types.
- Improve graph-based answer generation.
- Explore cross-domain applications in education and reasoning.

---

## 📄 Citation

If you use this dataset in your work, please cite:

Link to our paper: https://link.springer.com/chapter/10.1007/978-3-031-35299-7_16

```bibtex
@inproceedings{anbarasu2023deepqa,
  title={Deep QA: An Open-Domain Dataset of Deep Questions and Comprehensive Answers},
  author={Anbarasu, Hariharasudan Savithri and Navalli, Harshavardhan Veeranna and Vidapanakal, Harshita and Gowd, K Manish and Das, Bhaskarjyoti},
  booktitle={International Conference on Computer and Communication Engineering},
  year={2023}
}
```

---

## 🛠 Authors

- Hariharasudan Savithri Anbarasu  
- Harshavardhan Veeranna Navalli  
- Harshita Vidapanakal  
- K Manish Gowd  
- Bhaskarjyoti Das  

---

## 📬 Contact

For questions or feedback, feel free to reach out via [email](mailto:hari30082001@gmail.com).
