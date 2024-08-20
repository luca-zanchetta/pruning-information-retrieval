# A Pruning-Based Deep Learning Approach for Information Retrieval

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1xgEKauqx6sjTNUg5zu-YFrHpH-HFHN8T)
[![arXiv](https://img.shields.io/badge/arXiv-1234.56789-b31b1b.svg)]()

This repo introduces a refined approach to information retrieval by enhancing an existing inverted
index framework through pruning and fine-tuning techniques within the realm of deep learning.
Our work focuses on optimizing an established model, denoted as f, which takes a query q as input
and produces a ranked list of document IDs. By leveraging deep learning methodologies, we have
meticulously pruned and fine-tuned the existing framework to better capture semantic relationships
between queries and documents, thereby improving retrieval accuracy and efficiency.

Extensive experimentation on the proposed MS MARCO dataset validates the efficacy
of our refined approach, showcasing notable enhancements in retrieval performance compared to
the unmodified baseline. We have analyzed the effects of pruning and fine-tuning methods in order
to understand how they improve our model. This helps us see exactly how we can make the model
work better within the inverted index system.

---

## Dataset

The MS MARCO (Microsoft MAchine Reading Comprehension) is a collection of datasets focused on deep learning
in search. The first dataset was a question answering dataset featuring 100,000 real Bing questions and a human
generated answer. Since then, different datasets were released. We accessed the MS MARCO dataset through the
Pyserini toolkit, the Python interface to Anserini, designed to support reproducible IR research. Specifically, we chose
the 0.12.0 release, adapting the BM25 baseline for the MS MARCO passage ranking task to our needs. The queries
are already distributed in Pyserini. We initialized a searcher with a pre-built index, which Pyserini will automatically
download, in order to search for a specific query with associated ranked document IDs. So, we selected the most
relevant K document IDs for each query of the dataset, as well as the first 1000 document IDs by relevance for the
Recall@1000 metric computation, used only for this purpose and not for training. Since we used the Hugging Face T5
tokenizer, we tokenized our dataset accordingly by first extracting the maximum sequence lengths for encoder and
decoder and then by applying the T5 tokenizer with the proper padding to maximum length. Finally, the dataset was
splitted into 70% training set, 10 % validation set and 20% test set.

---

## Metrics

We assessed our model using the following metrics:
- **MAP (Mean Average Precision)**, used in IR to evaluate the effectiveness of search systems. To implement it,
we started from Precision@K, a measure of relevancy calculated as the number of relevant retrieved document
IDs divided by the total number of retrieved document IDs, so K. Hence, we defined the Average Precision
(AP) as the average of precision values calculated at the points in the ranking where each relevant document is
retrieved. Finally, the MAP is consequently defined as the mean of the average precision scores from a set of
queries.

- **Recall@1000**, indicating the proportion of relevant document IDs found in the top-1000 results.

---

## Baseline Fine-Tuning

We performed the fine-tuning of a pre-trained T5 model, in order to adapt it to our task: given a query q, the model should output a ranked list of document
IDs. The T5 model was included in a PyTorch Lightning module, useful to define our specific training, validation and testing
logic. We handled the aggregation logic of losses and metrics. T5 uses the regular cross-entropy loss, and we used
Adam optimizer for training our model (with betas fixed to 0.9 and 0.98). The training hyperparameters we have
considered are mainly the number K of retrieved document IDs for each query, the batch size, the number of epochs,
the learning rate, the epsilon associated to the optimizer and the chosen T5 model.

![image](https://github.com/user-attachments/assets/60af4800-934e-4ed8-b6d1-d8dc296631e2)

---

## Pruning

Our baseline is the result of a first fine-tuning of the whole model. Our proposed innovation is to provide a model compression technique on
this baseline in order to use this model in a resource constrained environment, showing promising future improvements.
After pruning, the model's performance may degrade due to the removal of certain parameters or connections. Recovery techniques aim to mitigate this performance loss by retraining the pruned model and fine-tuning its remaining parameter.

![image](https://github.com/user-attachments/assets/f29e0f35-1d54-4270-9d28-d675c276e796)

---

## Results

We have executed a train-prune-recovery approach: we have trained the baseline for 10 epochs, then we have pruned the model with different pruning rates, and finally we have fine-tuned the model on the same dataset for other 15 epochs.
We performed hyperparameter tuning on the t5-small model. Our aim was to
understand the hyperparameters leading to the best performance in terms of the Mean Average Precision (MAP) and Recall@1000 metrics.

The most notable results we have obtained are summarized in the following tables:
![image](https://github.com/user-attachments/assets/74bd1f88-0166-4cd2-8ab2-e40f747a304b)
![image](https://github.com/user-attachments/assets/ea77771b-60f2-4ea6-aaa5-148e205d7e8c)
As we can see from tables 2 and 3, the train-prune-recovery cycle
we propose in this paper is actually improving the performance of the model with respect to the baseline, proving that
our approach can be valid also in a less constrained environment.

---

## How to run the code

For running our proposed solution, just open our [colab notebook](https://colab.research.google.com/drive/1xgEKauqx6sjTNUg5zu-YFrHpH-HFHN8T) and run all the cells. The model should be able to run within the free colab runtime constraints.

---

## Contacts

If you encounter any issue or if you want to propose any kind of improvement to our model, feel free to send us an email:
- Luca Zanchetta [[Email](zanchetta.1848878@studenti.uniroma1.it)] [[GitHub](https://github.com/luca-zanchetta)]
- Simone Scaccia [[Email](scaccia.2045976@studenti.uniroma1.it)] [[GitHub](https://github.com/simonescaccia)]
- Pasquale Mocerino [[Email](mocerino.1919964@studenti.uniroma1.it)] [[GitHub](https://github.com/pasqualemocerino)]
