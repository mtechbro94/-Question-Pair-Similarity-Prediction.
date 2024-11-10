# Quora Question Pair Similarity

## Table of Contents
1. [Project Overview](#project-overview)
2. [Problem Statement](#problem-statement)
3. [Objectives and Constraints](#objectives-and-constraints)
4. [Performance Metrics](#performance-metrics)
5. [Data Overview](#data-overview)
6. [Exploratory Data Analysis](#exploratory-data-analysis)
7. [Feature Engineering](#feature-engineering)
8. [Model Training and Evaluation](#model-training-and-evaluation)
9. [Results](#results)
10. [References](#references)

---

## Project Overview
With over 100 million monthly visitors, Quora has become a popular platform for knowledge sharing. However, users often ask questions with similar intent, leading to redundant answers and an inefficient experience for both question seekers and answer writers. This project aims to predict whether a pair of questions on Quora are duplicates, allowing users to find answers faster and reducing redundant question-answer pairs.

## Problem Statement
Identify which questions asked on Quora are duplicates of questions that have already been asked. This helps in providing quicker answers and improves the user experience by consolidating information under canonical questions.

## Objectives and Constraints
- **Misclassification Cost**: High, as incorrect duplicate detection impacts the user experience.
- **Output as Probability**: Provides flexibility in threshold selection for classifying duplicates.
- **Latency**: No strict real-time requirement.
- **Interpretability**: Partially important for understanding feature relevance.

## Performance Metrics
- **Log Loss**: Measures the performance of the classification model.
- **Binary Confusion Matrix**: To evaluate true positives, true negatives, false positives, and false negatives.

## Data Overview
The dataset used for this project is `train.csv`, containing 404,290 entries with the following columns:
- `qid1`, `qid2`: Unique IDs for each question
- `question1`, `question2`: The questions in text format
- `is_duplicate`: Binary label indicating if the questions are duplicates

### Data Split
The data is split into 70% training and 30% testing sets.

## Exploratory Data Analysis
Performed EDA to understand the distribution and characteristics of the dataset:
- **Class Distribution**: Analyzed distribution of duplicate and non-duplicate question pairs.
- **Question Occurrence**: Explored frequency and unique occurrences of each question.
- **Handling Null Values**: Two null entries were replaced with an empty space.
- **Wordclouds**: Created wordclouds for both similar and dissimilar questions to visualize common terms.

## Feature Engineering
Extracted a variety of **basic** and **advanced** features to enhance model performance.

### Basic Features
- `freq_qid1`, `freq_qid2`: Frequency of each question ID in the dataset
- `q1len`, `q2len`: Lengths of question1 and question2
- `q1_n_words`, `q2_n_words`: Word count in each question
- `word_common`: Number of unique common words between the two questions
- `word_total`: Total words across both questions
- `word_share`: Ratio of common words to total words
- `freq_q1+freq_q2`, `freq_q1-freq_q2`: Sum and absolute difference of frequencies of `qid1` and `qid2`

### Advanced Features
Advanced features derived after preprocessing the text:
- **Ratios**: `cwc_min`, `cwc_max`, `csc_min`, `csc_max`, `ctc_min`, `ctc_max` to capture word, stop word, and token relationships.
- **Positional Equality**: `last_word_eq`, `first_word_eq` for comparing first and last words.
- **Text Distance Metrics**: `fuzz_ratio`, `fuzz_partial_ratio`, `token_sort_ratio`, `longest_substr_ratio` to measure string similarity.

### Distance-Based Features
Using pre-trained GloVe embeddings, calculated distances:
- **Word Mover’s Distance**
- Cosine, Cityblock, Canberra, Euclidean, and Minkowski distances

### TF-IDF Features
Generated 1-gram, 2-gram, and 3-gram features from combined question1 and question2 text.

## Model Training and Evaluation
The following models were trained and evaluated using the features extracted. Due to memory constraints, a sample of the training data was used for model development.

| Model                  | Features Used                    | Log Loss   |
|------------------------|----------------------------------|------------|
| Logistic Regression    | Basic + Advanced                | 0.4003     |
| Linear SVM             | Basic + Advanced                | 0.4036     |
| Random Forest          | Basic + Advanced                | 0.4144     |
| XGBoost                | Basic + Advanced                | 0.3625     |
| Logistic Regression    | Basic + Advanced + Tf-Idf       | 0.3584     |
| XGBoost                | Basic + Advanced + Distances + Avg-W2V | 0.3133 |

### Hyperparameter Tuning
Performed hyperparameter tuning using Random and Grid Search to optimize model performance.

### Model Performance
The best model achieved a log loss of 0.3133 using XGBoost with a combination of basic, advanced, and distance-based features.

## Results
- The XGBoost model with combined features performed best, achieving a log loss of 0.3133.
- The feature engineering, especially advanced distance features and TF-IDF, significantly improved the model’s capability to distinguish duplicate questions.

## References
- [Quora Question Pairs on Kaggle](https://www.kaggle.com/c/quora-question-pairs)
- [Applied AI Course](https://www.appliedaicourse.com/)
- [FuzzyWuzzy Python Package](https://github.com/seatgeek/fuzzywuzzy)
- [Word Mover's Distance - MLR Proceedings](http://proceedings.mlr.press/v37/kusnerb15.pdf)

---

