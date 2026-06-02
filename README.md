# Book Recommendation System using Embeddings & LLMs

## Overview

A book recommendation system that uses **text embeddings** and **Large Language Models (LLMs)** to provide personalized recommendations.

The system goes beyond keyword matching by understanding semantic meaning and user intent. It supports both **fiction** and **non-fiction** books and includes an optional **emotion-aware layer** to improve personalization.

---

## Features

- Semantic search using embeddings  
- LLM-based recommendation generation  
- Emotion-aware recommendations  
- Fiction and non-fiction support  
- Context-aware results  

---

## How it works

1. User enters a natural language query  
2. Text is converted into embeddings  
3. Similar books are retrieved using vector similarity  
4. LLM refines and explains recommendations  
5. Emotion layer adjusts results based on sentiment  

---

## Tech Stack

- Python  
- NLP (Natural Language Processing)  
- Sentence-BERT / embeddings model  
- Large Language Models (LLMs)  
- Pandas, NumPy  
- Jupyter Notebook / Streamlit  

---

## Project Structure
book-recommendation-project/
│
├── app/  
│   ├── main.py              # Entry point of the application  
│   ├── pages/              # UI pages (if using Streamlit multipage)  
│
├── core/  
│   ├── embeddings.py       # Embedding generation logic  
│   ├── recommender.py      # Similarity search & ranking  
│   ├── llm.py              # LLM prompt engineering & responses  
│   ├── emotion.py          # Emotion detection module  
│
├── data/  
│   ├── raw/                # Original dataset  
│   ├── processed/          # Cleaned dataset  
│
├── notebooks/  
│   ├── 01_data_analysis.ipynb  
│   ├── 02_embeddings_test.ipynb  
│   ├── 03_llm_experiments.ipynb  
│
├── utils/  
│   ├── preprocessing.py    # Text cleaning functions  
│   ├── helpers.py          # Utility functions  
│
├── assets/  
│   ├── demo.png            # App screenshot  
│   ├── architecture.png    # System diagram  
│
├── tests/  
│   ├── test_recommender.py # Basic unit tests  
│
├── requirements.txt  
├── README.md  
└── .gitignore  

---

## Example

**Input:**
"I want a motivational book that helps me focus"

**Output:**
A list of recommended books based on semantic similarity and emotional context.

---

## Goals

- Improve recommendation quality using NLP techniques  
- Combine semantic + emotional understanding  
- Move beyond traditional recommendation systems  

---

## Future Work

- Improve emotion detection  
- Expand dataset  
- Deploy as web app  
- Add feedback system  

---

## Author

**Isra Chaker**  
ESSAI – Data Science & AI Engineering
