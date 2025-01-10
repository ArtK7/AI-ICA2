# AI-ICA2: Sentiment Analysis Project

This repository contains the implementation of a sentiment analysis project for the ICA-2 assessment, leveraging two fine-tuned DistilBERT models to classify IMDb reviews as positive or negative. The project demonstrates the application of Natural Language Processing (NLP) techniques in a real-world task and fulfills the requirements of the ICA-2 assessment.

## Project Objectives
Perform sentiment analysis on the IMDb dataset. Compare the performance of two fine-tuned DistilBERT models: distilbert-base-uncased-finetuned-sst-2-english (fine-tuned on SST-2) and textattack/distilbert-base-uncased-imdb (fine-tuned on IMDb). Present results using accuracy, precision, recall, and F1 score metrics.

## Features
1. Dataset: Uses the IMDb dataset from Hugging Face's datasets library. Balanced dataset with 250 samples (125 positive, 125 negative) for fair evaluation.
2. Models: 
   - Model 1: distilbert-base-uncased-finetuned-sst-2-english, fine-tuned on Stanford Sentiment Treebank (SST-2).
   - Model 2: textattack/distilbert-base-uncased-imdb, fine-tuned specifically on IMDb data for sentiment analysis.
3. Performance Metrics: Evaluates models using Accuracy, Precision, Recall, and F1 Score.
4. Visualization: Clean, color-coded terminal outputs for enhanced readability.

## Requirements
Before running the project, ensure the following are installed:
- Python 3.8 or later
- Required libraries (listed in requirements.txt):
  - transformers
  - torch
  - datasets
  - scikit-learn
  - colorama

## Setup Instructions
1. Clone the repository:
   git clone https://github.com/ArtK7/AI-ICA2.git
   cd AI-ICA2
2. Create and activate a virtual environment:
   python3 -m venv venv
   source venv/bin/activate
3. Install dependencies:
   pip install -r requirements.txt

## Running the Project
1. Run the main script:
   python main.py
2. Observe the terminal output for:
   - Balanced dataset creation.
   - Performance metrics (accuracy, precision, recall, F1 score).
   - Positive and negative review counts for each model.

## Example Output
Below is a sample output of the project execution:

AI-ICA2: Sentiment Analysis Project  


* Step 1: Loading IMDb dataset...


* Step 2: Tokenizing the dataset...  


* Step 3: Balancing the dataset...  
Balanced dataset created with 250 samples (125 positive, 125 negative).  


* Step 4: Testing model: distilbert-base-uncased-finetuned-sst-2-english  
Loading model...  


* Evaluating model performance...  
* Positive Reviews: 114  
* Negative Reviews: 136  


* Results: 


* Accuracy 0.89,  
* Precision 0.93 
* Recall 0.85 
* F1 Score 0.89  

Testing model: textattack/distilbert-base-uncased-imdb  
* Positive Reviews: 122  
* Negative Reviews: 128  


* Results: 
* Accuracy 0.90 
* Precision 0.91
* Recall 0.89 
* F1 Score 0.90

## Assessment Relevance
This project aligns with the ICA-2 requirements by demonstrating:
* Implementation of an NLP Task: Sentiment analysis using pre-trained transformer models.
* Evaluation of Models: Comparison of performance metrics across two models.
* Practical Application of AI Concepts: Use of fine-tuned models for a real-world sentiment analysis task.

## Potential Improvements
* Fine-tune additional models for comparison.
* Include visualization plots of model performance metrics.
* Extend to multi-class sentiment analysis.

## Author
Art Kastrati  
AI-ICA2 Project | Prague City University | 2025
