# Part 3: NLP and Sequence Modeling - Customer Support Sentiment

## Project Overview
In this project, I built an NLP pipeline to classify customer support messages into three categories: Positive, Neutral, and Negative. I compared how text is processed and understood by machine learning models.

## Task 1: Dataset Understanding
- **Total Records:** 1500
- **Target Labels:** Positive, Neutral, Negative
- **Average Word Count:** ~12-13 words per message

## Task 3: Why Vectorization?
Machine learning models cannot read text; they only understand numbers. Vectorization (like TF-IDF) converts words into a numerical format, allowing the model to calculate the importance of each word and find patterns in the data.

## Task 5: Sequence Model Architecture (LSTM)
While I used Naive Bayes as a baseline, a sequence model like an **LSTM (Long Short-Term Memory)** would process this data as follows:
1. **Input Sequence:** The raw text tokens.
2. **Embedding Layer:** Converts tokens into dense vectors that capture word meanings.
3. **LSTM Layer:** Processes words one by one while keeping a "memory" of previous words.
4. **Output Layer:** A Dense layer with Softmax to predict the final sentiment.

## Task 6: Attention and Transformers
- **RNN Struggles:** Traditional RNNs often "forget" the beginning of a long sentence by the time they reach the end (Vanishing Gradient).
- **LSTM Memory:** LSTMs use a "cell state" to selectively remember or forget information over long sequences.
- **Attention:** Instead of reading word-by-word, Attention allows the model to "focus" on the most relevant words in a sentence regardless of their position.
- **Transformers:** These are the foundation of modern AI (like ChatGPT). They use self-attention to process all words in a sentence simultaneously, making them much faster and more accurate than RNNs.

## Repository Contents
- `notebook.ipynb`: Data cleaning and model training.
- `requirements.txt`: List of dependencies.
- `results/`: Contains the evaluation report and sample predictions.