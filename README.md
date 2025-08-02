# 🤖 Question Answering with Transformers

This repository contains a complete implementation of a **Question Answering (QA)** system using **pre-trained Transformer models**, specifically **BERT fine-tuned on the SQuAD v1.1 dataset**.

## 📌 Overview

Question Answering is a Natural Language Processing (NLP) task where a model is given a **context paragraph** and a **question**, and it must extract the most probable **answer span** from the context.

In this project, we:
- Use the `bert-large-uncased-whole-word-masking-finetuned-squad` model from Hugging Face
- Apply it to the **Stanford Question Answering Dataset (SQuAD v1.1)**
- Evaluate the model using **Exact Match (EM)** and **F1 Score**

---

## 📚 Dataset: SQuAD v1.1

The [Stanford Question Answering Dataset (SQuAD)](https://rajpurkar.github.io/SQuAD-explorer/) is a reading comprehension dataset consisting of:
- 100,000+ questions posed by crowdworkers on Wikipedia articles
- Each question has an answer span present in the text

---

## 🧰 Tools and Libraries

- Python 3.8+
- [Transformers](https://huggingface.co/transformers/)
- [Datasets](https://huggingface.co/docs/datasets/)
- PyTorch or TensorFlow
- Matplotlib, tqdm

---

## 🚀 How to Run

1. **Install dependencies**:

   ```bash
   pip install transformers datasets torch
   
2.Run the notebook or Python script to:

 Load the SQuAD v1.1 dataset

 Load the BERT QA model and tokenizer

 Tokenize context and question

 Predict the answer span

 Evaluate using EM and F1

3.Model Used:

  from transformers import BertTokenizer, BertForQuestionAnswering
  
  tokenizer = BertTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
  
  model = BertForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")

✅ Evaluation Metrics

    Exact Match (EM): Checks if the predicted span matches the ground truth exactly.

    F1 Score: Measures overlap between predicted and true answer tokens.

📊 Example

    Context:
    
       "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France." 
       
    Question:
    
       Where is the Eiffel Tower located?
       
    Predicted Answer:
    
       Paris, France
       
📦 Project Structure

     📂 question-answering-transformers
      ├── README.md
      ├── requirements.txt
      └── qa_with_bert.ipynb
      
🙋‍♀️ Author

  Hanen Ebrahim
  
  Intern @ Elevvo NLP Internship Program
  
🌐 References

    Hugging Face Transformers
    
    SQuAD Dataset
