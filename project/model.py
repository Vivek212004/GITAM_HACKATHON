import torch
from transformers import BertForQuestionAnswering, BertTokenizer
import random

# Load pre-trained BERT model and tokenizer
model_name = 'bert-large-uncased-whole-word-masking-finetuned-squad'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForQuestionAnswering.from_pretrained(model_name)

def get_answer(question, paragraph):
    inputs = tokenizer.encode_plus(question, paragraph, return_tensors='pt')
    input_ids = inputs['input_ids']
    token_type_ids = inputs['token_type_ids']

    # Get model outputs
    outputs = model(input_ids, token_type_ids=token_type_ids)
    start_scores = outputs.start_logits
    end_scores = outputs.end_logits

    # Get the most likely beginning and end of answer
    answer_start = torch.argmax(start_scores)
    answer_end = torch.argmax(end_scores) + 1

    # Convert tokens to string
    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[0][answer_start:answer_end]))
    return answer

paragraph ="India, country that occupies the greater part of South Asia. It is made up of 28 states and eight union territories, and its national capital is New Delhi, built in the 20th century just south of the historic hub of Old Delhi to serve as India’s administrative center. Its government is a constitutional republic that represents a highly diverse population consisting of thousands of ethnic groups and hundreds of languages. India became the world’s most populous country in 2023, according to estimates by the United Nations."

def generate_random_question(paragraph):
    # Split the paragraph into sentences
    sentences = paragraph.split('.')

    # Randomly select a sentence
    sentence = random.choice(sentences)

    # Extract a random word from the sentence
    words = sentence.split()
    word = random.choice(words)

    # Generate a question based on the sentence and word
    sentence_words = sentence.split()
    if word in sentence_words:
        index = sentence_words.index(word)
        context_words = sentence_words[max(0, index-2):index] + sentence_words[index+1:min(len(sentence_words), index+3)]
        context = ' '.join(context_words)
        question_types = ['What', 'Where', 'Who', 'When', 'How', 'Why']
        question_type = random.choice(question_types)
        question = f'{question_type} {context}?'
    else:
        question = f'What is {word} doing?'

    return question

def generate_questions(paragraph, num_questions):
    questions = set()
    while len(questions) < num_questions:
        question = generate_random_question(paragraph)
        questions.add(question)
    return list(questions)
