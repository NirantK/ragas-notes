from glob import glob
from importlib.metadata import files
from matplotlib.pyplot import get
import numpy as np
import openai
from langchain.embeddings import OpenAIEmbeddings
from sklearn import neighbors
from sklearn.feature_extraction.text import TfidfVectorizer
from sympy import content
import json
from prompts import *
import os
import time 
from tqdm import tqdm

Embedding = OpenAIEmbeddings()
MODEL = "gpt-4"

def get_text_files(folder_name):
    return glob(f'{folder_name}/*')
    
    
def choose_random(indices, excluded_indices=[]):
    
    indices = np.setdiff1d(indices, excluded_indices)
    return np.random.choice(indices, size=1)[0]

    
def split_text(text, max_length=1000):
        chunks = []
        corpus = text.split('\n')
        
        current_chunk = ""
        for chunk in corpus:
            
            if len(current_chunk + chunk)//4 < max_length:
                current_chunk = "\n".join([current_chunk, chunk])
            else:
                chunks.append(current_chunk)
                current_chunk = chunk
            
        return chunks
    
def find_neighbour(text, text_list, min_similarity=0.8, top_k=3):
        text_vec = np.asarray(Embedding.embed_query(text)).reshape(1, -1)
        text_list_vec = np.asarray(
            Embedding.embed_documents(text_list)
        )
        norm = np.linalg.norm(text_list_vec, axis=1) * np.linalg.norm(
            text_vec, axis=1
        )
        similarity =  (
            np.dot(text_list_vec, text_vec.T).reshape(
                -1,
            )
            / norm
        )
        similarity  = similarity[similarity>min_similarity]
        return similarity.argsort().tolist()[-top_k:]
    
def doc_filter(corpus, top_k=20):
    vectorizer = TfidfVectorizer(min_df=0.05, max_df=0.5)
    X = vectorizer.fit_transform(corpus)
    X = X.mean(axis=1).reshape(1,-1).argsort().tolist()[0]
    return X[-top_k:][::-1]


def get_llm_response(prompt, **kwargs):
        if MODEL == 'gpt-4':
            time.sleep(3)
        response = openai.ChatCompletion.create(
        model=kwargs.get("model", MODEL),
        messages=[{"role": "system", "content": prompt}],
        temperature=kwargs.get("temperature", 0),
        top_p=kwargs.get("top_p", 1),
        frequency_penalty=kwargs.get("frequency_penalty", 0.0),
        presence_penalty=kwargs.get("presence_penalty", 0.0),
        max_tokens=kwargs.get("max_tokens", 500),
        n=kwargs.get("n", 1),
        )
        return response['choices'][0]['message']['content']

def decide_path(prob):
    
    if prob<=0.25:
        return "NO-EVOLVE"
    elif prob <= 0.5:
        return "CONDITION"
    elif prob <= 0.75:
        return "MULTI-CONTEXT"
    else:
        return "REASON"

def Evolgen(chunk_id, chunks):
    
    evolved = False
    seed_context = chunks[chunk_id]
    seed_question = get_llm_response(SEED_QUESTION.format(context=seed_context))
    
    prob = np.random.uniform(0,1)
    decision = decide_path(prob)
    
    if decision == "CONDITION":
        question = get_llm_response(CONDITIONAL_QUESTION.format(question=seed_question,context=seed_context))
        evolved = True
    elif decision == "MULTI-CONTEXT":
        similar_indices = find_neighbour(chunks[chunk_id], chunks)
        similar_indices = np.setdiff1d(similar_indices,[chunk_id])
        neighbor_idx = similar_indices[0]
        question = get_llm_response(MULTICONTEXT_QUESTION.format(question=seed_question, context1=seed_context, context2=chunks[neighbor_idx]))
        seed_context = "\n\n".join([seed_context,chunks[neighbor_idx]])
        evolved = True
    elif decision == "REASON":
        question = get_llm_response(REASONING_QUESTION.format(question=seed_question,context=seed_context))
        evolved = True
        
    else:
        question = seed_question
        evolved = False
    
    if evolved:
        # TODO: Add a question elimination condition (this could in answering/context extraction phase)
        if np.random.uniform(0,1)>0.5:
            question = get_llm_response(COMPRESS_QUESTION.format(question=question))
        else:
            question = get_llm_response(CONVERSATION_QUESTION.format(question=question))
            
    
    
    return question, seed_context
    
def doc_filter(corpus, top_k=20):
    vectorizer = TfidfVectorizer(min_df=0.05,max_df=0.5)
    X = vectorizer.fit_transform(corpus)
    X = X.mean(axis=1).reshape(1,-1).argsort().tolist()[0]
    return X[-top_k:][::-1]
    
def read_doc(path):
    with open(path,'r') as file:
        return file.read()
    
def main(folder_name):
    
    questions = []    
    files = get_text_files(folder_name)
    for _ in tqdm(range(5)):
        file = choose_random(files)
        text = read_doc(file)
        chunks = split_text(text)
        top_chunk_ids = doc_filter(chunks)
        chunk_id = choose_random(top_chunk_ids)
        output = Evolgen(chunk_id, chunks)
        questions.append(output)
    
    with open("questions.json",'w') as json_file:
        json.dump(questions, json_file, indent=4)


if __name__ == "__main__":
    
    folder_name = "/Users/shahules/Myprojects/notes/arxiv-llm/textdata/"
    main(folder_name)