from fastapi import FastAPI
from pydantic import BaseModel

from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM, BitsAndBytesConfig
from collections import Counter
import pandas as pd
import numpy as np
import torch
import ast
import re
import os


app = FastAPI()

tokenizer = AutoTokenizer.from_pretrained('cardiffnlp/twitter-roberta-base-sentiment')
model = AutoModelForSequenceClassification.from_pretrained('cardiffnlp/twitter-roberta-base-sentiment', device_map='auto', torch_dtype='auto')

class ReviewRequest(BaseModel):
    review: str

labels = {0:'negative', 1: 'neutral', 2: 'positive'}
def get_sentiment_from_txt(text: str):
    input_data = tokenizer.encode(text, return_tensors='pt').to(model.device)
    results = model(input_data)
    label_idx = int(torch.argmax(results.logits))
    return labels[label_idx]


@app.post('/sentiment')
def get_sentiment(req: ReviewRequest):
    review = req.review
    sentiment = get_sentiment_from_txt(review)
    return {'sentiment': sentiment}


model_path = r"D:\models\Qwen2.5-Coder-7B-Instruct"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype='float16'
)

print('Loading Llama Model🔥 ...')
llm_tokenizer = AutoTokenizer.from_pretrained(model_path)
llm_model = AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config=bnb_config,
    device_map='auto'
)
print('✅ Model ready!')


class FeedbackRequest(BaseModel):
    review: str
    domain: str

@app.post('/feedbacks')
def get_feedback(req: FeedbackRequest):
    review = req.review
    domain = req.domain

    sentiment = get_sentiment_from_txt(review)

    DOMAIN_ACTOR_MAP = {
        "sports store": "the customer",
        "college": "the student",
        "restaurant": "the customer",
        "gym": "the member",
        "hospital": "the patient or relative",
        "shopping mall": "the customer",
        "school": "the student",
        "general": "the reviewer"
    }
    actor = DOMAIN_ACTOR_MAP.get(domain, 'the reviewer')
    
    prompt = f"""
    Context: 
    - The review is from the domain {domain.upper()}.
    - Always use "{actor}" when rewriting sentences in the third-person perspective.
    
    INSTRUCTIONS FOR STRICT PYTHON DICTIONARY GENERATION:

    IMPORTANT RULE:
    - You must ONLY use information explicitly present in the review.
    - Do NOT invent, assume, or add any detail that is not directly stated.

    Bad Example (hallucination):
    Review: "Fabulous experience"
    Output: {{"sentence1": "{actor} had a fantastic time.", "sentence2": "The service was impeccable.", "sentence3": "The food exceeded expectations."}}
    
    Good Example (faithful):
    Review: "Fabulous experience"
    Output: {{"sentence1": "{actor} had a fabulous experience.", "sentence2": "", "sentence3": ""}}

    TASK OVERVIEW:
    1.  You are a highly skilled data extraction specialist.
    2.  Your task is to analyze a user review and extract a MAXIMUM of 3 MEANINGFUL sentences that reflect the given sentiment.
    3.  Each extracted sentence must be a complete thought, be written in the THIRD-PERSON perspective, and PRESERVE THE ORIGINAL SENTIMENT OF THE REVIEW.
    4.  The output MUST strictly adhere to the specified Python dictionary format.
    
    # PROCESSING STEPS
    
    Step 1: Analyze the Review
    -   Read the provided review carefully.
    -   Identify the PRIMARY sentiment expressed: negative, positive, or neutral.
    -   Take note of specific nouns, actions, and descriptive language that support this sentiment.
    
    Step 2: Extract Core Ideas
    -   From the review, select up to three DISTINCT and NON-REDUNDANT pieces of information.
    -   Prioritize sentences that contain specific details (e.g., names, actions, reasons) over general statements.
    -   Preferentially select sentences that EXPLAIN *why* the sentiment is present (e.g., "The soup was cold" is better than "The food was bad").
    
    Step 3: Rewrite in Third Person
    -   Convert all first-person singular pronouns (`I`, `my`) to "{actor}".
    -   Convert all first-person plural pronouns (`we`, `our`) to "{actor}s".
    -   Retain the original punctuation and capitalization.
    -   Preserve the emotional intensity and nuance of the original statement.
    
    Step 4: Format the Output
    -   Create a Python dictionary with exactly three keys: `sentence1`, `sentence2`, and `sentence3`.
    -   Each value should be a string containing one of your extracted and rewritten sentences.
    -   If the review contains fewer than three distinct points, leave the remaining keys as empty strings (`""`).
    -   Your ONLY output should be the dictionary. Do NOT include any explanations, code fences, or additional text.
    
    # SPECIFICATIONS
    
    -   Sentence Length: Aim for a natural sentence length between 10 and 20 words. Do not force an exact count.
    -   Redundancy: Ensure each sentence provides a unique piece of information. Do NOT repeat ideas [[8]].
    -   Accuracy: Do NOT invent or generalize information. Only include details explicitly stated in the review.
    -   Formatting: The final output must be a SINGLE, valid Python dictionary printed exactly as shown in the examples.
    
    Review: {review}
    
    Sentiment: {sentiment}
    
    Output:
    """
    
    input_data = llm_tokenizer(prompt, return_tensors='pt').to(llm_model.device)
    results = llm_model.generate(
        **input_data,
        max_new_tokens=80,
        do_sample=False
    )

    prompt_len = input_data.input_ids.shape[1]
    generated_token = results[0][prompt_len:]

    output = llm_tokenizer.decode(generated_token, skip_special_tokens=True)

    patterns = r'\{.*\}'
    match = re.search(patterns, output, re.DOTALL)

    if match:
        result_str = match.group()
        try:
            result = ast.literal_eval(result_str)
            return {
                'feedback1': result.get('sentence1', ''),
                'feedback2': result.get('sentence2', ''),
                'feedback3': result.get('sentence3', '')
            }
        except Exception as e:
            print('Parsing error', e)
            return {
                'feedback1': '', 
                'feedback2': '', 
                'feedback3': ''
            }

    else:
        print('No match found')
        return {
                'feedback1': '', 
                'feedback2': '', 
                'feedback3': ''
            }
    