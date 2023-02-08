"""
Ok... So
I want to build a script that is a discord bot that uses semantic search of the 5e rules to answer questions
using gpt3 to generate answers to questions. May not even need the semantic search, as gpt3 can probably
do most of it from the foundation model.
"""
import asyncio
import re
from time import sleep
import openai
import os
import json
import discord
import langchain
import gpt_index
import datetime
from numpy.linalg import norm
import numpy as np

with open("/Users/alexthe5th/Documents/API Keys/OpenAI_API_key.txt", "r") as f:
    openai.api_key = f.read().strip()


def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()


def save_file(filepath, content):
    with open(filepath, 'w', encoding='utf-8') as outfile:
        outfile.write(content)


def load_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return json.load(infile)


def save_json(filepath, payload):
    with open(filepath, 'w', encoding='utf-8') as outfile:
        json.dump(payload, outfile, ensure_ascii=False, sort_keys=True, indent=2)


def timestamp_to_datetime(unix_time):
    return datetime.datetime.fromtimestamp(unix_time).strftime("%A, %B %d, %Y at %I:%M%p %Z")


def gpt3_embedding(content, engine='text-embedding-ada-002'):
    content = content.encode(encoding='ASCII', errors='ignore').decode()
    response = openai.Embedding.create(input=content, engine=engine)
    vector = response['data'][0]['embedding']  # this is a normal list
    return vector


def similarity(v1, v2):
    # based upon https://stackoverflow.com/questions/18424228/cosine-similarity-between-2-number-lists
    return np.dot(v1, v2) / (norm(v1) * norm(v2))  # return cosine similarity


def fetch_rule(srd, query, count=2):
    qvector = gpt3_embedding(query)
    scores = {}
    # use similarity() to compare the qvector to the vector of each rule, then return the top 5
    for rule in srd:
        scores[rule] = similarity(qvector, srd[rule])
    top = sorted(scores, key=scores.get, reverse=True)[:count]
    return top


def gpt3_completion(prompt, engine='text-davinci-003', temp=0.0, tokens=512, top_p=1.0, freq_pen=0.0, pres_pen=0.0):
    max_retry = 5
    retry = 0
    prompt = prompt.encode(encoding='ASCII', errors='ignore').decode()
    while True:
        try:
            response = openai.Completion.create(
                engine=engine,
                prompt=prompt,
                temperature=temp,
                max_tokens=tokens,
                top_p=top_p,
                frequency_penalty=freq_pen,
                presence_penalty=pres_pen)
            text = response['choices'][0]['text'].strip()
            text = re.sub('[\r\n]+', '\n', text)
            text = re.sub('[\t ]+', ' ', text)
            filename = '%s_gpt3.txt' % datetime.time()
            if not os.path.exists('gpt3_logs'):
                os.makedirs('gpt3_logs')
            save_file('gpt3_logs/%s' % filename, prompt + '\n\n==========\n\n' + text)
            return text
        except Exception as oops:
            retry += 1
            if retry >= max_retry:
                return "GPT3 error: %s" % oops
            print('Error communicating with OpenAI:', oops)
            sleep(1)


def main():
    srd = load_json('docs/srd.json')
    while True:
        query = input("What do you want to know? ")
        top = fetch_rule(srd, query)
        # make the list a string
        rules = ' '.join(top)
        # for rule in top:
        #     print(rule, "\n\n====================\n\n")
        prompt = f"""
        I am a dungeon master named "RulesLawyer" and I have been asked a question about the rules of Dungeons and Dragons.
        I will only answer questions based on the rules given below, and I will not say anything that can't be inferred from them.
        I will also refuse to answer questions that are not about the rules of Dungeons and Dragons.
        Using the Information below (from the SRD), I will answer the question as best I can.
        Rules: {rules}
        Question: {query}
        RulesLawyer's Answer, including the text of the rule the answer is based on:
        """
        response = gpt3_completion(query)
        print(response)


if __name__ == "__main__":
    main()
