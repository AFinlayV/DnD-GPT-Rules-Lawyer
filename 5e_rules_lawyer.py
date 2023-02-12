"""
Ok... So
I want to build a script that is a discord bot that uses semantic search of the 5e rules to answer questions
using gpt3 to generate answers to questions. May not even need the semantic search, as gpt3 can probably
do most of it from the foundation model.

TODO:
    [ ] Check token length of context and trim/summarize/semantic search if too long
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
import discord

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


SRD = load_json('docs/srd.json')
intents = discord.Intents.all()
client = discord.Client(intents=intents,
                        shard_id=0,
                        shard_count=1,
                        reconnect=True)


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


def fetch_rule(srd, query, count=3):
    qvector = gpt3_embedding(query)
    scores = {}
    # use similarity() to compare the qvector to the vector of each rule, then return the top 5
    for rule in srd:
        scores[rule] = similarity(qvector, srd[rule])
    top = sorted(scores, key=scores.get, reverse=True)[:count]
    return top


def gpt3_completion(prompt, engine='text-davinci-003', temp=0.0, tokens=256, top_p=1.0, freq_pen=0.0, pres_pen=0.0):
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
            return text
        except Exception as oops:
            retry += 1
            if retry >= max_retry:
                return "GPT3 error: %s" % oops
            print('Error communicating with OpenAI:', oops)
            sleep(1)


def process_message(discord_message):
    discord_text = discord_message.content
    user = discord_message.author.name
    try:
        query = discord_message.content
        top = fetch_rule(SRD, query)
        # make the list a string
        rules = ' '.join(top)
        # for rule in top:
        #     print(rule, "\n\n====================\n\n")
        prompt = f"""
                I am a dungeon master named "RulesLawyer" and I have been asked a question about the rules of Dungeons and Dragons.
                I will only answer questions based on the rules given below, and I will not say anything that can't be inferred from them.
                I will also refuse to answer questions that are not about the rules of Dungeons and Dragons.
                Using the only the information below (from the SRD), I will answer the question as best I can.

                +++++++

                Rules: {rules}

                +++++++

                Question: {query}

                Below I will write an answer to the question, as it applies to Dungeons and Dragons 5e. I will only use infrormation from the rules above.
                and I will provide a long, detailed, and thorough answer. I will also provide the text of the rule that my answer is based on.
                RulesLawyer:"""
        response = gpt3_completion(prompt)
        return response
    except Exception as oops:
        return {'output': 'Error in process_message: %s' % oops, 'user': user}


@client.event
async def on_message(discord_message, timeout=60):
    discord_text = discord_message.content
    user = discord_message.author.name
    try:
        if not discord_message.author.bot:
                await discord_message.channel.send(f'Generating response for {user}:')
                await asyncio.wait_for(
                    send_response(discord_message),
                    timeout=timeout)
    except asyncio.TimeoutError:
        print(f'Response timed out. \n {user}: {discord_text[:20]}...')
    except Exception as oops:
        await discord_message.channel.send(f'Error: {oops} \n {user}: {discord_text[:20]}...')


async def send_response(discord_message):
    try:
        channel = discord_message.channel
        user = discord_message.author.name
        discord_text = discord_message.content
        response = process_message(discord_message)
        # Create a new task for sending the response
        await discord_message.channel.send(f'Rules Lawyer Says:\n {response}')
    except Exception as oops:
        # Something else went wrong while sending the response
        await discord_message.channel.send('Error sending response:', oops)


@client.event
async def on_ready():
    channel = client.get_channel(CHAN_ID)
    if channel is not None:
        await channel.send(f"{client.user} has connected to Discord!")
    else:
        print("Channel not found")


if __name__ == "__main__":
    openai.api_key = os.environ.get('OPENAI_API_KEY')
    TOKEN = os.environ.get('RULES_LAWYER_TOKEN')
    CHAN_ID = int(os.environ.get('RULES_LAWYER_CHANNEL_ID'))
    try:
        client.run(TOKEN)
    except Exception as oops:
        print('Error in main:', oops)

'''
Old code for testing in command line:
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
        Using the only the information below (from the SRD), I will answer the question as best I can.
        
        +++++++
        
        Rules: {rules}
        
        +++++++
        
        Question: {query}
        
        Below I will write an answer to the question, as it applies to Dungeons and Dragons 5e. I will only use infrormation from the rules above.
        and I will provide a long, detailed, and thorough answer. I will also provide the text of the rule that my answer is based on.
        RulesLawyer:"""
        response = gpt3_completion(query)
        print(response)


if __name__ == "__main__":
    main()
'''
