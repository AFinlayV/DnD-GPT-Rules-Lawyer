"""
Ok... So
I want to build a script that is a discord bot that uses semantic search of the 5e rules to answer questions
using gpt3 to generate answers to questions. May not even need the semantic search, as gpt3 can probably
do most of it from the foundation model.

TODO:
    [ ] Check token length of context and trim/summarize/semantic search if too long
"""
import asyncio
from time import sleep
import openai
import os
import json
from numpy.linalg import norm
import numpy as np
import disnake
from disnake.ext import commands


def load_json(filename):
    with open(filename) as f:
        return json.load(f)


SRD = load_json('docs/srd.json')
intents = disnake.Intents.all()
bot = commands.Bot(intents=intents,
                   command_prefix='!')


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
            return text
        except Exception as oops:
            retry += 1
            if retry >= max_retry:
                return "GPT3 error: %s" % oops
            print('Error communicating with OpenAI:', oops)
            sleep(1)


def process_message(discord_message):
    discord_text = discord_message
    try:
        query = discord_message
        top = fetch_rule(SRD, query)
        rules = ' '.join(top)[:4096]
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
        return {'output': 'Error in process_message: %s' % oops}


@bot.command()
async def lawyer(ctx, *args):
    discord_text = " ".join(args)
    user = ctx.author.name
    try:
        if not ctx.author.bot:
            response = await asyncio.get_event_loop().run_in_executor(None, process_message, discord_text)
            await ctx.send(response)
    except Exception as oops:
        await ctx.channel.send(f'Error: {oops} \n {user}: {discord_text[:20]}...')


if __name__ == "__main__":
    openai.api_key = os.environ.get('OPENAI_API_KEY')
    TOKEN = os.environ.get('RULES_LAWYER_DISCORD_TOKEN')
    try:
        bot.run(TOKEN)
    except Exception as oops:
        print('Error in main:', oops)
