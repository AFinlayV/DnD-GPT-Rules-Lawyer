"""
I have the dungeons and dragons 5e rules in a .md file
this script will use gpt3 to greate embeddings for each sule and section of text
so that it can be semantically searched by another script
"""

import json
import os
import re
import openai


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


def gpt3_embedding(content, engine='text-embedding-ada-002'):
    content = content.encode(encoding='ASCII', errors='ignore').decode()
    response = openai.Embedding.create(input=content, engine=engine)
    vector = response['data'][0]['embedding']  # this is a normal list
    return vector


def crawl_file_structure(path):
    files = list()
    for root, dirs, filenames in os.walk(path):
        for filename in filenames:
            if filename.endswith('.md'):
                filepath = os.path.join(root, filename)
                files.append(filepath)
                print(f'Found {filepath}...')
    return files


def parse_file(filepath):
    print(f'Parsing {filepath}')
    content = open_file(filepath)
    content = re.sub(r'#+', '', content)
    content = re.sub(r'\n{2,}',
                     '   ', content)  # replace 2 or more newlines with 3 spaces
    content = re.sub(r'\n', ' ', content)  # replace newlines with spaces
    content = re.sub(r' {2,}', ' ', content)  # replace 2 or more spaces with 1 space
    return content


def split_into_sections(content):
    sections = list()
    section = ''
    for line in content.splitlines():
        if line.startswith('##'):
            if section:
                sections.append(section)
                section = ''
        section += line + ' '
    if section:
        sections.append(section)
    return sections


print('is it even running this?')
with open('/Users/alexthe5th/Documents/API Keys/OpenAI_API_key.txt', 'r') as f:
    openai.api_key = f.read().strip()
print(openai.api_key)
path = '/Users/alexthe5th/PycharmProjects/5thSRD/docs/'
files = crawl_file_structure(path)
for filepath in files:
    content = parse_file(filepath)
    sections = split_into_sections(content)
    for section in sections:
        print(f'Embedding {section}...')
        vector = gpt3_embedding(section)
        # check to see if srd.json exists
        if not os.path.exists('/Users/alexthe5th/PycharmProjects/rules_lawyer/docs/srd.json'):
            # if not create it
            with open('/Users/alexthe5th/PycharmProjects/rules_lawyer/docs/srd.json', 'w') as f:
                json.dump({}, f)
        # append to a json file with the text as the key and the vector as the value
        with open('/Users/alexthe5th/PycharmProjects/rules_lawyer/docs/srd.json', 'r') as f:
            doc_embeds = json.load(f)
        doc_embeds[section] = vector
        with open('/Users/alexthe5th/PycharmProjects/rules_lawyer/docs/srd.json', 'w') as f:
            json.dump(doc_embeds, f)
