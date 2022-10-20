import json
import os
# from os.path import isfile, join
import re
from tqdm import tqdm

# Read entire file
posts = []
with open('tifu_all_tokenized_and_filtered.json', 'r') as fp:
    for l in fp:
        line = json.loads(l)
        if line['tldr'] is not None:
            posts.append(line)

# The path where the articles are to be saved
doc_path = "raw_doc"
sum_path = "sum"
check = "check"
if not os.path.exists(doc_path):
    os.makedirs(doc_path)
if not os.path.exists(sum_path):
    os.makedirs(sum_path)
if not os.path.exists(check):
    os.makedirs(check)

sum_length = []
doc_length = []
file_name = []
# process for each post
for i in tqdm(range(len(posts))):
    post = posts[i]
    tldr = post['tldr']
    text = post['selftext_without_tldr']
    article = text.replace('\n\n', '\n')
    document = article.splitlines()
    document = [l for l in document if len(l) > 1]
    summary = [tldr]
    summary = [l for l in summary if len(l) > 1]
    if len(document) == 0:
        continue

    doc = str(i)+'.doc'
    sum = str(i)+'.sum'
    sum_length.append(str(len(summary)))
    doc_length.append(str(len(document)))
    file_name.append(str(i))
    with open(os.path.join(doc_path, doc), 'w', encoding='utf-8') as f:
        f.write('\n'.join(document))
    with open(os.path.join(sum_path, sum), 'w', encoding='utf-8') as f:
        f.write('\n'.join(summary))
with open('check/summary_length.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(sum_length))
with open('check/document_length.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(doc_length))
with open('check/file_name.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(file_name))

# Json entries
# print(posts[50000].keys())
# [u'title_tokenized',
#  u'permalink',
#  u'title', # short summary
#  u'url',
#  u'num_comments',
#  u'tldr',  # (optional) # long summary
#  u'created_utc',
#  u'trimmed_title_tokenized',
#  u'ups',
#  u'selftext_html',
#  u'score',
#  u'upvote_ratio',
#  u'tldr_tokenized',  # (optional)
#  u'selftext', # article
#  u'trimmed_title',
#  u'selftext_without_tldr_tokenized',
#  u'id',
#  u'selftext_without_tldr' # article
# ]

# splitting train, valid and test dataset (random way)
from sklearn.model_selection import train_test_split
import os

filename = open('check/file_name.txt', 'r').read().splitlines()
train, test = train_test_split(filename, test_size = 0.05, random_state=1)
train, val = train_test_split(train, test_size = 0.05, random_state=1)
print("Length of train:", len(train))
print("Length of valid:", len(val))
print("Length of test:", len(test))

unique_fn = set(filename)
all = train + test + val
uniqs = set(all)
print("Original length of filenames:", len(unique_fn))
print("Length of filenames after splitting:", len(uniqs))

if not os.path.exists('urls_reddit'):
    os.makedirs('urls_reddit')

with open('urls_reddit/mapping_train.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(train))
with open('urls_reddit/mapping_valid.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(val))
with open('urls_reddit/mapping_test.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(test))
