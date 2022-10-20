import pandas as pd
import os
from os.path import join
import re
from tqdm import tqdm

# read data from the csv file
data = pd.read_csv('wikihowAll.csv')
# data = data.astype(str)
rows, columns = data.shape

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
# go over the all the articles in the data file
for row in tqdm(range(rows)):
    abstract = data['headline'][row]      # headline is the column representing the summary sentences
    article = data['text'][row]           # text is the column representing the article
    filename = data['title'][row]

    if pd.isnull(abstract) or pd.isnull(article) or pd.isnull(filename):
        continue
    abstract = str(abstract)
    article = str(article)
    filename = str(filename)
    #  a threshold is used to remove short articles with long summaries as well as articles with no summary
    if len(abstract) < (0.75*len(article)):
        # remove extra commas in abstracts
        abstract = abstract.replace(".,",".")
        summary = abstract.splitlines()
        summary = [l for l in summary if len(l) > 1]
        if len(summary) == 0:
            continue
        # abstract = abstract.encode('utf-8')
        # remove extra commas in articles
        article = re.sub(r'[.]+[\n]+[,]',".\n", article)
        article = article.replace('\n\n\n', '\n')
        article = article.replace('\n\n', '\n')
        document = article.splitlines()
        document = [l for l in document if len(l) > 1]
        if len(document) == 0:
            continue
        # article = article.encode('utf-8')

        # file names are created using the alphanumeric charachters from the article titles.
        # they are stored in a separate text file.
        filename = "".join(x for x in filename if x.isalnum())
        doc = filename + '.doc'
        sum = filename + '.sum'
        # filename = filename.encode('utf-8')
        sum_length.append(str(len(summary)))
        doc_length.append(str(len(document)))
        file_name.append(filename)
        with open(join(doc_path, doc), 'w', encoding='utf-8') as f:
            f.write('\n'.join(document))
        with open(join(sum_path, sum), 'w', encoding='utf-8') as f:
            f.write('\n'.join(summary))
with open('check/summary_length.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(sum_length))
with open('check/document_length.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(doc_length))
with open('check/file_name.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(file_name))
