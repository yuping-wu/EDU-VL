from os import listdir
from os.path import isfile, join
import re
from tqdm import tqdm

original_dir = 'bbc-summary-data'
output_doc_dir = 'xsum_doc'
output_sum_dir = 'xsum_sum'

files = [f for f in listdir(original_dir) if isfile(join(original_dir, f))]
sum_length = []
doc_length = []
file_name = []
for fname in tqdm(files):
    lines = open(join(original_dir, fname), 'r').read().splitlines()
    text = [l for l in lines if len(l) > 1]
    file = fname.split(".")[0]
    doc = file+'.doc'
    sum = file+'.sum'
    summary_start = text.index('[SN]FIRST-SENTENCE[SN]')
    summary_end = text.index('[SN]RESTBODY[SN]')
    summary = text[summary_start+1:summary_end]
    document = text[summary_end+1:]
    sum_length.append(str(len(summary)))
    doc_length.append(str(len(document)))
    file_name.append(fname)
    with open(join(output_doc_dir, doc), 'w', encoding='utf-8') as f:
        f.write('\n'.join(document))
    with open(join(output_sum_dir, sum), 'w', encoding='utf-8') as f:
        f.write('\n'.join(summary))
with open('check/summary_length.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(sum_length))
with open('check/document_length.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(doc_length))
with open('check/file_name.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(file_name))
