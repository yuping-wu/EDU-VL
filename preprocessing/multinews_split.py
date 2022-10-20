import os
from tqdm import tqdm
from nltk import tokenize

train_src = 'MultiNews/train.txt.src.tokenized.fixed.cleaned.final.truncated'
train_tgt = 'MultiNews/train.txt.tgt.tokenized.fixed.cleaned.final.truncated'
val_src = 'MultiNews/val.txt.src.tokenized.fixed.cleaned.final.truncated'
val_tgt = 'MultiNews/val.txt.tgt.tokenized.fixed.cleaned.final.truncated'
test_src = 'MultiNews/test.txt.src.tokenized.fixed.cleaned.final.truncated'
test_tgt = 'MultiNews/test.txt.tgt.tokenized.fixed.cleaned.final.truncated'

# The path where the articles are to be saved
doc_path = "MultiNews/raw_doc"
sum_path = "MultiNews/sum"
url_path = "MultiNews/url"
if not os.path.exists(doc_path):
    os.makedirs(doc_path)
if not os.path.exists(sum_path):
    os.makedirs(sum_path)
if not os.path.exists(url_path):
    os.makedirs(url_path)

# read train/val/test src/tgt files
def read_write(src_file, tgt_file, doc_path, sum_path, url_path, type, count):

    with open(src_file, 'r', encoding='utf8') as sf:
        srcs = sf.read().splitlines()
    with open(tgt_file, 'r', encoding='utf8') as tf:
        tgts = tf.read().splitlines()
    assert len(srcs) == len(tgts)

    mapping = []
    print("Now handeling with dataset ", type)
    for i in tqdm(range(len(srcs))):
        src, tgt = srcs[i], tgts[i]
        src_sents = tokenize.sent_tokenize(src)
        src_sents = [t for t in src_sents if len(t) > 1]
        tgt_sents = tokenize.sent_tokenize(tgt)
        tgt_sents = [t for t in tgt_sents if len(t) > 1]
        if len(src_sents) == 0:
            continue

        count += 1
        doc = str(count) + '.doc'
        sum = str(count) + '.sum'
        mapping.append(str(count))

        with open(os.path.join(doc_path, doc), 'w', encoding='utf8') as df:
            df.write('\n'.join(src_sents))
        with open(os.path.join(sum_path, sum), 'w', encoding='utf8') as sf:
            sf.write('\n'.join(tgt_sents))

    url_name = 'mapping_' + type + '.txt'
    with open(os.path.join(url_path, url_name), 'w', encoding='utf8') as uf:
        uf.write('\n'.join(mapping))

    return count

c = 0
c = read_write(train_src, train_tgt, doc_path, sum_path, url_path, 'train', c)
c = read_write(val_src, val_tgt, doc_path, sum_path, url_path, 'valid', c)
c = read_write(test_src, test_tgt, doc_path, sum_path, url_path, 'test', c)
print("Total number of doc-sum pairs: ", c)
