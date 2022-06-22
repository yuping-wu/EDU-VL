import sys, os, json, logging, re, itertools
import torch
import torch.nn as nn
import numpy as np

from collections import deque
from typing import List

logger = logging.getLogger(__name__)

"""
DataReader utils
"""

REMAP = {"-lrb-": "(", "-rrb-": ")", "-lcb-": "{", "-rcb-": "}",
         "-lsb-": "[", "-rsb-": "]", "``": '"', "''": '"'}


def clean(x):
    return re.sub(
        r"-lrb-|-rrb-|-lcb-|-rcb-|-lsb-|-rsb-|``|''",
        lambda m: REMAP.get(m.group()), x)

def get_type(path):
    if 'train' in path:
        return 'train'
    elif 'valid' in path or 'dev' in path:
        return 'valid'
    elif 'test' in path:
        return 'test'
    else:
        return 'unk'

def label_filter(labs):
    rt_list = []
    cur_min_cnt = 100
    for l in labs:
        s = sum(l)
        if s < cur_min_cnt:
            cur_min_cnt = s
            rt_list.insert(0, l)
        else:
            rt_list.append(l)
    return rt_list

def get_text(sents_list, span_list):
    assert len(sents_list) == len(span_list)
    sents, edus = [], []
    for idx in range(len(sents_list)):
        sent_words = [clean(w.lower()) for w in sents_list[idx]['tokens']]
        sents.append(sent_words)
        span = span_list[idx]
        for sidx, eidx in span:
            edus.append([sent_words[j] for j in range(sidx, eidx+1)])
    assert len(sum(sents, [])) == len(sum(edus, []))
    return sents, edus

""" Copied from nlpyang"""
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

stop = set(stopwords.words('english'))

def _get_ngrams(n, text):
    """Calcualtes n-grams.

    Args:
      n: which n-grams to calculate
      text: An array of tokens

    Returns:
      A set of n-grams
    """
    ngram_set = set()
    text_length = len(text)
    max_index_ngram_start = text_length - n
    for i in range(max_index_ngram_start + 1):
        ngram_set.add(tuple(text[i:i + n]))
    return ngram_set

def _get_word_ngrams(n, sentences, rm_stop_unigram=True):
    """Calculates word n-grams for multiple sentences.
    """
    assert len(sentences) > 0
    assert n > 0

    # words = _split_into_words(sentences)

    words = sum(sentences, [])
    if rm_stop_unigram and n == 1:
        words = set(words)
        words = list(words.difference(stop))
        # words = [w for w in words if w not in stop]
    return _get_ngrams(n, words)

def cal_rouge(evaluated_ngrams, reference_ngrams):
    reference_count = len(reference_ngrams)
    evaluated_count = len(evaluated_ngrams)

    overlapping_ngrams = evaluated_ngrams.intersection(reference_ngrams)
    overlapping_count = len(overlapping_ngrams)

    if evaluated_count == 0:
        precision = 0.0
    else:
        precision = overlapping_count / evaluated_count

    if reference_count == 0:
        recall = 0.0
    else:
        recall = overlapping_count / reference_count

    f1_score = 2.0 * ((precision * recall) / (precision + recall + 1e-8))
    return {"f": f1_score, "p": precision, "r": recall}

def greedy_label_selection(doc_sent_list: List[List[str]], abstract_sent_list: List[List[str]], summary_size):
    def _rouge_clean(s):
        return re.sub(r'[^a-zA-Z0-9 ]', '', s)

    max_rouge = 0.0
    abstract = sum(abstract_sent_list, [])
    abstract = _rouge_clean(" ".join(abstract)).split()
    sents = [_rouge_clean(" ".join(s)).split() for s in doc_sent_list]
    evaluated_1grams = [_get_word_ngrams(1, [sent]) for sent in sents]
    reference_1grams = _get_word_ngrams(1, [abstract])
    evaluated_2grams = [_get_word_ngrams(2, [sent]) for sent in sents]
    reference_2grams = _get_word_ngrams(2, [abstract])

    selected = []
    for s in range(summary_size):
        cur_max_rouge = max_rouge
        cur_id = -1
        for i in range(len(sents)):
            if (i in selected):
                continue
            c = selected + [i]
            candidates_1 = [evaluated_1grams[idx] for idx in c]
            candidates_1 = set.union(*map(set, candidates_1))
            candidates_2 = [evaluated_2grams[idx] for idx in c]
            candidates_2 = set.union(*map(set, candidates_2))
            rouge_1 = cal_rouge(candidates_1, reference_1grams)['f']
            rouge_2 = cal_rouge(candidates_2, reference_2grams)['f']
            rouge_score = rouge_1 + rouge_2
            if rouge_score > cur_max_rouge:
                cur_max_rouge = rouge_score
                cur_id = i
        if (cur_id == -1):
            return selected
        selected.append(cur_id)
        max_rouge = cur_max_rouge
    return sorted(selected)

def edu_rouge(doc_sent: List[str], abstract_sent_list: List[List[str]]):
    def _rouge_clean(s):
        return re.sub(r'[^a-zA-Z0-9 ]', '', s)

    abstract = sum(abstract_sent_list, [])
    abstract = _rouge_clean(" ".join(abstract)).split()
    doc_text = _rouge_clean(" ".join(doc_sent)).split()
    evaluated_1grams = _get_word_ngrams(1, [doc_text])
    reference_1grams = _get_word_ngrams(1, [abstract])
    evaluated_2grams = _get_word_ngrams(2, [doc_text])
    reference_2grams = _get_word_ngrams(2, [abstract])

    candidates_1 = set(evaluated_1grams)
    candidates_2 = set(evaluated_2grams)
    rouge_1 = cal_rouge(candidates_1, reference_1grams)['f']
    rouge_2 = cal_rouge(candidates_2, reference_2grams)['f']
    rouge_score = (rouge_1 + rouge_2) / 2
    rouge_score = rouge_1

    return rouge_score

"""
Model utils
"""
def select_cls(top_vec, clss):
    assert top_vec.shape[0] == clss.shape[0] # check if batch_size equals
    batch_size, num_word, word_hdim = top_vec.shape
    num_cls = clss.shape[1]
    cls_mask = (clss >= -0.0001).float() # >> [[1.0, 1.0, 1.0, (num_cls), 0.0], (batch_size), [1.0, 1.0, ... (num_cls)]]
    clss_non_neg = torch.nn.functional.relu(clss).long()

    # flatten 3d top_vec tensor to 2d / concatenate all word reps in docs within same batch
    # flat_top_vec >> [[word1_doc1], [word2_doc1], ..., [word1_docn], [word2_docn]]
    flat_top_vec = top_vec.view(batch_size*num_word, word_hdim)

    # flatten 2d clss tensor to 1d / identify index of word [CLS] in flat_top_vec
    # flat_clss_idx >> [0, cls2_doc1, cls3_doc1, ..., 0+(num_cls*batch_size), cls2_docn]
    bias = torch.arange(start=0, end=batch_size, dtype=torch.long, device=clss_non_neg.device, requires_grad=False)*num_word # >> [0, 1*num_word, ..., batch_size*num_word]
    bias = bias.view(-1, 1) # >> [[0], [1*num_word], ..., [batch_size*num_word]]
    bias = bias.repeat(1, num_cls).view(-1) # >> [0, (repeat num_cls times), 0, ..., batch_size*num_word, (repeat num_cls times), batch_size*num_word]
    flat_clss_non_neg = clss_non_neg.view(-1) # >> [cls1_doc1, cls2_doc1, ..., cls1_docn, cls2_docn]
    flat_clss_idx = (flat_clss_non_neg + bias).long()

    flat_cls_rep = torch.index_select(flat_top_vec, 0, flat_clss_idx)
    cls_rep = flat_cls_rep.view(batch_size, num_cls, -1)
    cls_rep = cls_rep * cls_mask.unsqueeze(-1)

    return cls_rep, cls_mask

def extract_ngrams(str, n=3):
    #input: list of strings; output: set of trigram
    trigrams = []
    for s in str:
        tokens = s.split(" ")
        if len(tokens) < n:
            continue
        for i in range(len(tokens) - n + 1):
            trigrams.append("_".join(tokens[i:i+n]))
    return set(trigrams)

def decoding(probs=[], trigram=True, pred_unit=7, doc=[]):
    # probs = probs.cpu().data.numpy()

    if pred_unit >= len(doc):
        cand = [" ".join(e) for e in doc]
        return cand, [i for i in range(len(doc))]

    ranks = np.argsort(-probs) # return index of descending prob
    candidate = []
    cand_idx = []
    cand = []
    q = deque(ranks)

    if trigram:
        while len(q) > 0 and (len(candidate) < pred_unit):
            idx = q.popleft()
            try:
                edu = [" ".join(doc[idx])]
                trigram_cand = extract_ngrams(candidate, n=3)
                trigram_edu = extract_ngrams(edu, n=3)
                if trigram_cand.isdisjoint(trigram_edu):
                    candidate += edu
                    cand_idx.append(idx)
            except IndexError:
                logger.warning("IndexError in index {} with probs {} \nfor doc {}".format(idx, probs, doc))
                break
            except Exception as e:
                logger.warning("{} occured in index {} with probs {} \nfor doc {}".format(type(e), idx, probs, doc))
                break
    else:
        try:
            for i in range(pred_unit):
                # candidate += [" ".join(doc[i])] # list of strings
                cand_idx.append(ranks[i])
        except IndexError:
            logger.warning("IndexError occured when pred_unit is larger than length of variable ranks")
        except Exception as e:
            logger.warning("{} occured when extracting index from variable ranks".format(type(e)))
    cand_idx.sort()
    try:
        for i in cand_idx:
            cand.append(" ".join(doc[i]))
    except IndexError:
        logger.warning("IndexError for cand_idx {} \non doc{}".format(cand_idx, doc))
    except Exception as e:
        logger.warning("{} for cand_idx {} \non doc{}".format(type(e), cand_idx, doc))
    return cand, cand_idx

class MyMarginRankingLoss(nn.Module):

    def __init__(self, margin, score=None, summary_score=None):
        super(MyMarginRankingLoss, self).__init__()
        # self._init_param_map(score=score, summary_score=summary_score)
        self.margin = margin
        self.loss_func = torch.nn.MarginRankingLoss(margin)

    def get_loss(self, score, summary_score):

        # equivalent to initializing TotalLoss to 0
        # here is to avoid that some special samples will not go into the following for loop
        ones = torch.ones(score.size()).cuda(score.device)
        loss_func = torch.nn.MarginRankingLoss(0.0)
        TotalLoss = loss_func(score, score, ones)

        # candidate loss
        n = score.size(1)
        for i in range(1, n):
            pos_score = score[:, :-i]
            neg_score = score[:, i:]
            pos_score = pos_score.contiguous().view(-1)
            neg_score = neg_score.contiguous().view(-1)
            ones = torch.ones(pos_score.size()).cuda(score.device)
            loss_func = torch.nn.MarginRankingLoss(self.margin * i)
            TotalLoss += loss_func(pos_score, neg_score, ones)

        # gold summary loss
        pos_score = summary_score.unsqueeze(-1).expand_as(score)
        neg_score = score
        pos_score = pos_score.contiguous().view(-1)
        neg_score = neg_score.contiguous().view(-1)
        ones = torch.ones(pos_score.size()).cuda(score.device)
        loss_func = torch.nn.MarginRankingLoss(0.0)
        TotalLoss += loss_func(pos_score, neg_score, ones)

        return TotalLoss

"""
Process Rouge Score Results
"""
def process_rouge(results):
    best_key, best_value = "", -1
    for k, v in results.items():
        if k.endswith("_1"):
            if v > best_value:
                best_value = v
                best_key = k
    best_name = best_key.split("_")
    length = int(best_name[1])
    return "_".join(best_name[:2]), length
