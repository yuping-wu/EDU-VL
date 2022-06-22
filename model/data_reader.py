import os, random, json
import torch
import logging
import numpy as np

from overrides import overrides
from typing import Dict

from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.token_indexers.pretrained_transformer_indexer import PretrainedTransformerIndexer
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers.token_class import Token
from allennlp.data.tokenizers.tokenizer import Tokenizer
from allennlp.data.fields import *
from allennlp.data.instance import Instance

from transformers import RobertaTokenizer
# from transformers import BartTokenizer # need to be changed correspondingly

from model.utils import get_type, label_filter, greedy_label_selection, clean, get_text

logger = logging.getLogger(__name__)

@DatasetReader.register("my_dataset_reader")
class MyDatasetReader(DatasetReader):
    def __init__(self,
    seed: int = 123, dataset_name: str = "unk",
    token_indexers: Dict[str, TokenIndexer] = PretrainedTransformerIndexer(model_name="roberta-base", namespace='tokens'),
    # token_indexers: Dict[str, TokenIndexer] = PretrainedTransformerIndexer(model_name="facebook/bart-base", namespace='tokens'), # need to be changed correspondingly
    max_token: int = 512, sum_unit: str = "edu", ground_truth_number: int = 8):
        super(MyDatasetReader, self).__init__()
        self.seed = seed
        self.dataset_name = dataset_name
        self.token_indexer = token_indexers or {'tokens': SingleIdTokenIndexer()}
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base', do_lowercase=True)
        # self.tokenizer = BartTokenizer.from_pretrained('facebook/bart-base', do_lowercase=True) # need to be changed correspondingly
        self.max_token = max_token
        self.sum_unit = sum_unit
        self.gt_num = ground_truth_number


    def _read(self, file_path):
        """
        Main entrance of calling dataset loader. Load each data from .pt file and pass to read()

        Args:
            file_path: path to datasets
        Return:
            Data in dataset
        """
        type = get_type(file_path)
        files = os.listdir(file_path)
        files = [f for f in files if f.endswith("json")]
        if type == 'train':
            random.seed(self.seed)
            random.shuffle(files)
        print("Start loading %s %s dataset, number of .json files: %d" % (self.dataset_name, type, len(files)))
        logger.info("Start loading %s %s dataset, number of .json files: %d" % (self.dataset_name, type, len(files)))
        for f in files:
            dataset = json.load(open(os.path.join(file_path, f)))
            print("Loading dataset from %s, number of examples: %d" % (f, len(dataset)))
            logger.info("Loading dataset from %s, number of examples: %d" % (f, len(dataset)))
            for d in dataset:
                yield self.text_to_instance(d, type)

    @overrides
    def text_to_instance(self, data, type):
        """
        Process single data in dataset.

        Args:
            data: single data in dataset
        Return:
            Data after processed.
        """
        # load elements in data
        doc_id, tgt_list_str, tgt_tok_list_list_str = data['doc_id'], data['tgt_list_str'], data['tgt_tok_list_list_str']
        sent_txt, edu_txt = get_text(data['sent'], data['disco_span'])

        # preprocessing input text: tokenization, truncation
        if self.sum_unit == "edu":
            words, tokens, segs, clss, segment_num = self.token_process(edu_txt)
            oracle_ids = greedy_label_selection(words, tgt_tok_list_list_str, self.gt_num)
        else:
            words, tokens, segs, clss, segment_num = self.token_process(sent_txt)
            oracle_ids = greedy_label_selection(words, tgt_tok_list_list_str, self.gt_num)

        cand_indices = []
        for ele in data['cand_indices']:
            sort_ele = [cidx for cidx in sorted(ele) if cidx<=segment_num]
            cand_indices.append(sort_ele+[-1]*(10-len(sort_ele))) # need to be changed correspondingly
        assert len(cand_indices) == 5 # need to be changed correspondingly

        # preparing ground truth label for input text
        gt_label = [0] * segment_num
        for l in oracle_ids:
            gt_label[l] = 1
        gt_labels = [gt_label]

        # converting to fields in AllenNLP
        bert_tokens = [Token(text=tokens[i], idx=i) for i in range(len(tokens))]
        bert_src = TextField(bert_tokens, self.token_indexer)
        segs = ArrayField(np.asarray(segs), padding_value=0, dtype=np.int)
        clss = ArrayField(np.asarray(clss), padding_value=-1, dtype=np.int)
        label = label_filter(gt_labels)
        labels = ArrayField(np.asarray(label), padding_value=-1, dtype=np.int)
        gt_idx = ArrayField(np.asarray(sorted(oracle_ids)), padding_value=-1, dtype=np.int)
        cand_idxs = ArrayField(np.asarray(cand_indices), padding_value=-1, dtype=np.int)

        meta_field = MetadataField({"doc_id": doc_id, "dataset_name": self.dataset_name, "type": type,
        "words": words, "ref": "<q>".join(tgt_list_str), "label": label})

        fields = {"bert_src": bert_src, "segs": segs, "clss": clss, "labels": labels,  "oracle_idx": gt_idx, "cand_indices": cand_idxs, "meta_field": meta_field}

        return Instance(fields)

    def token_process(self, txt):
        """
        Process input text, i.e., tokenize

        Args:
            txt: list of list storing tokens for each segment

        Returns:
            words: list of all valid words
            tokens: list of valid tokens
            segs: list of segmentation indicators
            clss: list of index of [CLS] token index in tokens
            segment_num: number of valid segments
        """
        words, tokens, segs, clss = [], [], [], []
        segment_num = 0
        for segment in txt:
            tmp_text = ' '.join(segment)
            tmp_tokens = ["<s>"] + self.tokenizer.tokenize(tmp_text) + ["</s>"]
            if len(tokens) + len(tmp_tokens) <= self.max_token:
                clss.append(len(tokens))
                tokens += tmp_tokens
                words.append(segment)
                segs += [segment_num%2] * len(tmp_tokens)
                segment_num += 1
            else:
                break
        assert len(clss) == segment_num
        return words, tokens, segs, clss, segment_num



if __name__ == '__main__':
    """
    Demonstration
    """
    dataset_reader = MyDatasetReader(sum_unit='edu')
    x = dataset_reader._read("Datasets/Test")
