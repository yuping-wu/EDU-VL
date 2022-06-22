import logging, os, shutil
import torch
import torch.nn as nn
import numpy as np
import multiprocessing

from torch import autograd
from collections import deque
from overrides import overrides

from allennlp.models.model import Model
from allennlp.modules.token_embedders.pretrained_transformer_embedder import PretrainedTransformerEmbedder
from allennlp.modules.feedforward import FeedForward
from allennlp.data.vocabulary import Vocabulary
from allennlp.nn.initializers import zero

from transformers import RobertaModel
# from transformers import BartModel # need to be changed correspondingly

from model.transformer_block import *
from model.utils import *
from model.metric import RougeScore

logger = logging.getLogger(__name__)

def run_evaluation_worker(item):
    return item.get_metric(True)

@Model.register("my_model")
class MyModel(Model):
    def __init__(self, vocab: Vocabulary,
    transformer_encoder: TransformerEncoder,
    max_token: int = 512, tmp_dir: str = 'tmp/',
    transformer_name: str = "roberta-base",
    trigram_block: bool = True, min_pred_unit: int = 3,
    max_pred_unit: int = 7, sum_unit: str = "edu"):
        super(MyModel, self).__init__(vocab)
        self.tmp_dir = tmp_dir

        self.max_token = max_token
        self.sum_unit = sum_unit

        self.transformer = RobertaModel.from_pretrained(transformer_name)
        # self.transformer = BartModel.from_pretrained(transformer_name) # need to be changed correspondingly
        self.hidden_dim = self.transformer.config.hidden_size
        if 'bert' in transformer_name:
            if self.max_token > 514:
                new_pos_embeddings = nn.Embedding(self.max_token+2, self.hidden_dim)
                new_pos_embeddings.weight.data[:514] = self.transformer.embeddings.position_embeddings.weight.data
                new_pos_embeddings.weight.data[514:] = self.transformer.embeddings.position_embeddings.weight.data[-1][None,:].repeat(self.max_token-512, 1)
                self.transformer.embeddings.position_embeddings = new_pos_embeddings
                self.transformer.embeddings.position_embeddings.num_embeddings = self.max_token+2

        self.classification = nn.Linear(self.hidden_dim, 1)
        zero(self.classification.weight)

        self.transformer_encoder = transformer_encoder

        self.dropout = nn.Dropout(p=0.1)
        self.sigmoid = nn.Sigmoid()
        self.bce_loss = nn.BCELoss(reduction='none')
        self.margin_loss = MyMarginRankingLoss(margin=0.01)

        # params for decoding
        self.trigram_block = trigram_block
        self.min_pred_unit = min_pred_unit
        self.max_pred_unit = max_pred_unit

        # params for evaluation
        for i in range(min_pred_unit, max_pred_unit+1):
            setattr(self, "rouge_{}".format(i),
            RougeScore(name="rouge_{}".format(i), cand_path=tmp_dir, ref_path=tmp_dir, path_to_valid=tmp_dir))


    def forward(self, bert_src, segs, clss, labels, oracle_idx, cand_indices, meta_field, **kwargs):

        with autograd.detect_anomaly():
            input_ids = bert_src['transformer']['token_ids']
            input_mask = bert_src['transformer']['mask']
            transformer_out = self.transformer.forward(input_ids=input_ids, attention_mask=input_mask, token_type_ids=segs.eq(-1).long())[0]
            cls_out, cls_mask = select_cls(transformer_out, clss)
            # cls_out: (batch_size, #cls, hidden_dim); cls_mask: (batch_size, #cls)
            batch_size, unit_num, hidden_dim = cls_out.shape

            # Classification layer
            output = self.sigmoid(self.classification(cls_out))
            # output: (batch_size, #cls, 1)

            loss_mask = cls_mask

            probs = output.squeeze(-1) # probs: (batch_size, #cls)
            tuned_probs = probs + (loss_mask.float() - 1) * 10 # tuned_probs: (batch_size, #cls)

            # get doc_rep
            doc_ave = torch.mean(cls_out, 1) # doc_rep: (batch_size, hidden_dim)
            doc_input = torch.cat((doc_ave.unsqueeze(1), cls_out), dim=1) # doc_input: (batch_size, #cls+1, hidden_dim)
            mask = torch.ones(batch_size).cuda(cls_out.device)
            doc_input_mask = torch.cat((mask.unsqueeze(1), cls_mask), dim=1) # doc_input_mask: (batch_size, #cls+1)
            doc_rep = self.transformer_encoder.forward(doc_input, doc_input_mask)[:, 0] # doc_rep: (batch_size, hidden_dim)

            label = labels[:, 0, :]
            label = nn.functional.relu(label) # label: (batch_size, #cls)
            # BCE loss
            bce_loss = self.bce_loss(probs, label.float())
            bce_loss *= loss_mask.float()

            ptype = meta_field[0]['type']
            if ptype == 'train':

                # get ref_rep based on gt_labels
                gt_idx = torch.where(oracle_idx>0, oracle_idx, unit_num) # gt_idx: (batch_size, #gt)
                tmp_ext_cls_out = torch.cat((cls_out, torch.zeros(batch_size, 1, hidden_dim).cuda(cls_out.device)), dim=1) # (batch_size, #cls+1, hidden_dim)
                gt_ref = tmp_ext_cls_out[torch.arange(cls_out.shape[0]).unsqueeze(-1), gt_idx] # gt_ref: (batch_size, #gt, hidden_dim)
                ref_mean = torch.mean(gt_ref, 1) # ref_mean: (batch_size, hidden_dim)
                ref_input = torch.cat((ref_mean.unsqueeze(1), gt_ref), dim=1) # ref_input: (batch_size, #gt+1, hidden_dim)
                # ref_input_mask = (oracle_idx >= -0.0001).float() # (batch_size, #gt)
                ref_input_mask = torch.cat((mask.unsqueeze(1), (oracle_idx >= -0.0001).float()), dim=1) # ref_input_mask: (batch_size, #gt+1)
                ref_rep = self.transformer_encoder.forward(ref_input, ref_input_mask)[:, 0] # ref_rep: (batch_size, hidden_dim)

                # get cand_rep based on cand_indices
                tmp_cand_idx = torch.where(cand_indices>-1, cand_indices, unit_num) # tmp_cand_idx: (batch_size, #cand=5, #sel_edu)
                cand_batch_size = tmp_cand_idx.shape[0]*tmp_cand_idx.shape[1]
                cand_idx = tmp_cand_idx.view(batch_size, -1) # cand_idx: (batch_size, #cand*#sel_edu)
                tmp_cand_input = tmp_ext_cls_out[torch.arange(batch_size).unsqueeze(-1), cand_idx] # tmp_cand_input: (batch_size, #cand*#sel_edu, hidden_dim)
                cand_cls = tmp_cand_input.view(cand_batch_size, tmp_cand_idx.shape[2], hidden_dim)# (batch_size*#cand, #sel_edu, hidden_dim)
                cand_mean = torch.mean(cand_cls, 1) # cand_mean: (batch_size*#cand, hidden_dim)
                cand_input = torch.cat((cand_mean.unsqueeze(1), cand_cls), dim=1) # cand_input: (batch_size*#cand, #sel_edu+1, hidden_dim)
                cand_cls_mask = torch.ones(cand_batch_size).cuda(cls_out.device).unsqueeze(-1) # cand_cls_mask: (batch_size*#cand, 1)
                cand_indices_mask = (cand_indices >= -0.0001).float().view(cand_batch_size, -1)
                cand_input_mask = torch.cat((cand_cls_mask, cand_indices_mask), dim=1) # cand_input_mask: (batch_size*#cand, #sel_edu+1)
                cand_rep = self.transformer_encoder.forward(cand_input, cand_input_mask)[:, 0].view(batch_size, tmp_cand_idx.shape[1], hidden_dim) # cand_rep: (batch_size*#cand, hidden_dim) > (batch_size, #cand, hidden_dim)

                # claculate margin rank loss
                ref_sim = torch.cosine_similarity(ref_rep, doc_rep, dim=-1)
                # get candidate score
                doc_emb = doc_rep.unsqueeze(1).expand_as(cand_rep)
                cand_sim = torch.cosine_similarity(cand_rep, doc_emb, dim=-1) # [batch_size, candidate_num]
                # pass into marginrankloss function
                margin_loss = self.margin_loss.get_loss(cand_sim, ref_sim)
                # margin_loss: (batch_size)

                loss = torch.sum(bce_loss) + 100*torch.sum(margin_loss)

            else:
                # claculate loss
                loss = torch.sum(bce_loss)

                tuned_probs = tuned_probs.cpu().data.numpy()

                # decoding process
                for b in range(batch_size):
                    data = {}
                    doc_id = meta_field[b]['doc_id']
                    tgt_txt = meta_field[b]['ref']
                    list_of_words = meta_field[b]['words']
                    data['doc_id'] = doc_id
                    data['label'] = meta_field[b]['label']
                    prob_dist = tuned_probs[b]
                    data['probs'] = prob_dist.tolist()
                    this_cls = cls_out[b] # this_cls: (#cls, hidden_dim)
                    this_mask = cls_mask[b] # this_mask: (#cls)
                    this_doc = doc_rep[b] # this_doc: (hidden_dim)
                    this_preds, this_sim_scores = [], []
                    # this_t_mask = torch.cat((torch.tensor([1.0]).cuda(this_cls.device), this_mask), dim=0) # this_t_mask: (#cls+1)
                    for l in range(self.min_pred_unit, self.max_pred_unit):
                        pred, sel_idx = decoding(prob_dist, self.trigram_block, l, list_of_words)
                        pred = [x for x in pred if len(x)>1]
                        getattr(self, 'rouge_{}'.format(l))(pred="<q>".join(pred), ref=tgt_txt, id=doc_id)
                        data[str(l)+'_sel_idx'] = sel_idx

                        this_preds.append(pred)

                        # get cosine_similarity
                        cur_idx = torch.tensor(sel_idx).cuda(this_cls.device) # cur_idx: (l)
                        cur_sel_cls = torch.index_select(this_cls, 0, cur_idx) # cur_sel_cls: (l, hidden_dim)
                        # cur_cand_mean = torch.mean(cur_sel_cls, 0) # (hidden_dim)
                        cur_cand_mean = torch.mean(cur_sel_cls, 0) # (hidden_dim)
                        # cur_label = torch.zeros(unit_num).cuda(this_cls.device) # cur_label: (#cls)
                        # cur_label[cur_idx] = 1.0
                        # tmp_cur_cand = torch.mul(this_cls, cur_label.unsqueeze(-1)) # tmp_cur_cand: (#cls, hidden_dim)
                        cur_cand_input = torch.cat((cur_cand_mean.unsqueeze(0), cur_sel_cls), dim=0) # (l+1, hidden_dim)
                        this_t_mask = torch.ones(len(sel_idx)+1).cuda(this_cls.device) # (l+1)
                        cur_cand_rep = self.transformer_encoder.forward(cur_cand_input.unsqueeze(0), this_t_mask.unsqueeze(0))[0, 0] # cur_cand_rep: (hidden_dim)


                        cur_sim = torch.cosine_similarity(cur_cand_rep, this_doc, dim=-1)
                        this_sim_scores.append(cur_sim.item())

                    max_sim_l = this_sim_scores.index(max(this_sim_scores))
                    getattr(self, 'rouge_{}'.format(self.max_pred_unit))(pred="<q>".join(this_preds[max_sim_l]), ref=tgt_txt, id=doc_id)
                    data['cos_sim_l'] = this_sim_scores

                    # if ptype == 'test':
                    #     try:
                    #         with open(ptype+'_predictions.txt','a') as f:
                    #             print(data, file=f)
                    #     except Exception as e:
                    #         logger.warning("Error {} occured when trying to write data into predictions.txt file".format(e))


            output = {'probs': probs, 'loss': loss}

        return output


    def get_metrics(self, reset: bool = False):
        rouge_scores = {}
        if reset:
            # calculate rouge score via calling method get_metric for RougeScore
            objects = [(getattr(self, 'rouge_{}'.format(l)),) for l in range(self.min_pred_unit, self.max_pred_unit+1)]
            pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
            results = pool.starmap(run_evaluation_worker, objects)
            pool.close()
            pool.join()

            # reset data stroed in RougeScore for next batch
            for l in range(self.min_pred_unit, self.max_pred_unit+1):
                getattr(self, 'rouge_{}'.format(l)).reset()
            for r in results:
                rouge_scores = {**rouge_scores, **r}
        else:
            for l in range(self.min_pred_unit, self.max_pred_unit+1):
                result = getattr(self, 'rouge_{}'.format(l)).get_metric(reset)
                rouge_scores = {**rouge_scores, **result}
        best_name, length = process_rouge(rouge_scores)

        if reset:
            print('-->Best Key: {}'.format(best_name))

        metrics = {'R_1': rouge_scores['{}_1'.format(best_name)],
        'R_2': rouge_scores['{}_2'.format(best_name)],
        'R_L': rouge_scores['{}_L'.format(best_name)],
        'L': length}

        return metrics
