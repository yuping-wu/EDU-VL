import re, ast, os, json, shutil, multiprocessing
from multiprocessing.pool import Pool
import numpy as np
from metric import test_rouge, rouge_results_to_str
# from utils import decoding

REMAP = {"-lrb-": "(", "-rrb-": ")", "-lcb-": "{", "-rcb-": "}",
         "-lsb-": "[", "-rsb-": "]", "``": '"', "''": '"'}

def clean(x):
    return re.sub(
        r"-lrb-|-rrb-|-lcb-|-rcb-|-lsb-|-rsb-|``|''",
        lambda m: REMAP.get(m.group()), x)

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


def get_rouge_score(cand, ref, tmp_path):
    os.mkdir(tmp_path)
    tmp_dir = os.path.join(tmp_path, 'tmp')
    os.mkdir(tmp_dir)
    with open(tmp_path+'/cand.txt', 'w', encoding='utf-8') as candf:
        candf.write(cand)
    with open(tmp_path+'/ref.txt', 'w', encoding='utf-8') as reff:
        reff.write(ref)
    scores = test_rouge(tmp_dir, os.path.join(tmp_path, 'cand.txt'), os.path.join(tmp_path, 'ref.txt'))
    result = (scores['rouge_1_f_score'] * 100 + scores['rouge_2_f_score'] * 100 + scores['rouge_l_f_score'] * 100) / 3
    shutil.rmtree(tmp_path)
    return result


# add candidate summaries into dataset
def add_cand(params):
    data, cands, min_pred, max_pred = params
    doc_id = data['doc_id']
    # get edu_text
    _, edu_txt = get_text(data['sent'], data['disco_span'])
    try:
        cand_indices, cand_score = [], []
        for num in range(min_pred, max_pred+1):
            cand_edus = []
            sel_idx = sorted([int(p) for p in cands[str(num)+'_sel_idx']])
            cand_indices.append(sel_idx)
            for i in sel_idx:
                cand_edus.append(edu_txt[i])
            cand_score.append(get_rouge_score("<q>".join([" ".join(edu) for edu in cand_edus]), "<q>".join(data['tgt_list_str']), os.path.join('tmp', doc_id.split('.')[0]+'_pred_'+str(num))))
        assert len(cand_indices) == 5
        assert len(cand_score) == 5
        data['cand_indices'], data['cand_score'] = zip(*sorted(zip(cand_indices, cand_score), key=lambda x: x[1], reverse=True))
        return data
    except:
        print('Doc {} gets error when processing candidate summaries.'.format(doc_id))
        return {}
    # for num in range(min_pred, max_pred+1):
    #     try:
    #         sel_idx = sorted([int(p) for p in cands[str(num)+'_sel_idx']])
    #         cand_indices.append(sel_idx)
    #     except:
    #         print('Error for processing candidate indices on doc {} with prediction length {}.'.format(doc_id, num))
    #         continue
    #
    #     cand_edus = []
    #     for i in sel_idx:
    #         cand_edus.append(edu_txt[i])
    #
    #     try:
    #         cand_score.append(get_rouge_score("<q>".join([" ".join(edu) for edu in cand_edus]), "<q>".join(data['tgt_list_str']), os.path.join('tmp', doc_id.split('.')[0]+'_pred_'+str(num))))
    #     except:
    #         print('Error for calculating rouge score on doc {} with prediction length {}.'.format(doc_id, num))
    # try:
    #     assert len(cand_indices) == 5
    #     assert len(cand_score) == 5
    # except:
    #     print('Doc {} faild to get 5 candidates.'.format(doc_id))
    #     continue
    # data['cand_indices'], data['cand_score'] = zip(*sorted(zip(cand_indices, cand_score), key=lambda x: x[1], reverse=True))
    #
    # return data


if __name__ == '__main__':
    dataset_name = 'xsum'
    dataset_path = 'Datasets/XSUM'
    min_pred, max_pred = 3, 7

    # output_path = 'processed'
    output_path = 'Datasets/XSUM_new'
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    predictions_path = 'tmp'

    for split in ['train', 'valid', 'test']:
        print('-'*20, 'Start dealing with ', split, ' dataset','-'*20)
        new_dataset_path = os.path.join(output_path, split)
        if not os.path.exists(new_dataset_path):
            os.mkdir(new_dataset_path)

        # read prediction file
        all_preds = {}
        print("Start reading prediction file for split {}!".format(split))
        pred_file = os.path.join(predictions_path, dataset_name+'_'+split+'_predictions.txt')
        with open(pred_file, 'r') as predf:
            for i, l in enumerate(predf):
                data = ast.literal_eval(l)
                id = data['doc_id']
                all_preds[id] = {}
                for num in range(min_pred, max_pred+1):
                    all_preds[id][str(num)+'_sel_idx'] = data[str(num)+'_sel_idx']

        # read dataset file individually
        split_path = os.path.join(dataset_path, split)
        files = os.listdir(split_path)
        files = [f for f in files if f.endswith('json')]
        for f in files:
            print('Start dedling with file ', f)
            new_dataset, todo_list = [], []
            dataset = json.load(open(os.path.join(split_path, f)))
            for d in dataset:
                todo_list.append((d, all_preds[d['doc_id']], min_pred, max_pred))

            # deploy multiprocessing
            _p = Pool(multiprocessing.cpu_count())
            for r in _p.imap_unordered(add_cand, todo_list):
                if bool(r):
                    new_dataset.append(r)
            _p.close()
            _p.join()

            # save new_dataset
            with open(os.path.join(new_dataset_path, f), 'w') as outf:
                json.dump(new_dataset, outf)
