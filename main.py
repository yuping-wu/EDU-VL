import argparse
import os, tempfile

from allennlp.common import Params
from allennlp.commands.train import train_model

from model.data_reader import MyDatasetReader
from model.model import MyModel
from model.metric import RougeScore
from model.utils import *

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser( )
    parser.add_argument("finetune", type=str2bool, nargs='?',const=True,default=False)
    args = parser.parse_args()
    if os.path.isdir('text_summarization/Model'):
        root = "text_summarization/Model"
    else:
        raise NotImplementedError("Please specify root directory.")

    params = Params.from_file(os.path.join(root, 'configs/model.json'))
    print(params.params)

    tmp_dir = tempfile.mktemp(prefix=os.path.join(root, 'tmp_exps'))

    if args.finetune:
        model_path = 'text_summarization/Model/tmp_expsyj8uupql'
    else:
        model = train_model(params, tmp_dir)
        total_num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        with open('model_info.txt', 'a') as f:
            print('Number of trainable parameters:', total_num_params, file=f)
            print('Number of all set parameters:', sum(p.numel() for p in model.parameters()), file=f)
            print("-"*100, file=f)
            print('Model path:', tmp_dir, file=f)
            print("-"*100, file=f)
            print('Model architecture:\n', model, file=f)
            print("-"*100, file=f)
            for name, param in model.named_parameters():
                if param.requires_grad:
                    print(name, param.data, '\n\n', file=f)
