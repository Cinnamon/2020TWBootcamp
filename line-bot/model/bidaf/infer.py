import torch
import torch.nn as nn
import torch.utils.data as data
from . import util

from .args import get_test_args
from .models import BiDAF
from tqdm import tqdm
from ujson import load as json_load
from .util import collate_fn, SQuAD

device, gpu_ids = util.get_available_devices()
args = get_test_args()


def build_inference(args_=None):
    args_ = args_ if args_ is not None else args

    # device, gpu_ids = util.get_available_devices()
    args_.batch_size *= max(1, len(gpu_ids))

    # Get embeddings
    word_vectors = util.torch_from_json(args_.word_emb_file)

    char_vectors = util.torch_from_json(args_.char_emb_file)

    # Get model
    model = BiDAF(word_vectors=word_vectors,
                  char_vectors=char_vectors,
                  hidden_size=args_.hidden_size)
    model = nn.DataParallel(model, gpu_ids)
    model = util.load_model(model, args_.load_path, gpu_ids, return_step=False)
    model = model.to(device)
    model.eval()

    return model


def inference(model):
    # Get data loader
    record_file = vars(args)[f'{args.split}_record_file']
    dataset = SQuAD(record_file, args.use_squad_v2)
    data_loader = data.DataLoader(dataset,
                                  batch_size=args.batch_size,
                                  shuffle=False,
                                  num_workers=args.num_workers,
                                  collate_fn=collate_fn)
    # Evaluate
    nll_meter = util.AverageMeter()
    #    pred_dict = {}  # Predictions for TensorBoard
    sub_dict = {}  # Predictions for submission
    eval_file = vars(args)[f'{args.split}_eval_file']
    with open(eval_file, 'r') as fh:
        gold_dict = json_load(fh)
    with torch.no_grad(), \
         tqdm(total=len(dataset)) as progress_bar:
        for cw_idxs, cc_idxs, qw_idxs, qc_idxs, y1, y2, ids in data_loader:
            # Setup for forward
            cw_idxs = cw_idxs.to(device)
            qw_idxs = qw_idxs.to(device)
            cc_idxs = cc_idxs.to(device)
            qc_idxs = qc_idxs.to(device)
            batch_size = cw_idxs.size(0)

            # Forward
            log_p1, log_p2 = model(cw_idxs, qw_idxs, cc_idxs, qc_idxs)

            # Get F1 and EM scores
            p1, p2 = log_p1.exp(), log_p2.exp()
            starts, ends = util.discretize(p1, p2, args.max_ans_len, args.use_squad_v2)

            # Log info
            progress_bar.update(batch_size)
            idx2pred, uuid2pred = util.convert_tokens(gold_dict,
                                                      ids.tolist(),
                                                      starts.tolist(),
                                                      ends.tolist(),
                                                      args.use_squad_v2)
            #            pred_dict.update(idx2pred)
            sub_dict.update(uuid2pred)
    return sub_dict
