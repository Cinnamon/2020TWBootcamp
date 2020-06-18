from model.bert.infer_utils import evaluate
from model.bert.infer_utils import transfer_format as bert_transfer_format
from model.bert.infer import inference as bert_inference

from model.bidaf.prepro import transfer_format
from model.bidaf.prepro import gen
from model.bidaf.infer import build_inference
from model.bidaf.infer import inference as bidaf_inference
from model.bidaf.infer import get_test_args

bert_src_root = 'line-bot/model/bert/'
bidaf_src_root = 'line-bot/model/bidaf/'


def init_bert(src_root=bert_src_root):
    return bert_inference(src_root=src_root)


def run_bert(_input, loaded_model, src_root=bert_src_root):
    args, model, tokenizer = loaded_model
    bert_transfer_format(_input, src_root=src_root)
    result = evaluate(args, model, tokenizer)
    return result


def init_bidaf(src_root=bidaf_src_root, record_file='infer.npz', w_emb_file='word_emb.json',
               c_emb_file='char_emb.json', eval_file='infer_eval.json', weight_load_path='best.pth.tar'):
    args = get_test_args()
    setattr(args, 'test_record_file', src_root + 'data/' + record_file)
    setattr(args, 'word_emb_file', src_root + 'data/' + w_emb_file)
    setattr(args, 'char_emb_file', src_root + 'data/' + c_emb_file)
    setattr(args, 'test_eval_file', src_root + 'data/' + eval_file)
    setattr(args, 'load_path', src_root + 'weight/' + weight_load_path)

    m = build_inference(args)
    return m


def run_bidaf(_input, loaded_model, src_root=bidaf_src_root, word2idx_file='word2idx.json', char2idx='char2idx.json'):
    d = transfer_format(_input)
    gen(d, word2idx=src_root + 'data/' + word2idx_file, char2idx=src_root + 'data/' + char2idx)
    result = bidaf_inference(loaded_model)
    return result
