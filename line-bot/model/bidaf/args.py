"""Command-line arguments for setup.py, train.py, test.py.

Author:
    Chris Chute (chute@stanford.edu)
"""

import argparse
bidaf_src_root = 'line-bot/model/bidaf/'


def get_test_args():
    """Get arguments needed in test.py."""
    parser = argparse.ArgumentParser('Test a trained model on SQuAD')

    add_common_args(parser)
    add_train_test_args(parser)

    parser.add_argument('--split',
                        type=str,
                        default='test',
                        choices=('train', 'dev', 'test'),
                        help='Split to use for testing.')
    parser.add_argument('--glove_url',
                        type=str,
                        default='http://nlp.stanford.edu/data/glove.840B.300d.zip')    
    parser.add_argument('--test_para_limit',
                        type=int,
                        default=1000,
                        help='Max number of words in a paragraph at test time')
    parser.add_argument('--test_ques_limit',
                        type=int,
                        default=100,
                        help='Max number of words in a question at test time')
    parser.add_argument('--ans_limit',
                        type=int,
                        default=30,
                        help='Max number of words in a training example answer')
    parser.add_argument('--char_limit',
                        type=int,
                        default=16,
                        help='Max number of chars to keep from a word')
    parser.add_argument('--test_meta_file',
                        type=str,
                        default='./data/infer_meta.json')
    args = parser.parse_args(args=[])

    return args


def add_common_args(parser):
    """Add arguments common to all 3 scripts: setup.py, train.py, test.py"""
    parser.add_argument('--test_record_file',
                        type=str,
                        default=bidaf_src_root + 'data/infer.npz')
    parser.add_argument('--word_emb_file',
                        type=str,
                        # default='./data/word_emb.json',
                        default=bidaf_src_root + 'data/word_emb.json')
    parser.add_argument('--char_emb_file',
                        type=str,
                        # default='./data/char_emb.json',
                        default=bidaf_src_root + 'data/char_emb.json')
    parser.add_argument('--test_eval_file',
                        type=str,
                        default=bidaf_src_root + 'data/infer_eval.json')


def add_train_test_args(parser):
    """Add arguments common to train.py and test.py"""
    parser.add_argument('--name',
                        type=str,
                        default='test',
                        help='Name to identify training or test run.')
    parser.add_argument('--max_ans_len',
                        type=int,
                        default=15,
                        help='Maximum length of a predicted answer.')
    parser.add_argument('--num_workers',
                        type=int,
                        default=4,
                        help='Number of sub-processes to use per data loader.')
    parser.add_argument('--batch_size',
                        type=int,
                        default=16,#64,
                        help='Batch size per GPU. Scales automatically when \
                              multiple GPUs are available.')
    parser.add_argument('--use_squad_v2',
                        type=lambda s: s.lower().startswith('t'),
                        default=True,
                        help='Whether to use SQuAD 2.0 (unanswerable) questions.')
    parser.add_argument('--hidden_size',
                        type=int,
                        default=100,
                        help='Number of features in encoder hidden layers.')
    parser.add_argument('--num_visuals',
                        type=int,
                        default=10,
                        help='Number of examples to visualize in TensorBoard.')
    parser.add_argument('--load_path',
                        type=str,
                        default=bidaf_src_root + 'weight/best.pth.tar',
                        help='Path to load as a model checkpoint.')
