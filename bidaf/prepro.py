import numpy as np
import os
import spacy
import ujson as json
import urllib.request

from codecs import open
from collections import Counter
from subprocess import run
from tqdm import tqdm
from .args import get_test_args

nlp = spacy.blank("en")


def transfer_format(inp):
    data_f = {}
    data = {'title': 'demo'}
    for i, q in enumerate(inp['qas']):
        q['id'] = str(i + 1)
        q['answers'] = []
    temp = {'qas': inp['qas']}
    temp['context'] = inp['context']
    data['paragraphs'] = [temp]
    data_f['data'] = [data]
    return data_f


def word_tokenize(sent):
    doc = nlp(sent)
    return [token.text for token in doc]


def convert_idx(text, tokens):
    current = 0
    spans = []
    for token in tokens:
        current = text.find(token, current)
        if current < 0:
            print(f"Token {token} cannot be found")
            raise Exception()
        spans.append((current, current + len(token)))
        current += len(token)
    return spans


def process_file(file, data_type, word_counter, char_counter):
    print(f"Pre-processing {data_type} examples...")
    examples = []
    eval_examples = {}
    total = 0
    source = file
    for article in tqdm(source["data"]):
        for para in article["paragraphs"]:
            context = para["context"].replace(
                "''", '" ').replace("``", '" ')  # .replace('\n','')
            context_tokens = word_tokenize(context)
            context_chars = [list(token) for token in context_tokens]
            spans = convert_idx(context, context_tokens)
            for token in context_tokens:
                word_counter[token] += len(para["qas"])
                for char in token:
                    char_counter[char] += len(para["qas"])
            for qa in para["qas"]:
                total += 1
                ques = qa["question"].replace(
                    "''", '" ').replace("``", '" ')
                ques_tokens = word_tokenize(ques)
                ques_chars = [list(token) for token in ques_tokens]
                for token in ques_tokens:
                    word_counter[token] += 1
                    for char in token:
                        char_counter[char] += 1
                y1s, y2s = [], []
                answer_texts = []
                for answer in qa["answers"]:
                    answer_text = answer["text"]
                    answer_start = answer['answer_start']
                    answer_end = answer_start + len(answer_text)
                    answer_texts.append(answer_text)
                    answer_span = []
                    for idx, span in enumerate(spans):
                        if len(article['paragraphs']) > 400:
                            if len(answer_text) == 1:
                                if answer_start == span[1]:
                                    answer_span.append(idx)
                                elif answer_end == span[0]:
                                    answer_span.append(idx)
                                elif answer_start == span[0]:
                                    answer_span.append(idx)
                            else:
                                if not (answer_end <= span[0] or answer_start >= span[1]):
                                    answer_span.append(idx)
                        else:
                            if not (answer_end <= span[0] or answer_start >= span[1]):
                                answer_span.append(idx)
                    y1, y2 = answer_span[0], answer_span[-1]
                    y1s.append(y1)
                    y2s.append(y2)
                example = {"context_tokens": context_tokens,
                           "context_chars": context_chars,
                           "ques_tokens": ques_tokens,
                           "ques_chars": ques_chars,
                           "y1s": y1s,
                           "y2s": y2s,
                           "id": total}
                examples.append(example)
                eval_examples[str(total)] = {"context": context,
                                             "question": ques,
                                             "spans": spans,
                                             "answers": answer_texts,
                                             "uuid": qa["id"]}
    print(f"{len(examples)} questions in total")
    return examples, eval_examples


def convert_to_features(args, data, word2idx_dict, char2idx_dict, is_test):
    example = {}
    context, question = data
    context = context.replace("''", '" ').replace("``", '" ')
    question = question.replace("''", '" ').replace("``", '" ')
    example['context_tokens'] = word_tokenize(context)
    example['ques_tokens'] = word_tokenize(question)
    example['context_chars'] = [list(token) for token in example['context_tokens']]
    example['ques_chars'] = [list(token) for token in example['ques_tokens']]

    para_limit = args.test_para_limit if is_test else args.para_limit
    ques_limit = args.test_ques_limit if is_test else args.ques_limit
    char_limit = args.char_limit

    def filter_func(example):
        return len(example["context_tokens"]) > para_limit or \
               len(example["ques_tokens"]) > ques_limit

    if filter_func(example):
        raise ValueError("Context/Questions lengths are over the limit")

    context_idxs = np.zeros([para_limit], dtype=np.int32)
    context_char_idxs = np.zeros([para_limit, char_limit], dtype=np.int32)
    ques_idxs = np.zeros([ques_limit], dtype=np.int32)
    ques_char_idxs = np.zeros([ques_limit, char_limit], dtype=np.int32)

    def _get_word(word):
        for each in (word, word.lower(), word.capitalize(), word.upper()):
            if each in word2idx_dict:
                return word2idx_dict[each]
        return 1

    def _get_char(char):
        if char in char2idx_dict:
            return char2idx_dict[char]
        return 1

    for i, token in enumerate(example["context_tokens"]):
        context_idxs[i] = _get_word(token)

    for i, token in enumerate(example["ques_tokens"]):
        ques_idxs[i] = _get_word(token)

    for i, token in enumerate(example["context_chars"]):
        for j, char in enumerate(token):
            if j == char_limit:
                break
            context_char_idxs[i, j] = _get_char(char)

    for i, token in enumerate(example["ques_chars"]):
        for j, char in enumerate(token):
            if j == char_limit:
                break
            ques_char_idxs[i, j] = _get_char(char)

    return context_idxs, context_char_idxs, ques_idxs, ques_char_idxs


def is_answerable(example):
    return len(example['y2s']) > 0 and len(example['y1s']) > 0


def build_features(args, examples, data_type, out_file, word2idx_dict, char2idx_dict, is_test=False):
    para_limit = args.test_para_limit if is_test else args.para_limit
    ques_limit = args.test_ques_limit if is_test else args.ques_limit
    ans_limit = args.ans_limit
    char_limit = args.char_limit

    def drop_example(ex, is_test_=False):
        if is_test_:
            drop = False
        else:
            drop = len(ex["context_tokens"]) > para_limit or \
                   len(ex["ques_tokens"]) > ques_limit or \
                   (is_answerable(ex) and
                    ex["y2s"][0] - ex["y1s"][0] > ans_limit)

        return drop

    print(f"Converting {data_type} examples to indices...")
    total = 0
    total_ = 0
    meta = {}
    context_idxs = []
    context_char_idxs = []
    ques_idxs = []
    ques_char_idxs = []
    y1s = []
    y2s = []
    ids = []
    for n, example in tqdm(enumerate(examples)):
        total_ += 1

        if drop_example(example, is_test):
            continue

        total += 1

        def _get_word(word):
            for each in (word, word.lower(), word.capitalize(), word.upper()):
                if each in word2idx_dict:
                    return word2idx_dict[each]
            return 1

        def _get_char(char):
            if char in char2idx_dict:
                return char2idx_dict[char]
            return 1

        context_idx = np.zeros([para_limit], dtype=np.int32)
        context_char_idx = np.zeros([para_limit, char_limit], dtype=np.int32)
        ques_idx = np.zeros([ques_limit], dtype=np.int32)
        ques_char_idx = np.zeros([ques_limit, char_limit], dtype=np.int32)

        for i, token in enumerate(example["context_tokens"]):
            context_idx[i] = _get_word(token)
        context_idxs.append(context_idx)

        for i, token in enumerate(example["ques_tokens"]):
            ques_idx[i] = _get_word(token)
        ques_idxs.append(ques_idx)

        for i, token in enumerate(example["context_chars"]):
            for j, char in enumerate(token):
                if j == char_limit:
                    break
                context_char_idx[i, j] = _get_char(char)
        context_char_idxs.append(context_char_idx)

        for i, token in enumerate(example["ques_chars"]):
            for j, char in enumerate(token):
                if j == char_limit:
                    break
                ques_char_idx[i, j] = _get_char(char)
        ques_char_idxs.append(ques_char_idx)

        if is_answerable(example):
            start, end = example["y1s"][-1], example["y2s"][-1]
        else:
            start, end = -1, -1

        y1s.append(start)
        y2s.append(end)
        ids.append(example["id"])

    np.savez(out_file,
             context_idxs=np.array(context_idxs),
             context_char_idxs=np.array(context_char_idxs),
             ques_idxs=np.array(ques_idxs),
             ques_char_idxs=np.array(ques_char_idxs),
             y1s=np.array(y1s),
             y2s=np.array(y2s),
             ids=np.array(ids))
    print(f"Built {total} / {total_} instances of features in total")
    meta["total"] = total
    return meta


def save(filename, obj, message=None):
    if message is not None:
        print(f"Saving {message}...")
        with open(filename, "w") as fh:
            json.dump(obj, fh)


def pre_process(args, file, word2idx, char2idx):
    # Process training set and use it to decide on the word/character vocabularies
    word_counter, char_counter = Counter(), Counter()
    with open(word2idx, "r") as wd:
        word2idx_dict = json.load(wd)
    with open(char2idx, "r") as cd:
        char2idx_dict = json.load(cd)
    # Process dev and test sets
    test_examples, test_eval = process_file(file, "test", word_counter, char_counter)
    save(args.test_eval_file, test_eval, message="test eval")
    test_meta = build_features(args, test_examples, "test",
                               args.test_record_file, word2idx_dict, char2idx_dict, is_test=True)
    # save(args.test_meta_file, test_meta, message="test meta")


def gen(data, word2idx='./data/word2idx.json', char2idx='./data/char2idx.json'):
    # Get command-line args
    args_ = get_test_args()
    pre_process(args_, data, word2idx, char2idx)
