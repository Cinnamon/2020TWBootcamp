import os
import time
import copy
import logging

from flask import Flask, request, abort
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import *

from bidaf.prepro import transfer_format
from bidaf.prepro import gen
from bidaf.infer import build_inference
from bidaf.infer import inference as bidaf_inference
from bidaf.infer import get_test_args

from bert.infer_utils import evaluate
from bert.infer_utils import transfer_format as bert_transfer_format
from bert.infer import inference as bert_inference

app = Flask(__name__)


# print env variables
for env_k, env_v in os.environ.items():
    print(f'{env_k}: {env_v}')

# read config
if os.path.exists('./config.yml'):
    # Load config
    import yaml

    with open('./config.yml', 'r') as yml_f:
        config = yaml.load(yml_f, Loader=yaml.BaseLoader)

    # Channel Access Token
    line_bot_api = LineBotApi(config['Linebot']['access_token'])
    # Channel Secret
    handler = WebhookHandler(config['Linebot']['secret'])
else:
    line_bot_api = LineBotApi(os.environ.get('access_token'))
    handler = WebhookHandler(os.environ.get('secret'))

input_ = {
    'context': '',  # type = str
    'qas': [
        {
            'question': ''  # type = str
        },
        {
            'question': ''  # type = str
        },
        {
            'question': ''  # type = str
        }
    ]
}

# An example
# input_ = {
#     'context': "Rap is usually delivered over a beat, typically provided by a DJ, turntablist, beatboxer, or performed a cappella without accompaniment. Stylistically, rap occupies a gray area between speech, prose, poetry, and singing. The word, which predates the musical form, originally meant to lightly strike, and is now used to describe quick speech or repartee. The word had been used in British English since the 16th century. It was part of the African American dialect of English in the 1960s meaning to converse, and very soon after that in its present usage as a term denoting the musical style. Today, the term rap is so closely associated with hip-hop music that many writers use the terms interchangeably.",
#     "qas": [{'question': 'What is the original meaning of the word rap?'},
#             {'question': 'What does rap currently mean?'},
#             {'question': 'When was rap first used?'},
#             {'question': 'What is rap closely associated with today?'},
#             {'question': 'Who is the most famous rap singer today?'},
#             {'question': 'When did rap become popular?'},
#             {'question': 'What does hip-hop mean?'},
#             {'question': 'What is the difference between hip-hop and rap?'},
#             {'question': 'What are the other musical styles besides rap?'},
#             {'question': 'How is rap usually delivered?'}
#     ]
# }
input_context_msg = 'Input context'
input_question_msg = 'Input questions'
input_answer_msg = 'Answer'
input_reset_msg = 'Reset'


def run_bidaf(_input):
    args = get_test_args()
    setattr(args, 'test_record_file', './infer/data/infer.npz')
    setattr(args, 'word_emb_file', './infer/data/word_emb.json')
    setattr(args, 'char_emb_file', './infer/data/char_emb.json')
    setattr(args, 'test_eval_file', './infer/data/infer_eval.json')
    setattr(args, 'load_path', './infer/weight/best.pth.tar')

    m = build_inference(args)
    d = transfer_format(_input)
    gen(d, word2idx='./infer/data/word2idx.json', char2idx='./infer/data/char2idx.json')
    result = bidaf_inference(m)
    return result


def run_bert(_input):
    args, model, tokenizer = bert_inference(src_root='bert/')
    bert_transfer_format(_input, src_root='bert/')
    result = evaluate(args, model, tokenizer)
    return result


@app.route("/callback", methods=['POST'])
def callback():
    # get X-Line-Signature header value
    signature = request.headers['X-Line-Signature']
    # get request body as text
    body = request.get_data(as_text=True)
    app.logger.info("Request body: " + body)
    # handle webhook body
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)
    return 'OK'


@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    def reply(text):
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text=text))

    # init
    message = TextSendMessage(text=event.message.text)
    try:
        handle_message.state_ = handle_message.state_
    except:
        handle_message.state_ = 'init'
    print(handle_message.state_)

    try:
        handle_message.input_ = handle_message.input_
    except:
        handle_message.input_ = copy.deepcopy(input_)
    print(handle_message.input_)

    # state
    if handle_message.state_ == 'init':
        pass
    elif handle_message.state_ == 'context':
        reply(f'You input a context\n{message.text}')
        handle_message.input_['context'] = message.text
        handle_message.state_ = 'init'
    elif handle_message.state_ == 'question':
        num_q = len([q for q in handle_message.input_['qas'] if len(q['question']) > 0])
        if num_q >= 3:
            reply('You have already input 3 questions')
        else:
            reply(f'You input a question\n{message.text}')
            handle_message.input_['qas'][num_q]['question'] = message.text
        handle_message.state_ = 'init'

    for k, v in handle_message.input_.items():
        print(f'{k}: {v}')

    # reply message
    if message.text == input_reset_msg:
        handle_message.input_ = copy.deepcopy(input_)
        handle_message.state_ = 'init'
        reply('Reset Done')
        print(handle_message.input_)
    elif message.text == input_context_msg:
        reply('Please input a context')
        handle_message.state_ = 'context'
    elif message.text == input_question_msg:
        num_q = len([q for q in handle_message.input_['qas'] if len(q['question']) > 0])
        if num_q >= 3:
            reply('You have already input 3 questions')
            handle_message.state_ = 'init'
        else:
            reply('Please input a question')
            handle_message.state_ = 'question'
    elif message.text == input_answer_msg:
        # run model
        t1 = time.time()
        # answers = run_bidaf(handle_message.input_)
        answers = run_bert(handle_message.input_)
        print(time.time() - t1)

        # send messages
        if len(answers) > 0:
            ans = '\n'.join(f'{aid}: {ans}' for aid, ans in answers.items())
            reply(ans)
        else:
            reply('No Answer')
        handle_message.state_ = 'init'


if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
