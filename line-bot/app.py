import os
import time
import copy
import yaml
import json
import logging
import argparse

from flask import Flask, request, abort
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import *

# from model.bidaf.prepro import transfer_format
# from model.bidaf.prepro import gen
# from model.bidaf.infer import build_inference
# from model.bidaf.infer import inference as bidaf_inference
# from model.bidaf.infer import get_test_args

# from model.bert.infer_utils import evaluate
# from model.bert.infer_utils import transfer_format as bert_transfer_format
# from model.bert.infer import inference as bert_inference

from model.utils import init_bert
from model.utils import run_bert
from model.utils import init_bidaf
from model.utils import run_bidaf

from utils import check_input_valid
from utils import is_input_command
from utils import input_context_msg, input_question_msg, input_answer_msg, input_reset_msg
from utils import input_template as input_
from utils import question_limit


# setup logger
logging.basicConfig(
    level="DEBUG",  # INFO, DEBUG
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# check env variables
logger.debug("Check environment variables")
for env_k, env_v in os.environ.items():
    logger.debug(f'{env_k}: {env_v}')


# parse args
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--model', help='Select model', default='bert', required=False, choices=['bert', 'bidaf'])
parser.add_argument('--example', help='Use example to test model', action='store_true')
args, unknown = parser.parse_known_args()

# setup app
app = Flask(__name__)
if args.model == 'bert':
    app.model = init_bert()
    app.predict = run_bert
elif args.model == 'bidaf':
    app.model = init_bidaf()
    app.predict = run_bidaf

# load example
if args.example:
    with open('line-bot/example.json', 'r') as f:
        example_input = json.load(f)
        logger.debug(f'Loaded example: {example_input}')

# load config
config_path = 'line-bot/config.yml'
try:
    with open(config_path, 'r') as yml_f:
        config = yaml.load(yml_f, Loader=yaml.BaseLoader)
    # Channel Access Token
    line_bot_api = LineBotApi(config['Linebot']['access_token'])
    # Channel Secret
    handler = WebhookHandler(config['Linebot']['secret'])
except:
    logger.exception(f'Please check if {config_path} exists')


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

    def handle_q():
        num_qa_ = len(handle_message.answers)
        if num_qa_ >= question_limit:
            reply('You have already input 3 questions')
            handle_message.state_ = 'init'
        elif not check_input_valid(handle_message.input_):
            reply('Please input context first')
            handle_message.state_ = 'init'
        else:
            reply(f'You input a question\n{message.text}')
            handle_message.input_['qas'].append({'question': message.text})

            # run model
            t1 = time.time()
            handle_message.state_ = 'processing'
            answers = app.predict(handle_message.input_, app.model)
            answers = answers.get('1')
            logger.debug(f'Running time of model: {time.time() - t1} sec')
            logger.debug(f'This answer is \'{answers}\'')

            # save to answer list
            qa_id = len(handle_message.answers) + 1
            handle_message.answers.append(f'Q{qa_id}: {message.text}\nA{qa_id}: {answers}\n')

            # release question list
            handle_message.input_['qas'].clear()

            handle_message.state_ = 'question'

    # Receive Message
    logger.info('Enter Callback: handle_message()')
    message = TextSendMessage(text=event.message.text)

    # Check attributes
    logger.debug(f'Current state:\n{handle_message.state_}')
    logger.debug(f'Current input:\n{handle_message.input_}')
    logger.debug(f'Current answers:\n{handle_message.answers}')

    # Update state
    if handle_message.state_ == 'context':
        if message.text == input_reset_msg:
            pass
        elif is_input_command(message.text):
            reply(f'You input a command, please re-input')
            return
        else:
            reply(f'You input a context\n{message.text}')
            handle_message.input_['context'] = message.text
            handle_message.state_ = 'init'

    elif handle_message.state_ == 'question':
        if message.text == input_reset_msg or message.text == input_answer_msg:
            pass
        elif is_input_command(message.text):
            reply(f'You input a command, please re-input')
            return
        else:
            handle_q()
    else:
        pass
    logger.debug(f'Current input after state changes:\n{handle_message.input_}')

    # Reply message
    if message.text == input_reset_msg:
        handle_message.input_ = copy.deepcopy(input_)
        handle_message.state_ = 'init'
        reply('Reset Done')
        logger.debug(f'Current input after reset:\n{handle_message.input_}')

    elif message.text == input_context_msg:
        reply('Please input a context')
        handle_message.state_ = 'context'

    elif message.text == input_question_msg:
        num_qa = len(handle_message.answers)
        if num_qa >= question_limit:
            reply('You have already input 3 questions')
            handle_message.state_ = 'init'
        elif not check_input_valid(handle_message.input_):
            reply('Please input context first')
        else:
            reply('Please start to input questions')
            handle_message.state_ = 'question'

    elif message.text == input_answer_msg:
        if args.example:
            answers = app.predict(handle_message.input_, app.model)
            handle_message.answers = []
            for qa_id, qas in enumerate(handle_message.input_.get("qas")):
                ques = qas.get('question')
                a = answers.get(str(qa_id + 1))
                handle_message.answers.append(f'Q{qa_id}: {ques}\nA{qa_id}: {a}\n')

        # send messages
        if len(handle_message.answers) > 0:
            ans = '\n'.join([a for a in handle_message.answers])
            reply(ans)
            handle_message.answers.clear()
        elif handle_message.state_ == 'processing':
            reply('Still in processing, please try again later')
        else:
            reply('No Answer')
        handle_message.state_ = 'init'


setattr(handle_message, 'state_', 'init' if args.example is False else 'question')  # [init, context, question, processing]
setattr(handle_message, 'input_', copy.deepcopy(input_ if args.example is False else example_input))
setattr(handle_message, 'answers', list())


if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
