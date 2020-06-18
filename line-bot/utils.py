
# Template for input
input_template = {
    'context': '',  # type = str
    'qas': [
        # {'question': ''}, # type = dict(key: str)
    ]
}

# Commands
input_context_msg = 'Input context'
input_question_msg = 'Input questions'
input_answer_msg = 'Answer'
input_reset_msg = 'Reset'
command_list = [input_context_msg, input_question_msg, input_answer_msg, input_reset_msg]
question_limit = 3


def check_input_valid(_input):
    return _input.get('context', '') != ''


def is_input_command(input_text):
    return input_text in command_list
