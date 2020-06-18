
# Template for input
input_template = {
    'context': '',  # type = str
    'qas': [
        # {'question': ''}, # type = dict(key: str)
    ]
}

# An example for input
# input_template = {
#     'context': "Rap is usually delivered over a beat, typically provided by a DJ, turntablist, beatboxer,
#     or performed a cappella without accompaniment. Stylistically, rap occupies a gray area between speech,
#     prose, poetry, and singing. The word, which predates the musical form, originally meant to lightly strike,
#     and is now used to describe quick speech or repartee. The word had been used in British English since the
#     16th century. It was part of the African American dialect of English in the 1960s meaning to converse,
#     and very soon after that in its present usage as a term denoting the musical style. Today, the term rap
#     is so closely associated with hip-hop music that many writers use the terms interchangeably.",
#     "qas": [
#         {'question': 'What is the original meaning of the word rap?'},
#         {'question': 'What does rap currently mean?'},
#         {'question': 'When was rap first used?'},
#         {'question': 'What is rap closely associated with today?'},
#         {'question': 'Who is the most famous rap singer today?'},
#         {'question': 'When did rap become popular?'},
#         {'question': 'What does hip-hop mean?'},
#         {'question': 'What is the difference between hip-hop and rap?'},
#         {'question': 'What are the other musical styles besides rap?'},
#         {'question': 'How is rap usually delivered?'}
#     ]
# }

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
