"""Convert the source TSSB-3M  dataset to instruction data
"""
import json
import random
import re
from os.path import join
from tqdm import tqdm
INSTRUCTIONS_LIST = ['Find the bug in the following code:', 'Identify the error in the code snippet provided:', 'Spot the issue within the given code segment:', 'Locate the problem in the code example below:', 'Uncover the malfunction in the following piece of code:', 'Detect the flaw in the code provided:', 'Pinpoint the glitch in the code sample below:', 'Search for the anomaly in the given code:', 'Determine the defect within the following code:', 'Discover the fault in the code segment provided:', 'Trace the irregularity in the code example below:', 'Please locate the error in the code provided.', 'Can you identify the mistake in this code?', 'There seems to be a problem with this code. Can you find it?', 'Please investigate the code and locate the bug.', 'Please examine the code and find the error.', 'Can you pinpoint the issue with this code?', 'Please review the code and identify the bug.', 'Can you detect the problem with this code?', 'Please analyze the code and find the mistake.', 'Can you spot the bug in the code provided?']
RESPONSE_PREFIX_WORDS = ['The fix of the bug can be laid out as', 'The resolution of the error can be portrayed like so', 'The solution for the flaw can be summarized as such', 'The remedy of the mistake can be captured in this way', 'The correction of the fault can be depicted like this', 'The patch for the glitch can be articulated as', 'The workaround of the defect can be conveyed in this manner', 'The troubleshooting of the issue can be explained like this', 'The adjustment to the anomaly can be illustrated as follows', 'The modification for the irregularity can be exemplified like this']

def gen_instruction():
    if False:
        print('Hello World!')
    idx = random.randint(0, len(INSTRUCTIONS_LIST) - 1)
    return INSTRUCTIONS_LIST[idx]

def gen_response_prefix():
    if False:
        for i in range(10):
            print('nop')
    idx = random.randint(0, len(RESPONSE_PREFIX_WORDS) - 1)
    return RESPONSE_PREFIX_WORDS[idx]
TEMPLATE = 'User: {}\n{}\nReply: The fixed code is:\n```\n{}\n```\n'
TEMPLATE_COMMIT_MSG = 'User: {}\n{}\nReply: {}:\n{}\nThe fixed code is:\n```\n{}\n```\n'
INSTRUCTON_TEMPLATE = '{}\n{}\n'
RESPONSE_TEMPLATE = 'The fixed code is:\n```\n{}\n```\n'
RESPONSE_TEMPLATE_COMMIT_MSG = '{}:\n{}\n\nThe fixed code is:\n```\n{}\n```\n'

def remove_starting_plus_minus(text):
    if False:
        return 10
    if text.startswith('+') or text.startswith('-'):
        return text[1:]
    else:
        return text

def remove_extraneous_diff_info(text):
    if False:
        print('Hello World!')
    pattern = '@@.*@@'
    return re.sub(pattern, '', text)

def clean(text):
    if False:
        return 10
    return remove_extraneous_diff_info(remove_starting_plus_minus(text))

def clean_PII(text):
    if False:
        print('Hello World!')
    signoff_index = text.rfind('\n\nSigned-off-by:')
    if signoff_index != -1:
        text = text[:signoff_index]
    email_pattern = '[a-zA-Z0-9._%+-]+@(?:[a-zA-Z0-9-]+\\.)+[a-zA-Z]{2,}'
    clean_text = re.sub(email_pattern, '', text)
    return clean_text
INVALID_COMMIT_MESSAGES = set([line.strip().split('\t')[0] for line in open('invalid_commit_messages.tsv').readlines()])

def is_invaid_commit_msg(text):
    if False:
        for i in range(10):
            print('nop')
    'commit message that is incomplete, eg. "fix bug", "hotfix" '
    return text.strip() in INVALID_COMMIT_MESSAGES

def clean_commit_msg(text):
    if False:
        while True:
            i = 10
    '\n    # 1. remove issue id , eg. msg: "rename (hetr_passes -> passes) #1195" -> "rename (hetr_passes -> passes)"\n    # 2. remove `fix` prefix:\n    some typical cases:\n    ## eg. [fix] 拼写错误 -> 拼写错误\n    ## eg. [FIX] purchase_indonesia : AttributeError \'NoneType\' object has no attribute \'id\' ->  AttributeError \'NoneType\' object has no attribute \'id\'\n    ## "fix force insert error refs #2" -> "fix force insert error"\n    ## "Fix namespace of RPCError Fixes #76" ->  "Fix namespace of RPCError"\n    ## "fix a minor bug in survey_spec password field handling see: #5477" -> "fix a minor bug in survey_spec password field handling"\n    ## issue #973 -> ""\n    ## "Fixes #246"  -> ""\n    ## "Close #152." -> ""\n    ## "wrong learning rate schedule (#2360)"  -> "wrong learning rate schedule"\n    '
    text = clean_PII(text)
    pattern = '\\(?#\\d{1,6}\\)?'
    text = re.sub(pattern, '', text)
    text = re.sub('\\s+', ' ', text).strip()
    if len(text) < 4:
        return None
    if is_invaid_commit_msg(text):
        return None
    return text

def create(input_file, output_file, output_json=True):
    if False:
        return 10
    fout = open(output_file, 'w')
    with open(input_file) as fin:
        for line in tqdm(fin):
            row = json.loads(line.strip())
            wrong = '\n'.join((clean(line) for line in row['diff'].split('\n') if not line.startswith('+')))
            correct = '\n'.join((clean(line) for line in row['diff'].split('\n') if not line.startswith('-')))
            instruction = INSTRUCTON_TEMPLATE.format(wrong, correct)
            commit_msg = clean_commit_msg(row['commit_message']) if 'commit_message' in row else None
            if commit_msg:
                out_str = TEMPLATE_COMMIT_MSG.format(gen_instruction(), wrong, gen_response_prefix(), commit_msg, correct)
                response = RESPONSE_TEMPLATE_COMMIT_MSG.format(gen_response_prefix(), commit_msg, correct)
            else:
                out_str = TEMPLATE.format(gen_instruction(), wrong, correct)
                response = RESPONSE_TEMPLATE.format(correct)
            if output_json:
                row = {'INSTRUCTION': instruction, 'RESPONSE': response, 'SOURCE': 'TSSM-3M', 'METADATA': {'project_url': row['project_url'], 'file_path': row['file_path'], 'commit_sha': row['commit_sha']}}
                out_str = json.dumps(row, ensure_ascii=False)
            print(out_str, file=fout)
        fout.close()
if __name__ == '__main__':
    '\n    # get source data from huggingface repository\n     !wget https://huggingface.co/datasets/zirui3/TSSB-3M-ext/blob/main/data.jsonl.gz\n     !gzip -d data.jsonl.gz\n    '
    data_dir = '.'
    input_file = join(data_dir, 'data.jsonl')
    output_file = join(data_dir, 'instructions.jsonl')
    create(input_file, output_file, output_json=True)