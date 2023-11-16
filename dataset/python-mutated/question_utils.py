"""
Question utils functions
"""
import pathlib
from random import choice
from typing import List
import re
p = pathlib.Path(__file__).parent.parent.joinpath('README.md')

def get_file_list():
    if False:
        i = 10
        return i + 15
    file_list = ''
    with open(p, 'rb') as f:
        for line in f.readlines():
            file_list += line.rstrip().decode()
    return file_list

def get_question_list(file_list: List[str]) -> list:
    if False:
        print('Hello World!')
    file_list = re.findall('<details>(.*?)</details>', file_list)
    questions_list = []
    for i in file_list:
        q = re.findall('<summary>(.*?)</summary>', i)[0]
        questions_list.append(q)
    return questions_list

def get_answered_questions(question_list: List[str]) -> list:
    if False:
        print('Hello World!')
    t = []
    question_list = re.findall('<details>(.*?)</details>', question_list)
    for i in question_list:
        q = re.findall('<summary>(.*?)</summary>', i)
        if q and q[0] == '':
            continue
        a = re.findall('<b>(.*?)</b>', i)
        if a and a[0] == '':
            continue
        else:
            t.append(q[0])
    return t

def get_answers_count() -> List:
    if False:
        i = 10
        return i + 15
    '\n    Return [answer_questions,all_questions] ,PASS complete. FAIL incomplete.\n    >>> get_answers_count()\n    [463, 463]\n    '
    ans_questions = get_answered_questions(get_file_list())
    len_ans_questions = len(ans_questions)
    all_questions = get_question_list(get_file_list())
    len_all_questions = len(all_questions)
    return [len_ans_questions, len_all_questions]

def get_challenges_count() -> int:
    if False:
        for i in range(10):
            print('nop')
    challenges_path = pathlib.Path(__file__).parent.parent.joinpath('exercises').glob('*.md')
    return len(list(challenges_path))

def get_random_question(question_list: List[str], with_answer=False):
    if False:
        while True:
            i = 10
    if with_answer:
        return choice(get_answered_questions(question_list))
    return choice(get_question_list(question_list))
"Use this question_list. Unless you have already opened/worked/need the file, then don't or\nyou will end up doing the same thing twice.\neg:\n#my_dir/main.py\nfrom scripts import question_utils\nprint(question_utils.get_answered_questions(question_utils.question_list)\n>> 123\n # noqa: E501\n"
if __name__ == '__main__':
    import doctest
    doctest.testmod()