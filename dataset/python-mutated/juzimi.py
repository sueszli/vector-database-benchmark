"""
句子迷：（https://www.juzimi.com/）
民国情书：朱生豪先生的情话 && 爱你就像爱生命
Author: ClaireYiu(https://github.com/ClaireYiu)
"""
import random
import requests

def get_zsh_info():
    if False:
        print('Hello World!')
    '\n    句子迷：（https://www.juzimi.com/）\n    朱生豪：https://www.juzimi.com/writer/朱生豪\n    爱你就像爱生命（王小波）：https://www.juzimi.com/article/爱你就像爱生命\n    三行情书：https://www.juzimi.com/article/25637\n    :return: str 情话\n    '
    print('正在获取民国情话...')
    try:
        name = [['writer/朱生豪', 38], ['article/爱你就像爱生命', 22], ['article/25637', 55]]
        apdix = random.choice(name)
        url = 'https://www.juzimi.com/{}?page={}'.format(apdix[0], random.randint(1, apdix[1]))
        resp = requests.get(url)
        if resp.status_code == 200:
            return None
        print('获取民国情话失败..')
    except Exception as exception:
        print(exception)
    return None
get_one_words = get_zsh_info
if __name__ == '__main__':
    pass