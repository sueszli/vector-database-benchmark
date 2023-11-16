"""
ECR Packaging Utils
"""
import re
'\nRegular Expressions for Resources.\n'
HOSTNAME_ECR_AWS = '(?:[a-zA-Z0-9][\\w-]*)\\.dkr\\.ecr\\.(?:[a-zA-Z0-9][\\w-]*)\\.amazonaws\\.com(\\.cn)?'
HOSTNAME_LOCALHOST = 'localhost(?::\\d{1,5})?'
HOSTNAME_127_0_0_1 = '127\\.0\\.0\\.1(?::\\d{1,5})?'
ECR_URL = f'^(?:{HOSTNAME_ECR_AWS}|{HOSTNAME_LOCALHOST}|{HOSTNAME_127_0_0_1})\\/(?:[a-z0-9]+(?:[._-][a-z0-9]+)*\\/)*[a-z0-9]+(?:[._-][a-z0-9]+)*'

def is_ecr_url(url: str) -> bool:
    if False:
        for i in range(10):
            print('nop')
    return bool(re.match(ECR_URL, url)) if url else False