import json
import sys

def check_approval(count, required_reviewers):
    if False:
        for i in range(10):
            print('nop')
    json_buff = ''
    for line in sys.stdin:
        json_buff = ''.join([json_buff, line])
    json_resp = json.loads(json_buff)
    approves = 0
    approved_user_ids = []
    approved_user_logins = set()
    for review in json_resp:
        if review['state'] == 'APPROVED':
            approves += 1
            approved_user_ids.append(review['user']['id'])
            approved_user_logins.add(review['user']['login'])
    required_reviewers_int = set()
    required_reviewers_login = set()
    for rr in required_reviewers:
        if rr.isdigit():
            required_reviewers_int.add(int(rr))
        else:
            required_reviewers_login.add(rr)
    if len(set(approved_user_ids) & required_reviewers_int) + len(approved_user_logins & required_reviewers_login) >= count:
        print('TRUE')
    else:
        print('FALSE')
if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        check_approval(int(sys.argv[1]), sys.argv[2:])
    else:
        print('Usage: python check_pr_approval.py [count] [required reviewer id] ...')