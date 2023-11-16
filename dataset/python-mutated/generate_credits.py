from collections import Counter
import locale
import re
import subprocess
TEMPLATE = ".. Note: This file is auto-generated using generate_credits.py\n\n.. _credits:\n\n*******\nCredits\n*******\n\n\nMatplotlib was written by John D. Hunter, with contributions from an\never-increasing number of users and developers.  The current lead developer is\nThomas A. Caswell, who is assisted by many `active developers\n<https://www.openhub.net/p/matplotlib/contributors>`_.\nPlease also see our instructions on :doc:`/citing`.\n\nThe following is a list of contributors extracted from the\ngit revision control history of the project:\n\n{contributors}\n\nSome earlier contributors not included above are (with apologies\nto any we have missed):\n\nCharles Twardy,\nGary Ruben,\nJohn Gill,\nDavid Moore,\nPaul Barrett,\nJared Wahlstrand,\nJim Benson,\nPaul Mcguire,\nAndrew Dalke,\nNadia Dencheva,\nBaptiste Carvello,\nSigve Tjoraand,\nTed Drain,\nJames Amundson,\nDaishi Harada,\nNicolas Young,\nPaul Kienzle,\nJohn Porter,\nand Jonathon Taylor.\n\nThanks to Tony Yu for the original logo design.\n\nWe also thank all who have reported bugs, commented on\nproposed changes, or otherwise contributed to Matplotlib's\ndevelopment and usefulness.\n"

def check_duplicates():
    if False:
        print('Hello World!')
    text = subprocess.check_output(['git', 'shortlog', '--summary', '--email'])
    lines = text.decode('utf8').split('\n')
    contributors = [line.split('\t', 1)[1].strip() for line in lines if line]
    emails = [re.match('.*<(.*)>', line).group(1) for line in contributors]
    email_counter = Counter(emails)
    if email_counter.most_common(1)[0][1] > 1:
        print('DUPLICATE CHECK: The following email addresses are used with more than one name.\nConsider adding them to .mailmap.\n')
        for (email, count) in email_counter.items():
            if count > 1:
                print('{}\n{}'.format(email, '\n'.join((l for l in lines if email in l))))

def generate_credits():
    if False:
        while True:
            i = 10
    text = subprocess.check_output(['git', 'shortlog', '--summary'])
    lines = text.decode('utf8').split('\n')
    contributors = [line.split('\t', 1)[1].strip() for line in lines if line]
    contributors.sort(key=locale.strxfrm)
    with open('credits.rst', 'w') as f:
        f.write(TEMPLATE.format(contributors=',\n'.join(contributors)))
if __name__ == '__main__':
    check_duplicates()
    generate_credits()