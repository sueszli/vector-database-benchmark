import sys
import argparse
parser = argparse.ArgumentParser()
print('uso correcto: python3 jaliagag.py "4 + 5 / 6 - -4"\n')
op = ['+', '-', '*', '/', '%']
args = sys.argv[1]
inp = args.split()
no_paren = []
for i in inp:
    if i == '(' or i == ')':
        pass
    elif '(' in i:
        b = i.replace('(', '')
        no_paren.append(b)
    elif ')' in i:
        b = i.replace(')', '')
        no_paren.append(b)
    else:
        no_paren.append(i)

def check_string(sth):
    if False:
        i = 10
        return i + 15
    for (idx, i) in enumerate(sth):
        if idx % 2 == 0:
            try:
                a = int(i)
                if isinstance(a, int):
                    continue
            except:
                return False
        elif i in op:
            continue
        else:
            return False
    return True
if __name__ == '__main__':
    if check_string(no_paren):
        print('true')
    else:
        print('false')