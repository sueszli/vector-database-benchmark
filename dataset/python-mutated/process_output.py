from operator import itemgetter
import json
import sys
import csv

def translate_breakdown(breakdown):
    if False:
        print('Hello World!')
    if breakdown is None:
        return ''
    if breakdown == ['age', 'gender']:
        return 'ag'
    if breakdown == ['country']:
        return 'c'
    if breakdown == ['placement', 'impression_device']:
        return 'pd'
    return 'other'

def load_records():
    if False:
        i = 10
        return i + 15
    for line in sys.stdin:
        yield json.loads(line)

def translate_raw_record(raw):
    if False:
        for i in range(10):
            print('nop')
    breakdowns = translate_breakdown(raw['table']['breakdowns'])
    return {'level': raw['table']['level'], 'bd': breakdowns, 'nabd': len(raw['table']['action_breakdowns']), 'naaw': len(raw['table']['action_attribution_windows']), 'success': raw['return_code'] == 0, 'duration': round(raw['duration'] / 60.0, 1)}

def success(rec):
    if False:
        i = 10
        return i + 15
    return rec['success']

def proportion(pred, recs):
    if False:
        for i in range(10):
            print('nop')
    return float(len(list(filter(pred, recs)))) / float(len(recs))

def p_success(recs):
    if False:
        print('Hello World!')
    return proportion(success, recs)

def p_breakdown(breakdown, recs):
    if False:
        return 10
    return proportion(lambda r: r['bd'] == breakdown, recs)

def p_nabd(nabd, recs):
    if False:
        return 10
    return proportion(lambda r: r['nabd'] == nabd, recs)

def p_naaw(naaw, recs):
    if False:
        i = 10
        return i + 15
    return proportion(lambda r: r['naaw'] == naaw, recs)

def p_success_and_breakdown(breakdown, recs):
    if False:
        return 10
    return proportion(lambda r: success(r) and r['bd'] == breakdown, recs)

def p_success_given_breakdown(breakdown, recs):
    if False:
        i = 10
        return i + 15
    return p_success_and_breakdown(breakdown, recs) / p_breakdown(breakdown, recs)

def p_success_and_nabd(nabd, recs):
    if False:
        for i in range(10):
            print('nop')
    return proportion(lambda r: success(r) and r['nabd'] == nabd, recs)

def p_success_given_nabd(nabd, recs):
    if False:
        while True:
            i = 10
    return p_success_and_nabd(nabd, recs) / p_nabd(nabd, recs)

def p_success_and_naaw(naaw, recs):
    if False:
        return 10
    return proportion(lambda r: success(r) and r['naaw'] == naaw, recs)

def p_success_given_naaw(naaw, recs):
    if False:
        for i in range(10):
            print('nop')
    denom = p_naaw(naaw, recs)
    if denom > 0:
        return p_success_and_naaw(naaw, recs) / denom

def main():
    if False:
        while True:
            i = 10
    writer = csv.DictWriter(sys.stdout, delimiter='\t', fieldnames=['level', 'bd', 'nabd', 'naaw', 'success', 'duration'])
    writer.writeheader()
    records = [translate_raw_record(r) for r in load_records()]
    for rec in sorted(records, key=itemgetter('success')):
        writer.writerow(rec)
    print('p(success) = {}'.format(p_success(records)))
    print('p(bd==c) = {}'.format(p_breakdown('c', records)))
    print('p(success and bd=c) = {}'.format(p_success_and_breakdown('c', records)))
    for breakdown in ['', 'ag', 'c', 'pd']:
        print('p(success | bd={}) = {}'.format(breakdown, p_success_given_breakdown(breakdown, records)))
    for nabd in range(4):
        print('p(success | nabd={}) = {}'.format(nabd, p_success_given_nabd(nabd, records)))
    for naaw in range(6):
        print('p(success | naaw={}) = {}'.format(naaw, p_success_given_naaw(naaw, records)))
if __name__ == '__main__':
    main()