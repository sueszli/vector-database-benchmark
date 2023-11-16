from collections import defaultdict
import pickle
import pymysql.cursors
import os
import random
from common.dataset.corpus import Corpus
from common.util.log_helper import LogHelper

def preprocess(p):
    if False:
        print('Hello World!')
    return p.replace(' ', '_').replace('(', '-LRB-').replace(')', '-RRB-').replace(':', '-COLON-').split('#')[0]
lut = dict()
LogHelper.setup()
pages = Corpus('page', 'data/fever', 50, lambda x: x)
for (page, doc) in pages:
    lut[page] = doc
claim_evidence = defaultdict(lambda : [])
connection = pymysql.connect(host=os.getenv('DB_HOST', 'localhost'), user=os.getenv('DB_USER', 'root'), password=os.getenv('DB_PASS', ''), db=os.getenv('DB_SCHEMA', 'fever'), charset='utf8mb4', cursorclass=pymysql.cursors.DictCursor)
data = []
try:
    with connection.cursor() as cursor:
        sql = "\n\n        select claim.id, claim.text,\n          CASE\n            WHEN annotation.verifiable =1 THEN 'VERIFIABLE'\n            WHEN annotation.verifiable =2 THEN 'NOT ENOUGH INFO'\n            WHEN annotation.verifiable =3 THEN 'NOT VERIFIABLE'\n            WHEN annotation.verifiable =4 THEN 'TYPO'\n          END as verifiable,\n          CASE\n            WHEN verdict=1 THEN 'SUPPORTS'\n            WHEN verdict=2 THEN 'REFUTES'\n          END as label, sentence.entity_id as entity, annotation.id as aid, verdict_line.page, verdict_line.line_number, testing, isOracle,isReval, isTestMode,isOracleMaster,isDiscounted from annotation\n        inner join claim on annotation.claim_id = claim.id\n        left join annotation_verdict on annotation.id = annotation_verdict.annotation_id\n        left join verdict_line on annotation_verdict.id = verdict_line.verdict_id\n        inner join sentence on claim.sentence_id = sentence.id\n        where isForReportingOnly = 0 and isTestMode = 0 and testing= 0\n\n        "
        cursor.execute(sql)
        result = cursor.fetchall()
        for res in result:
            claim_evidence[res['id']].append(res)
finally:
    connection.close()
r = random.Random()
keys = list(claim_evidence.keys())
r.shuffle(keys)
for i in range(0, 500, 100):
    done = []
    texts = dict()
    for datum in keys[i:i + 100]:
        try:
            id = datum
            text = claim_evidence[id][0]['text']
            isOracle = claim_evidence[id][0]['isOracle']
            isReval = claim_evidence[id][0]['isReval']
            originalPage = preprocess(claim_evidence[id][0]['entity'])
            verdicts = []
            texts[originalPage] = lut[originalPage]
            for evidence in claim_evidence[id]:
                verdicts.append({'label': evidence['label'], 'isOracleMaster': evidence['isOracleMaster'], 'verifiable': evidence['verifiable'], 'page': preprocess(evidence['page']), 'line': evidence['line_number']})
                texts[preprocess(evidence['page'])] = lut[preprocess(evidence['page'])]
            done.append({'id': id, 'text': text, 'isOracle': isOracle, 'isReval': isReval, 'original_page': originalPage, 'annotations': verdicts})
        except:
            continue
    import json
    with open('data/dump{0}.json'.format(i), 'w+') as f:
        json.dump({'annotations': done, 'texts': texts}, f)