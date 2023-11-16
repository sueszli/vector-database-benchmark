import os
import random
import sqlite3
import json
from .ontology import all_domains, db_domains

class MultiWozDB(object):

    def __init__(self, db_dir, db_paths):
        if False:
            for i in range(10):
                print('nop')
        self.dbs = {}
        self.sql_dbs = {}
        for domain in all_domains:
            with open(os.path.join(db_dir, db_paths[domain]), 'r', encoding='utf-8') as f:
                self.dbs[domain] = json.loads(f.read().lower())

    def oneHotVector(self, domain, num):
        if False:
            for i in range(10):
                print('nop')
        'Return number of available entities for particular domain.'
        vector = [0, 0, 0, 0]
        if num == '':
            return vector
        if domain != 'train':
            if num == 0:
                vector = [1, 0, 0, 0]
            elif num == 1:
                vector = [0, 1, 0, 0]
            elif num <= 3:
                vector = [0, 0, 1, 0]
            else:
                vector = [0, 0, 0, 1]
        elif num == 0:
            vector = [1, 0, 0, 0]
        elif num <= 5:
            vector = [0, 1, 0, 0]
        elif num <= 10:
            vector = [0, 0, 1, 0]
        else:
            vector = [0, 0, 0, 1]
        return vector

    def addBookingPointer(self, turn_da):
        if False:
            i = 10
            return i + 15
        'Add information about availability of the booking option.'
        vector = [0, 0]
        if turn_da.get('booking-nobook'):
            vector = [1, 0]
        if turn_da.get('booking-book') or turn_da.get('train-offerbooked'):
            vector = [0, 1]
        return vector

    def addDBPointer(self, domain, match_num, return_num=False):
        if False:
            i = 10
            return i + 15
        'Create database pointer for all related domains.'
        if domain in db_domains:
            vector = self.oneHotVector(domain, match_num)
        else:
            vector = [0, 0, 0, 0]
        return vector

    def addDBIndicator(self, domain, match_num, return_num=False):
        if False:
            while True:
                i = 10
        'Create database indicator for all related domains.'
        if domain in db_domains:
            vector = self.oneHotVector(domain, match_num)
        else:
            vector = [0, 0, 0, 0]
        if vector == [0, 0, 0, 0]:
            indicator = '[db_nores]'
        else:
            indicator = '[db_%s]' % vector.index(1)
        return indicator

    def get_match_num(self, constraints, return_entry=False):
        if False:
            while True:
                i = 10
        'Create database pointer for all related domains.'
        match = {'general': ''}
        entry = {}
        for domain in all_domains:
            match[domain] = ''
            if domain in db_domains and constraints.get(domain):
                matched_ents = self.queryJsons(domain, constraints[domain])
                match[domain] = len(matched_ents)
                if return_entry:
                    entry[domain] = matched_ents
        if return_entry:
            return entry
        return match

    def pointerBack(self, vector, domain):
        if False:
            return 10
        if domain.endswith(']'):
            domain = domain[1:-1]
        if domain != 'train':
            nummap = {0: '0', 1: '1', 2: '2-3', 3: '>3'}
        else:
            nummap = {0: '0', 1: '1-5', 2: '6-10', 3: '>10'}
        if vector[:4] == [0, 0, 0, 0]:
            report = ''
        else:
            num = vector.index(1)
            report = domain + ': ' + nummap[num] + '; '
        if vector[-2] == 0 and vector[-1] == 1:
            report += 'booking: ok'
        if vector[-2] == 1 and vector[-1] == 0:
            report += 'booking: unable'
        return report

    def queryJsons(self, domain, constraints, exactly_match=True, return_name=False):
        if False:
            return 10
        "Returns the list of entities for a given domain\n        based on the annotation of the belief state\n        constraints: dict e.g. {'pricerange': 'cheap', 'area': 'west'}\n        "
        if domain == 'taxi':
            return [{'taxi_colors': random.choice(self.dbs[domain]['taxi_colors']), 'taxi_types': random.choice(self.dbs[domain]['taxi_types']), 'taxi_phone': [random.randint(1, 9) for _ in range(10)]}]
        if domain == 'police':
            return self.dbs['police']
        if domain == 'hospital':
            if constraints.get('department'):
                for entry in self.dbs['hospital']:
                    if entry.get('department') == constraints.get('department'):
                        return [entry]
            else:
                return []
        valid_cons = False
        for v in constraints.values():
            if v not in ['not mentioned', '']:
                valid_cons = True
        if not valid_cons:
            return []
        match_result = []
        if 'name' in constraints:
            for db_ent in self.dbs[domain]:
                if 'name' in db_ent:
                    cons = constraints['name']
                    dbn = db_ent['name']
                    if cons == dbn:
                        db_ent = db_ent if not return_name else db_ent['name']
                        match_result.append(db_ent)
                        return match_result
        for db_ent in self.dbs[domain]:
            match = True
            for (s, v) in constraints.items():
                if s == 'name':
                    continue
                if s in ['people', 'stay'] or (domain == 'hotel' and s == 'day') or (domain == 'restaurant' and s in ['day', 'time']):
                    continue
                skip_case = {"don't care": 1, "do n't care": 1, 'dont care': 1, 'not mentioned': 1, 'dontcare': 1, '': 1}
                if skip_case.get(v):
                    continue
                if s not in db_ent:
                    match = False
                    break
                v = 'yes' if v == 'free' else v
                if s in ['arrive', 'leave']:
                    try:
                        (h, m) = v.split(':')
                        v = int(h) * 60 + int(m)
                    except Exception:
                        match = False
                        break
                    time = int(db_ent[s].split(':')[0]) * 60 + int(db_ent[s].split(':')[1])
                    if s == 'arrive' and v > time:
                        match = False
                    if s == 'leave' and v < time:
                        match = False
                elif exactly_match and v != db_ent[s]:
                    match = False
                    break
                elif v not in db_ent[s]:
                    match = False
                    break
            if match:
                match_result.append(db_ent)
        if not return_name:
            return match_result
        else:
            if domain == 'train':
                match_result = [e['id'] for e in match_result]
            else:
                match_result = [e['name'] for e in match_result]
            return match_result

    def querySQL(self, domain, constraints):
        if False:
            i = 10
            return i + 15
        if not self.sql_dbs:
            for dom in db_domains:
                db = 'db/{}-dbase.db'.format(dom)
                conn = sqlite3.connect(db)
                c = conn.cursor()
                self.sql_dbs[dom] = c
        sql_query = 'select * from {}'.format(domain)
        flag = True
        for (key, val) in constraints.items():
            if val == '' or val == 'dontcare' or val == 'not mentioned' or (val == "don't care") or (val == 'dont care') or (val == "do n't care"):
                pass
            elif flag:
                sql_query += ' where '
                val2 = val.replace("'", "''")
                if key == 'leaveAt':
                    sql_query += ' ' + key + ' > ' + "'" + val2 + "'"
                elif key == 'arriveBy':
                    sql_query += ' ' + key + ' < ' + "'" + val2 + "'"
                else:
                    sql_query += ' ' + key + '=' + "'" + val2 + "'"
                flag = False
            else:
                val2 = val.replace("'", "''")
                if key == 'leaveAt':
                    sql_query += ' and ' + key + ' > ' + "'" + val2 + "'"
                elif key == 'arriveBy':
                    sql_query += ' and ' + key + ' < ' + "'" + val2 + "'"
                else:
                    sql_query += ' and ' + key + '=' + "'" + val2 + "'"
        try:
            print(sql_query)
            return self.sql_dbs[domain].execute(sql_query).fetchall()
        except Exception:
            return []
if __name__ == '__main__':
    dbPATHs = {'attraction': 'db/attraction_db_processed.json', 'hospital': 'db/hospital_db_processed.json', 'hotel': 'db/hotel_db_processed.json', 'police': 'db/police_db_processed.json', 'restaurant': 'db/restaurant_db_processed.json', 'taxi': 'db/taxi_db_processed.json', 'train': 'db/train_db_processed.json'}
    db = MultiWozDB(dbPATHs)
    while True:
        constraints = {}
        inp = input('input belief state in fomat: domain-slot1=value1;slot2=value2...\n')
        (domain, cons) = inp.split('-')
        for sv in cons.split(';'):
            (s, v) = sv.split('=')
            constraints[s] = v
        res = db.queryJsons(domain, constraints, return_name=True)
        report = []
        reidx = {'hotel': 8, 'restaurant': 6, 'attraction': 5, 'train': 1}
        print(constraints)
        print(res)
        print('count:', len(res), '\nnames:', report)