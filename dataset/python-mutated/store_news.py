import os, re
import pymysql
import setting
db_name = 'db_news'
conn = pymysql.connect(host=setting.MYSQL_REMOTE, port=3306, user=setting.MYSQL_REMOTE_USER, passwd=setting.MYSQL_PASSWORD, db=db_name, charset='utf8')
cur = conn.cursor()

def create_tb():
    if False:
        print('Hello World!')
    cmd = 'CREATE TABLE IF NOT EXISTS tb_cnstock(Date DATETIME ,Title VARCHAR (80),URL VARCHAR (80),PRIMARY KEY (URL)) charset=utf8;'
    try:
        cur.execute(cmd)
        conn.commit()
        return True
    except Exception as e:
        print(e)
        conn.rollback()
        return False

def save_sql():
    if False:
        for i in range(10):
            print('nop')
    if not create_tb():
        return False
    files = os.listdir('.')
    for file in files:
        years = re.findall('StockNews-\\[(.*?)\\]-\\[.*?\\].log', file)
        if len(years):
            print(file)
            cur_year = years[0].split('-')[0]
            f = open(file).readlines()
            loop = 4
            count = 1
            for content in f:
                s = content.strip()
                if count % loop == 2:
                    date_times = re.findall('(\\d+-\\d+ \\d+:\\d+)', s)[0]
                    date_times = cur_year + '-' + date_times
                    titles = re.findall('\\d+-\\d+ \\d+:\\d+(.*)', s)[0]
                    titles = titles.strip()
                if count % loop == 3:
                    url_link = re.findall('---> (.*)', s)[0]
                if count % loop == 0 and date_times and titles and url_link:
                    cmd = "INSERT INTO tb_cnstock (Date,Title,URL ) VALUES('%s','%s','%s');" % (date_times, titles, url_link)
                    print(cmd)
                    try:
                        cur.execute(cmd)
                        conn.commit()
                    except Exception as e:
                        print(e)
                        conn.rollback()
                count = count + 1
    conn.close()
    return True
if __name__ == '__main__':
    sub_folder = 'C:\\OneDrive\\Python\\all_in_one\\data'
    if not os.path.exists(sub_folder):
        os.mkdir(sub_folder)
    os.chdir(sub_folder)
    save_sql()