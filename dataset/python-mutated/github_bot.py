import os
import logging
import smtplib
import datetime
from operator import itemgetter
from email.mime.text import MIMEText
from email.header import Header
import requests
logging.basicConfig(level=logging.WARNING, filename=os.path.join(os.path.dirname(__file__), 'bot_log.txt'), filemode='a', format='%(name)s %(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s')
logger = logging.getLogger('Bot')
ACCOUNT = {'username': '', 'password': ''}
API = {'events': 'https://api.github.com/users/{username}/received_events'.format(username=ACCOUNT['username'])}
MAIL = {'mail': '', 'username': '', 'password': '', 'host': 'smtp.qq.com', 'port': 465}
RECEIVERS = []
DAY = 1
STARS = 100
CONTENT_FORMAT = '\n    <table border="2" align="center">\n      <tr>\n        <th>头像</th>\n        <th>用户名</th>\n        <th>项目名</th>\n        <th>starred 日期</th>\n        <th>项目 star 数量</th>\n      </tr>\n      {project_info_string}\n    </table>\n'

def get_data(page=1):
    if False:
        i = 10
        return i + 15
    '\n    从目标源获取数据\n    https://developer.github.com/v3/activity/events/\n    GitHub 规定：默认每页 30 条，最多 300 条目\n    '
    args = '?page={page}'.format(page=page)
    response = requests.get(API['events'] + args, auth=(ACCOUNT['username'], ACCOUNT['password']))
    status_code = response.status_code
    if status_code == 200:
        resp_json = response.json()
        return resp_json
    else:
        logging.error('请求 event api 失败：', status_code)
        return []

def get_all_data():
    if False:
        i = 10
        return i + 15
    '\n    获取全部 300 条的数据\n    https://developer.github.com/v3/activity/events/\n    GitHub 规定：默认每页 30 条，最多 300 条目\n    '
    all_data_list = []
    for i in range(10):
        response_json = get_data(i + 1)
        if response_json:
            all_data_list.extend(response_json)
    return all_data_list

def check_condition(data):
    if False:
        i = 10
        return i + 15
    '\n    过滤条件\n    '
    create_time = datetime.datetime.strptime(data['created_at'], '%Y-%m-%dT%H:%M:%SZ') + datetime.timedelta(hours=8)
    date_condition = create_time >= datetime.datetime.now() - datetime.timedelta(days=DAY)
    if data['type'] == 'WatchEvent' and date_condition:
        if data['payload']['action'] == 'started' and ACCOUNT['username'] not in data['repo']['name']:
            data['date_time'] = create_time.strftime('%Y-%m-%d %H:%M:%S')
            return True
    else:
        return False

def analyze(json_data):
    if False:
        while True:
            i = 10
    '\n    分析获取的数据\n    :return 符合过滤条件的数据\n    '
    result_data = []
    for fi_data in json_data:
        if check_condition(fi_data):
            result_data.append(fi_data)
    return result_data

def get_stars(data):
    if False:
        return 10
    '\n    获取stars数量，同时过滤掉stars数量少的项目\n    '
    project_info_list = []
    for fi_data in data:
        project_info = dict()
        project_info['user'] = fi_data['actor']['login']
        project_info['user_url'] = 'https://github.com/' + project_info['user']
        project_info['avatar_url'] = fi_data['actor']['avatar_url']
        project_info['repo_name'] = fi_data['repo']['name']
        project_info['repo_url'] = 'https://github.com/' + project_info['repo_name']
        project_info['date_time'] = fi_data['date_time']
        try:
            repo_stars = requests.get(fi_data['repo']['url'], timeout=2).json()
            if repo_stars:
                project_info['repo_stars'] = int(repo_stars['stargazers_count'])
            else:
                project_info['repo_stars'] = -1
        except Exception as e:
            project_info['repo_stars'] = -1
            logger.warning(u'获取：{} 项目星数失败——{}'.format(project_info['repo_name'], e))
        finally:
            if project_info['repo_stars'] >= STARS or project_info['repo_stars'] == -1:
                project_info_list.append(project_info)
    project_info_list = sorted(project_info_list, key=itemgetter('repo_stars'), reverse=True)
    return project_info_list

def make_content():
    if False:
        while True:
            i = 10
    '\n    生成发布邮件的内容\n    '
    json_data = get_all_data()
    data = analyze(json_data)
    content = []
    project_info_list = get_stars(data)
    for project_info in project_info_list:
        project_info_string = '<tr>\n                                <td><img src={avatar_url} width=32px></img></td>\n                                <td><a href={user_url}>{user}</a></td>\n                                <td><a href={repo_url}>{repo_name}</a></td>\n                                <td>{date_time}</td>\n                                <td>{repo_stars}</td>\n                              </tr>\n                           '.format(**project_info)
        content.append(project_info_string)
    return content

def send_email(receivers, email_content):
    if False:
        return 10
    '\n    发送邮件\n    '
    sender = MAIL['mail']
    receivers = receivers
    message = MIMEText(CONTENT_FORMAT.format(project_info_string=''.join(email_content)), 'html', 'utf-8')
    message['From'] = Header(u'GitHub 机器人', 'utf-8')
    message['To'] = Header(u'削微寒', 'utf-8')
    subject = u'今日 GitHub 热点'
    message['Subject'] = Header(subject, 'utf-8')
    try:
        smtp_obj = smtplib.SMTP_SSL()
        smtp_obj.connect(MAIL['host'], MAIL['port'])
        smtp_obj.login(MAIL['username'], MAIL['password'])
        smtp_obj.sendmail(sender, receivers, message.as_string())
    except smtplib.SMTPException as e:
        logger.error(u'无法发送邮件: {}'.format(e))
if __name__ == '__main__':
    content = make_content()
    send_email(RECEIVERS, content)