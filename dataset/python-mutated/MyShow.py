import logging
import GetData_zhihu
from flask import Flask, session, request
from flask import render_template, flash, redirect, url_for, jsonify
from flask_bootstrap import Bootstrap
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import Length, Email
app = Flask(__name__)
app.config['SECRET_KEY'] = 'hard to guess string'
bootstrap = Bootstrap(app=app)
zhihu_all_topics = GetData_zhihu.get_all_topics()
zhihu_all_topics_key = {}
zhihu_init_topics = GetData_zhihu.get_topic_data(topic_id='19559424', topic_name='数据分析')

class UserForm(FlaskForm):
    name = StringField('name', validators=[Email(message='邮箱格式不正确！')])
    password = PasswordField('password', validators=[Length(min=6, message='密码长度至少6位！')])
    submit = SubmitField('提 交')

@app.route('/', methods=['GET', 'POST'])
def temp():
    if False:
        while True:
            i = 10
    return redirect(url_for('index'))

@app.route('/index/', methods=['GET', 'POST'])
def index():
    if False:
        return 10
    user_form = UserForm()
    if request.method == 'POST':
        if user_form.validate_on_submit():
            session['username'] = user_form.name.data
        else:
            flash(user_form.errors['name'][0] if 'name' in user_form.errors else user_form.errors['password'][0])
    elif request.args.get('action') == 'login_out':
        flash('您已成功退出系统！')
        session['username'] = None
        return redirect(url_for('index'))
    elif request.args.get('action') == 'overview':
        session['page_type'] = 'overview'
        return redirect(url_for('index'))
    elif request.args.get('action') == 'zhihu_topics':
        session['page_type'] = 'zhihu_topics'
        return redirect(url_for('index'))
    return render_template('index.html', name=session.get('username'), page_type=session.get('page_type', 'overview'), form=user_form)

@app.route('/zhihu_get_topics_list/', methods=['post'])
def zhihu_get_topics_list():
    if False:
        i = 10
        return i + 15
    key = request.form.get('key')
    result = {'success': 1, 'data': []}
    if key:
        if key in zhihu_all_topics_key:
            result = zhihu_all_topics_key[key]
        else:
            for item in zhihu_all_topics:
                if item[1].find(key) >= 0:
                    result['data'].append({'id': item[0], 'name': item[1]})
            if len(result['data']) > 0:
                result['success'] = 1
                zhihu_all_topics_key[key] = result
                logging.debug('all_topics_key increase: %s', len(zhihu_all_topics_key))
    return jsonify(result)

@app.route('/zhihu_get_topics_data/', methods=['post'])
def zhihu_get_topics_data():
    if False:
        print('Hello World!')
    if request.form['id'] == '19554449':
        result = zhihu_init_topics
    else:
        result = GetData_zhihu.get_topic_data(request.form['id'], request.form['name'])
    return jsonify(result)

@app.errorhandler(404)
def page_not_found(excep):
    if False:
        print('Hello World!')
    return (render_template('error.html', error=excep, name=session.get('username')), 404)
if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s\t%(levelname)s\t%(message)s')
    logging.debug('app url_map: %s', app.url_map)
    app.run()