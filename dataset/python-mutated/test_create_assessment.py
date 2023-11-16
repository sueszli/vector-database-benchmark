import multiprocessing
import os
import re
import time
from _pytest.capture import CaptureFixture
from flask import Flask, render_template, url_for
from google.cloud import recaptchaenterprise_v1
from google.cloud.recaptchaenterprise_v1 import Assessment
import pytest
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.webdriver import WebDriver
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
from annotate_assessment import annotate_assessment
from create_assessment import create_assessment
from create_mfa_assessment import create_mfa_assessment
from create_site_key import create_site_key
from delete_site_key import delete_site_key
from util import get_hashed_account_id
GOOGLE_CLOUD_PROJECT = os.environ['GOOGLE_CLOUD_PROJECT']
DOMAIN_NAME = 'localhost'
multiprocessing.set_start_method('fork')

@pytest.fixture(scope='session')
def app() -> Flask:
    if False:
        while True:
            i = 10
    app = Flask(__name__)

    @app.route('/assess/<site_key>', methods=['GET'])
    def assess(site_key: str) -> str:
        if False:
            i = 10
            return i + 15
        return render_template('index.html', site_key=site_key)

    @app.route('/', methods=['GET'])
    def index() -> str:
        if False:
            i = 10
            return i + 15
        return 'Helloworld!'
    return app

@pytest.fixture(scope='module')
def browser() -> WebDriver:
    if False:
        print('Hello World!')
    chrome_options = Options()
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--window-size=1420,1080')
    chrome_options.add_argument('--headless')
    chrome_options.add_argument('--disable-gpu')
    browser = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
    yield browser
    browser.close()

@pytest.fixture(scope='module')
def recaptcha_site_key() -> str:
    if False:
        while True:
            i = 10
    recaptcha_site_key = create_site_key(project_id=GOOGLE_CLOUD_PROJECT, domain_name=DOMAIN_NAME)
    yield recaptcha_site_key
    delete_site_key(project_id=GOOGLE_CLOUD_PROJECT, recaptcha_site_key=recaptcha_site_key)

@pytest.mark.usefixtures('live_server')
def test_assessment(capsys: CaptureFixture, recaptcha_site_key: str, browser: WebDriver) -> None:
    if False:
        for i in range(10):
            print('nop')
    (token, action) = get_token(recaptcha_site_key, browser)
    assessment_response = assess_token(recaptcha_site_key, token=token, action=action)
    score = str(assessment_response.risk_analysis.score)
    client = recaptchaenterprise_v1.RecaptchaEnterpriseServiceClient()
    assessment_name = client.parse_assessment_path(assessment_response.name).get('assessment')
    assert assessment_name != ''
    set_score(browser, score)
    annotate_assessment(project_id=GOOGLE_CLOUD_PROJECT, assessment_id=assessment_name)
    (out, _) = capsys.readouterr()
    assert re.search('Annotated response sent successfully !', out)

@pytest.mark.usefixtures('live_server')
def test_mfa_assessment(capsys: CaptureFixture, recaptcha_site_key: str, browser: WebDriver) -> None:
    if False:
        i = 10
        return i + 15
    (token, action) = get_token(recaptcha_site_key, browser)
    account_id = 'alicebob'
    key = 'your_secret_key'
    create_mfa_assessment(project_id=GOOGLE_CLOUD_PROJECT, recaptcha_site_key=recaptcha_site_key, token=token, recaptcha_action=action, hashed_account_id=get_hashed_account_id(account_id, key), email='abc@example.com', phone_number='+12345678901')
    (out, _) = capsys.readouterr()
    assert re.search('Result unspecified. Trigger MFA challenge in the client by passing the request token.', out)

def get_token(recaptcha_site_key: str, browser: WebDriver) -> tuple:
    if False:
        i = 10
        return i + 15
    browser.get(url_for('assess', site_key=recaptcha_site_key, _external=True))
    time.sleep(5)
    browser.find_element(By.ID, 'username').send_keys('username')
    browser.find_element(By.ID, 'password').send_keys('password')
    browser.find_element(By.ID, 'recaptchabutton').click()
    time.sleep(5)
    element = browser.find_element(By.CSS_SELECTOR, '#assessment')
    token = element.get_attribute('data-token')
    action = element.get_attribute('data-action')
    return (token, action)

def assess_token(recaptcha_site_key: str, token: str, action: str) -> Assessment:
    if False:
        print('Hello World!')
    return create_assessment(project_id=GOOGLE_CLOUD_PROJECT, recaptcha_site_key=recaptcha_site_key, token=token, recaptcha_action=action)

def set_score(browser: WebDriver, score: str) -> None:
    if False:
        for i in range(10):
            print('nop')
    browser.find_element(By.CSS_SELECTOR, '#assessment').send_keys(score)