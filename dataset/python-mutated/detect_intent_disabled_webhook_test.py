"""Tests for detect_intent_with_sentiment_analysis.py"""
from __future__ import absolute_import
import os
from detect_intent_disabled_webhook import detect_intent_disabled_webhook
PROJECT_ID = os.getenv('GOOGLE_CLOUD_PROJECT')
AGENT_ID = os.getenv('AGENT_ID')

def test_detect_intent_positive():
    if False:
        print('Hello World!')
    response_text_list = detect_intent_disabled_webhook(PROJECT_ID, 'global', AGENT_ID, 'Perfect!', 'en-us')
    for response_text in response_text_list:
        assert response_text[0] in ['You are welcome!', "It's my pleasure.", 'Anytime.', 'Of course.', "It's my pleasure to serve you."]