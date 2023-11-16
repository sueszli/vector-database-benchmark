"""Tests for detect_intent_with_sentiment_analysis.py"""
from __future__ import absolute_import
import os
from detect_intent_with_intent_input import detect_intent_with_intent_input
PROJECT_ID = os.getenv('GOOGLE_CLOUD_PROJECT')
AGENT_ID = os.getenv('AGENT_ID')
INTENT_ID = os.getenv('INTENT_ID')

def test_detect_intent_with_intent_input():
    if False:
        i = 10
        return i + 15
    response_text = detect_intent_with_intent_input(PROJECT_ID, 'global', AGENT_ID, INTENT_ID, 'en-us')
    assert len(response_text) == 2
    assert response_text[0] == ["Let's find a one-way ticket for you. "]
    assert response_text[1] == ['Which city are you leaving from?']