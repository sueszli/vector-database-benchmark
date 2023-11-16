"""Tests for detect_intent_with_sentiment_analysis.py"""
from __future__ import absolute_import
import os
from detect_intent_event import detect_intent_with_event_input
PROJECT_ID = os.getenv('GOOGLE_CLOUD_PROJECT')
AGENT_ID = os.getenv('AGENT_ID')

def test_detect_intent_positive():
    if False:
        return 10
    response_text = detect_intent_with_event_input(PROJECT_ID, 'global', AGENT_ID, 'sys.no-match-default', 'en-us')
    assert response_text in ['Can you say that again?', "I didn't get that. Can you repeat?", "I didn't get that. Can you say it again?", 'I missed that, say that again?', 'I missed what you said. What was that?', 'One more time?', 'Say that one more time?', 'Sorry, can you say that again?', 'Sorry, could you say that again?', "Sorry, I didn't get that. Can you rephrase?", 'Sorry, what was that?', 'What was that?']