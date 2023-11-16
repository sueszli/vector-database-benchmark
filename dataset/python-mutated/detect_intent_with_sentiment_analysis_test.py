"""Tests for detect_intent_with_sentiment_analysis.py"""
from __future__ import absolute_import
import os
import pytest
from detect_intent_with_sentiment_analysis import detect_intent_with_sentiment_analysis
PROJECT_ID = os.getenv('GOOGLE_CLOUD_PROJECT')
AGENT_ID = os.getenv('AGENT_ID')

@pytest.mark.parametrize('text, expected_score_min, expected_score_max', (['Perfect', -1, 1], ['I am not happy', -1, 1]))
def test_detect_intent_positive(text, expected_score_min, expected_score_max):
    if False:
        i = 10
        return i + 15
    score = detect_intent_with_sentiment_analysis(PROJECT_ID, 'global', AGENT_ID, text, 'en-us')
    assert expected_score_min < score < expected_score_max