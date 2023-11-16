"""Tests for detect_intent_with_sentiment_analysis.py"""
from __future__ import absolute_import
import os
from detect_intent_synthesize_tts_response import detect_intent_synthesize_tts_response
PROJECT_ID = os.getenv('GOOGLE_CLOUD_PROJECT')
AGENT_ID = os.getenv('AGENT_ID')

def test_detect_intent_positive(capsys, tmp_path_factory):
    if False:
        return 10
    output_file = tmp_path_factory.mktemp('data') / 'tmp.wav'
    detect_intent_synthesize_tts_response(PROJECT_ID, 'global', AGENT_ID, 'Perfect!', 'OUTPUT_AUDIO_ENCODING_LINEAR_16', 'en-us', output_file)
    (out, _) = capsys.readouterr()
    assert f'Audio content written to file: {output_file}' in out