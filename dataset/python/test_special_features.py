#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import unittest

import speech_recognition as sr


class TestSpecialFeatures(unittest.TestCase):
    def setUp(self):
        self.AUDIO_FILE_EN = os.path.join(os.path.dirname(os.path.realpath(__file__)), "english.wav")
        self.addTypeEqualityFunc(str, self.assertSameWords)

    def test_sphinx_keywords(self):
        r = sr.Recognizer()
        with sr.AudioFile(self.AUDIO_FILE_EN) as source: audio = r.record(source)
        self.assertEqual(r.recognize_sphinx(audio, keyword_entries=[("one", 1.0), ("two", 1.0), ("three", 1.0)]), "three two one")
        self.assertEqual(r.recognize_sphinx(audio, keyword_entries=[("wan", 0.95), ("too", 1.0), ("tree", 1.0)]), "tree too wan")
        self.assertEqual(r.recognize_sphinx(audio, keyword_entries=[("un", 0.95), ("to", 1.0), ("tee", 1.0)]), "tee to un")

    def assertSameWords(self, tested, reference, msg=None):
        set_tested = set(tested.split())
        set_reference = set(reference.split())
        if set_tested != set_reference:
            raise self.failureException(msg if msg is not None else "%r doesn't consist of the same words as %r" % (tested, reference))


if __name__ == "__main__":
    unittest.main()
