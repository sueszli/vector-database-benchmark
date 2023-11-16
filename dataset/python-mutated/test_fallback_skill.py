from unittest import TestCase, mock
from mycroft.skills import FallbackSkill

def setup_fallback(fb_class):
    if False:
        while True:
            i = 10
    fb_skill = fb_class()
    fb_skill.bind(mock.Mock(name='bus'))
    fb_skill.initialize()
    return fb_skill

class TestFallbackSkill(TestCase):

    def test_life_cycle(self):
        if False:
            return 10
        'Test startup and shutdown of a fallback skill.\n\n        Ensure that an added handler is removed as part of default shutdown.\n        '
        self.assertEqual(len(FallbackSkill.fallback_handlers), 0)
        fb_skill = setup_fallback(SimpleFallback)
        self.assertEqual(len(FallbackSkill.fallback_handlers), 1)
        self.assertEqual(FallbackSkill.wrapper_map[0][0], fb_skill.fallback_handler)
        self.assertEqual(len(FallbackSkill.wrapper_map), 1)
        fb_skill.default_shutdown()
        self.assertEqual(len(FallbackSkill.fallback_handlers), 0)
        self.assertEqual(len(FallbackSkill.wrapper_map), 0)

    def test_manual_removal(self):
        if False:
            i = 10
            return i + 15
        'Test that the call to remove_fallback() removes the handler'
        self.assertEqual(len(FallbackSkill.fallback_handlers), 0)
        fb_skill = setup_fallback(SimpleFallback)
        self.assertEqual(len(FallbackSkill.fallback_handlers), 1)
        self.assertTrue(fb_skill.remove_fallback(fb_skill.fallback_handler))
        self.assertEqual(len(FallbackSkill.fallback_handlers), 0)
        self.assertEqual(len(FallbackSkill.wrapper_map), 0)
        self.assertFalse(fb_skill.remove_fallback(fb_skill.fallback_handler))

class SimpleFallback(FallbackSkill):
    """Simple fallback skill used for test."""

    def initialize(self):
        if False:
            i = 10
            return i + 15
        self.register_fallback(self.fallback_handler, 42)

    def fallback_handler(self):
        if False:
            i = 10
            return i + 15
        pass