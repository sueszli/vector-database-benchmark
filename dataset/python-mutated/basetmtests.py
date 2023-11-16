"""
Automated tests for checking transformation algorithms (the models package).
"""
import numpy as np

class TestBaseTopicModel:

    def test_print_topic(self):
        if False:
            print('Hello World!')
        topics = self.model.show_topics(formatted=True)
        for (topic_no, topic) in topics:
            self.assertTrue(isinstance(topic_no, int))
            self.assertTrue(isinstance(topic, str))

    def test_print_topics(self):
        if False:
            for i in range(10):
                print('nop')
        topics = self.model.print_topics()
        for (topic_no, topic) in topics:
            self.assertTrue(isinstance(topic_no, int))
            self.assertTrue(isinstance(topic, str))

    def test_show_topic(self):
        if False:
            for i in range(10):
                print('nop')
        topic = self.model.show_topic(1)
        for (k, v) in topic:
            self.assertTrue(isinstance(k, str))
            self.assertTrue(isinstance(v, (np.floating, float)))

    def test_show_topics(self):
        if False:
            return 10
        topics = self.model.show_topics(formatted=False)
        for (topic_no, topic) in topics:
            self.assertTrue(isinstance(topic_no, int))
            self.assertTrue(isinstance(topic, list))
            for (k, v) in topic:
                self.assertTrue(isinstance(k, str))
                self.assertTrue(isinstance(v, (np.floating, float)))

    def test_get_topics(self):
        if False:
            print('Hello World!')
        topics = self.model.get_topics()
        vocab_size = len(self.model.id2word)
        for topic in topics:
            self.assertTrue(isinstance(topic, np.ndarray))
            self.assertEqual(vocab_size, topic.shape[0])
            self.assertAlmostEqual(np.sum(topic), 1.0, 5)