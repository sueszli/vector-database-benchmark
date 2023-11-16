class BaseTopicModel:

    def print_topic(self, topicno, topn=10):
        if False:
            for i in range(10):
                print('nop')
        'Get a single topic as a formatted string.\n\n        Parameters\n        ----------\n        topicno : int\n            Topic id.\n        topn : int\n            Number of words from topic that will be used.\n\n        Returns\n        -------\n        str\n            String representation of topic, like \'-0.340 * "category" + 0.298 * "$M$" + 0.183 * "algebra" + ... \'.\n\n        '
        return ' + '.join(('%.3f*"%s"' % (v, k) for (k, v) in self.show_topic(topicno, topn)))

    def print_topics(self, num_topics=20, num_words=10):
        if False:
            i = 10
            return i + 15
        'Get the most significant topics (alias for `show_topics()` method).\n\n        Parameters\n        ----------\n        num_topics : int, optional\n            The number of topics to be selected, if -1 - all topics will be in result (ordered by significance).\n        num_words : int, optional\n            The number of words to be included per topics (ordered by significance).\n\n        Returns\n        -------\n        list of (int, list of (str, float))\n            Sequence with (topic_id, [(word, value), ... ]).\n\n        '
        return self.show_topics(num_topics=num_topics, num_words=num_words, log=True)

    def get_topics(self):
        if False:
            i = 10
            return i + 15
        'Get words X topics matrix.\n\n        Returns\n        --------\n        numpy.ndarray:\n            The term topic matrix learned during inference, shape (`num_topics`, `vocabulary_size`).\n\n        Raises\n        ------\n        NotImplementedError\n\n        '
        raise NotImplementedError