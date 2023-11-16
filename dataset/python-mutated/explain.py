"""
Explain module
"""
import numpy as np

class Explain:
    """
    Explains the importance of each token in an input text element for a query. This method creates n permutations of the input text, where n
    is the number of tokens in the input text. This effectively masks each token to determine its importance to the query.
    """

    def __init__(self, embeddings):
        if False:
            print('Hello World!')
        '\n        Creates a new explain action.\n\n        Args:\n            embeddings: embeddings instance\n        '
        self.embeddings = embeddings
        self.content = embeddings.config.get('content')
        self.database = embeddings.database

    def __call__(self, queries, texts, limit):
        if False:
            for i in range(10):
                print('nop')
        '\n        Explains the importance of each input token in text for a list of queries.\n\n        Args:\n            query: input queries\n            texts: optional list of (text|list of tokens), otherwise runs search queries\n            limit: optional limit if texts is None\n\n        Returns:\n            list of dict per input text per query where a higher token scores represents higher importance relative to the query\n        '
        texts = self.texts(queries, texts, limit)
        return [self.explain(query, texts[x]) for (x, query) in enumerate(queries)]

    def texts(self, queries, texts, limit):
        if False:
            return 10
        '\n        Constructs lists of dict for each input query.\n\n        Args:\n            queries: input queries\n            texts: optional list of texts\n            limit: optional limit if texts is None\n\n        Returns:\n            lists of dict for each input query\n        '
        if texts:
            results = []
            for scores in self.embeddings.batchsimilarity(queries, texts):
                results.append([{'id': uid, 'text': texts[uid], 'score': score} for (uid, score) in scores])
            return results
        return self.embeddings.batchsearch(queries, limit) if self.content else [[]] * len(queries)

    def explain(self, query, texts):
        if False:
            i = 10
            return i + 15
        '\n        Explains the importance of each input token in text for a list of queries.\n\n        Args:\n            query: input query\n            texts: list of text\n\n        Returns:\n            list of {"id": value, "text": value, "score": value, "tokens": value} covering each input text element\n        '
        results = []
        if self.database:
            query = self.database.parse(query)
            query = ' '.join([' '.join(clause) for clause in query['similar']]) if 'similar' in query else None
        if not query or not texts or 'score' not in texts[0] or ('text' not in texts[0]):
            return texts
        for result in texts:
            text = result['text']
            tokens = text if isinstance(text, list) else text.split()
            permutations = []
            for i in range(len(tokens)):
                data = tokens.copy()
                data.pop(i)
                permutations.append([' '.join(data)])
            scores = [(i, result['score'] - np.abs(s)) for (i, s) in self.embeddings.similarity(query, permutations)]
            result['tokens'] = [(tokens[i], score) for (i, score) in sorted(scores, key=lambda x: x[0])]
            results.append(result)
        return sorted(results, key=lambda x: x['score'], reverse=True)