import os
from topicmodel import cluster_stories
from topicmodel import load
stories_file = os.path.join(os.path.dirname(__file__), 'all_stories.json')

def test_end_to_end():
    if False:
        print('Hello World!')
    (word_clusters, document_clusters) = cluster_stories(load(stories_file))