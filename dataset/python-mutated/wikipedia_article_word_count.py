import json
try:
    from urllib.request import urlopen
except ImportError:
    from urllib2 import urlopen
import collections
import operator
import sys
WIKIPEDIA_ARTICLE_API_URL = 'https://en.wikipedia.org/w/api.php?action=query&titles=Spoon&prop=revisions&rvprop=content&format=json'

def download():
    if False:
        for i in range(10):
            print('nop')
    return urlopen(WIKIPEDIA_ARTICLE_API_URL).read()

def parse(json_data):
    if False:
        print('Hello World!')
    return json.loads(json_data)

def most_common_words(page):
    if False:
        while True:
            i = 10
    word_occurences = collections.defaultdict(int)
    for revision in page['revisions']:
        article = revision['*']
        for word in article.split():
            if len(word) < 2:
                continue
            word_occurences[word] += 1
    word_list = sorted(word_occurences.items(), key=operator.itemgetter(1), reverse=True)
    return word_list[0:5]

def main():
    if False:
        i = 10
        return i + 15
    data = parse(download())
    page = list(data['query']['pages'].values())[0]
    sys.stderr.write('This most common words were %s\n' % most_common_words(page))
if __name__ == '__main__':
    main()