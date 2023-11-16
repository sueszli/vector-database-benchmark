"""Downloads the necessary NLTK corpora for TextBlob.

Usage: ::

    $ python -m textblob.download_corpora

If you only intend to use TextBlob's default models, you can use the "lite"
option: ::

    $ python -m textblob.download_corpora lite

"""
import sys
import nltk
MIN_CORPORA = ['brown', 'punkt', 'wordnet', 'averaged_perceptron_tagger']
ADDITIONAL_CORPORA = ['conll2000', 'movie_reviews']
ALL_CORPORA = MIN_CORPORA + ADDITIONAL_CORPORA

def download_lite():
    if False:
        i = 10
        return i + 15
    for each in MIN_CORPORA:
        nltk.download(each)

def download_all():
    if False:
        while True:
            i = 10
    for each in ALL_CORPORA:
        nltk.download(each)

def main():
    if False:
        return 10
    if 'lite' in sys.argv:
        download_lite()
    else:
        download_all()
    print('Finished.')
if __name__ == '__main__':
    main()