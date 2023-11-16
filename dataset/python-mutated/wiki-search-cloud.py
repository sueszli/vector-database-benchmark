from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt
import wikipedia
import sys
import warnings
warnings.filterwarnings('ignore')

def gen_cloud(topic):
    if False:
        return 10
    try:
        content = str(wikipedia.page(topic).content)
    except:
        print('Error, try searching something else...')
        sys.exit()
    STOPWORDS.add('==')
    stopwords = set(STOPWORDS)
    wordcloud = WordCloud(stopwords=stopwords, max_words=200, background_color='black', width=600, height=350).generate(content)
    return wordcloud

def save_cloud(wordcloud):
    if False:
        while True:
            i = 10
    wordcloud.to_file('./wordcloud.png')

def show_cloud(wordcloud):
    if False:
        return 10
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()
if __name__ == '__main__':
    topic = input('What do you want to search: ').strip()
    wordcloud = gen_cloud(topic)
    save_cloud(wordcloud)
    print('Wordcloud saved to current directory as wordcloud.png')
    desc = input('Do you wish to see the output(y/n): ')
    if desc == 'y':
        show_cloud(wordcloud)
    sys.exit()