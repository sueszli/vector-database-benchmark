import twint
import schedule
import time

def jobone():
    if False:
        i = 10
        return i + 15
    print('Fetching Tweets')
    c = twint.Config()
    c.Username = 'insert username here'
    c.Search = 'insert search term here'
    c.Since = '2018-01-01'
    c.Limit = 1000
    c.Store_csv = True
    c.Custom = ['date', 'time', 'username', 'tweet', 'link', 'likes', 'retweets', 'replies', 'mentions', 'hashtags']
    c.Output = 'filename.csv'
    twint.run.Search(c)

def jobtwo():
    if False:
        i = 10
        return i + 15
    print('Fetching Tweets')
    c = twint.Config()
    c.Username = 'insert username here'
    c.Search = 'insert search term here'
    c.Since = '2018-01-01'
    c.Limit = 1000
    c.Store_csv = True
    c.Custom = ['date', 'time', 'username', 'tweet', 'link', 'likes', 'retweets', 'replies', 'mentions', 'hashtags']
    c.Output = 'filename2.csv'
    twint.run.Search(c)
jobone()
jobtwo()
schedule.every().hour.do(jobone)
schedule.every().hour.do(jobtwo)
while True:
    schedule.run_pending()
    time.sleep(1)