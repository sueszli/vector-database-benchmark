from urllib.parse import quote_plus
import praw
QUESTIONS = ['what is', 'who is', 'what are']
REPLY_TEMPLATE = '[Let me google that for you](https://lmgtfy.com/?q={})'

def main():
    if False:
        while True:
            i = 10
    reddit = praw.Reddit(client_id='CLIENT_ID', client_secret='CLIENT_SECRET', password='PASSWORD', user_agent='LMGTFY (by u/USERNAME)', username='USERNAME')
    subreddit = reddit.subreddit('AskReddit')
    for submission in subreddit.stream.submissions():
        process_submission(submission)

def process_submission(submission):
    if False:
        print('Hello World!')
    if len(submission.title.split()) > 10:
        return
    normalized_title = submission.title.lower()
    for question_phrase in QUESTIONS:
        if question_phrase in normalized_title:
            url_title = quote_plus(submission.title)
            reply_text = REPLY_TEMPLATE.format(url_title)
            print(f'Replying to: {submission.title}')
            submission.reply(reply_text)
            break
if __name__ == '__main__':
    main()