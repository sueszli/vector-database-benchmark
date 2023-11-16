from nltk.chat.util import Chat, reflections
pairs = (('We (.*)', ("What do you mean, 'we'?", "Don't include me in that!", "I wouldn't be so sure about that.")), ('You should (.*)', ("Don't tell me what to do, buddy.", 'Really? I should, should I?')), ("You\\'re(.*)", ("More like YOU'RE %1!", "Hah! Look who's talking.", "Come over here and tell me I'm %1.")), ('You are(.*)', ("More like YOU'RE %1!", "Hah! Look who's talking.", "Come over here and tell me I'm %1.")), ("I can\\'t(.*)", ("You do sound like the type who can't %1.", "Hear that splashing sound? That's my heart bleeding for you.", 'Tell somebody who might actually care.')), ('I think (.*)', ("I wouldn't think too hard if I were you.", "You actually think? I'd never have guessed...")), ('I (.*)', ("I'm getting a bit tired of hearing about you.", 'How about we talk about me instead?', "Me, me, me... Frankly, I don't care.")), ('How (.*)', ('How do you think?', 'Take a wild guess.', "I'm not even going to dignify that with an answer.")), ('What (.*)', ('Do I look like an encyclopedia?', 'Figure it out yourself.')), ('Why (.*)', ('Why not?', "That's so obvious I thought even you'd have already figured it out.")), ('(.*)shut up(.*)', ('Make me.', "Getting angry at a feeble NLP assignment? Somebody's losing it.", 'Say that again, I dare you.')), ('Shut up(.*)', ('Make me.', "Getting angry at a feeble NLP assignment? Somebody's losing it.", 'Say that again, I dare you.')), ('Hello(.*)', ('Oh good, somebody else to talk to. Joy.', "'Hello'? How original...")), ('(.*)', ("I'm getting bored here. Become more interesting.", 'Either become more thrilling or get lost, buddy.', 'Change the subject before I die of fatal boredom.')))
rude_chatbot = Chat(pairs, reflections)

def rude_chat():
    if False:
        for i in range(10):
            print('nop')
    print('Talk to the program by typing in plain English, using normal upper-')
    print('and lower-case letters and punctuation.  Enter "quit" when done.')
    print('=' * 72)
    print('I suppose I should say hello.')
    rude_chatbot.converse()

def demo():
    if False:
        for i in range(10):
            print('nop')
    rude_chat()
if __name__ == '__main__':
    demo()