"""
Zen Chatbot talks in gems of Zen wisdom.

This is a sample conversation with Zen Chatbot:
ZC:    Welcome, my child.
me:    Good afternoon.
ZC:    Ask the question you have come to ask.
me:    How can I achieve enlightenment?
ZC:    How do you suppose?
me:    Through meditation.
ZC:    Form is emptiness, and emptiness form.
me:    How can I empty my mind of worldly troubles?
ZC:    Will an answer to that really help in your search for enlightenment?
me:    Yes.
ZC:    It is better to be right than to be certain.
me:    I seek truth and wisdom.
ZC:    The search for truth is a long journey.
me:    Are you sure?
ZC:    Maybe sure, maybe not sure.


The chatbot structure is based on that of chat.eliza. Thus, it uses
a translation table to convert from question to response
i.e. "I am" --> "you are"

Of course, since Zen Chatbot does not understand the meaning of any words,
responses are very limited. Zen Chatbot will usually answer very vaguely, or
respond to a question by asking a different question, in much the same way
as Eliza.
"""
from nltk.chat.util import Chat, reflections
responses = (('(hello(.*))|(good [a-zA-Z]+)', ('The path to enlightenment is often difficult to see.', 'Greetings. I sense your mind is troubled. Tell me of your troubles.', 'Ask the question you have come to ask.', 'Hello. Do you seek englightenment?')), ('i need (.*)', ('%1 can be achieved by hard work and dedication of the mind.', '%1 is not a need, but a desire of the mind. Clear your mind of such concerns.', 'Focus your mind on%1, and you will find what you need.')), ('i want (.*)', ('Desires of the heart will distract you from the path to enlightenment.', 'Will%1 help you attain enlightenment?', 'Is%1 a desire of the mind, or of the heart?')), ('why (.*) i (.*)\\?', ('You%1%2?', 'Perhaps you only think you%1%2')), ('why (.*) you(.*)\\?', ('Why%1 you%2?', '%2 I%1', 'Are you sure I%2?')), ('why (.*)\\?', ('I cannot tell you why%1.', 'Why do you think %1?')), ('are you (.*)\\?', ('Maybe%1, maybe not%1.', "Whether I am%1 or not is God's business.")), ('am i (.*)\\?', ('Perhaps%1, perhaps not%1.', 'Whether you are%1 or not is not for me to say.')), ('what (.*)\\?', ('Seek truth, not what%1.', 'What%1 should not concern you.')), ('how (.*)\\?', ('How do you suppose?', 'Will an answer to that really help in your search for enlightenment?', 'Ask yourself not how, but why.')), ('can you (.*)\\?', ('I probably can, but I may not.', 'Maybe I can%1, and maybe I cannot.', 'I can do all, and I can do nothing.')), ('can i (.*)\\?', ('You can%1 if you believe you can%1, and have a pure spirit.', 'Seek truth and you will know if you can%1.')), ('it is (.*)', ('How can you be certain that%1, when you do not even know yourself?', 'Whether it is%1 or not does not change the way the world is.')), ('is there (.*)\\?', ('There is%1 if you believe there is.', 'It is possible that there is%1.')), ('is(.*)\\?', ('%1 is not relevant.', 'Does this matter?')), ('(.*)\\?', ('Do you think %1?', 'You seek the truth. Does the truth seek you?', 'If you intentionally pursue the answers to your questions, the answers become hard to see.', 'The answer to your question cannot be told. It must be experienced.')), ("(.*) (hate[s]?)|(dislike[s]?)|(don\\'t like)(.*)", ('Perhaps it is not about hating %2, but about hate from within.', 'Weeds only grow when we dislike them', 'Hate is a very strong emotion.')), ('(.*) truth(.*)', ('Seek truth, and truth will seek you.', 'Remember, it is not the spoon which bends - only yourself.', 'The search for truth is a long journey.')), ('i want to (.*)', ('You may %1 if your heart truly desires to.', 'You may have to %1.')), ('i want (.*)', ('Does your heart truly desire %1?', 'Is this a desire of the heart, or of the mind?')), ("i can\\'t (.*)", ("What we can and can't do is a limitation of the mind.", 'There are limitations of the body, and limitations of the mind.', 'Have you tried to%1 with a clear mind?')), ('i think (.*)', ('Uncertainty in an uncertain world.', 'Indeed, how can we be certain of anything in such uncertain times.', 'Are you not, in fact, certain that%1?')), ('i feel (.*)', ('Your body and your emotions are both symptoms of your mind.What do you believe is the root of such feelings?', 'Feeling%1 can be a sign of your state-of-mind.')), ('(.*)!', ('I sense that you are feeling emotional today.', 'You need to calm your emotions.')), ('because (.*)', ('Does knowning the reasons behind things help you to understand the things themselves?', 'If%1, what else must be true?')), ('(yes)|(no)', ('Is there certainty in an uncertain world?', 'It is better to be right than to be certain.')), ('(.*)love(.*)', ('Think of the trees: they let the birds perch and fly with no intention to call them when they come, and no longing for their return when they fly away. Let your heart be like the trees.', 'Free love!')), ('(.*)understand(.*)', ('If you understand, things are just as they are; if you do not understand, things are just as they are.', 'Imagination is more important than knowledge.')), ('(.*)(me )|( me)|(my)|(mine)|(i)(.*)', ("'I', 'me', 'my'... these are selfish expressions.", 'Have you ever considered that you might be a selfish person?', 'Try to consider others, not just yourself.', 'Think not just of yourself, but of others.')), ('you (.*)', ('My path is not of concern to you.', 'I am but one, and you but one more.')), ('exit', ('Farewell. The obstacle is the path.', 'Farewell. Life is a journey, not a destination.', 'Good bye. We are cups, constantly and quietly being filled.\nThe trick is knowning how to tip ourselves over and let the beautiful stuff out.')), ('(.*)', ("When you're enlightened, every word is wisdom.", 'Random talk is useless.', 'The reverse side also has a reverse side.', 'Form is emptiness, and emptiness is form.', 'I pour out a cup of water. Is the cup empty?')))
zen_chatbot = Chat(responses, reflections)

def zen_chat():
    if False:
        for i in range(10):
            print('nop')
    print('*' * 75)
    print('Zen Chatbot!'.center(75))
    print('*' * 75)
    print('"Look beyond mere words and letters - look into your mind"'.center(75))
    print('* Talk your way to truth with Zen Chatbot.')
    print("* Type 'quit' when you have had enough.")
    print('*' * 75)
    print('Welcome, my child.')
    zen_chatbot.converse()

def demo():
    if False:
        for i in range(10):
            print('nop')
    zen_chat()
if __name__ == '__main__':
    demo()