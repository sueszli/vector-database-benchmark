"""
Basic example for a bot that works with polls. Only 3 people are allowed to interact with each
poll/quiz the bot generates. The preview command generates a closed poll/quiz, exactly like the
one the user sends the bot
"""
import logging
from telegram import KeyboardButton, KeyboardButtonPollType, Poll, ReplyKeyboardMarkup, ReplyKeyboardRemove, Update
from telegram.constants import ParseMode
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, PollAnswerHandler, PollHandler, filters
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logging.getLogger('httpx').setLevel(logging.WARNING)
logger = logging.getLogger(__name__)
TOTAL_VOTER_COUNT = 3

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Inform user about what this bot can do"""
    await update.message.reply_text('Please select /poll to get a Poll, /quiz to get a Quiz or /preview to generate a preview for your poll')

async def poll(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Sends a predefined poll"""
    questions = ['Good', 'Really good', 'Fantastic', 'Great']
    message = await context.bot.send_poll(update.effective_chat.id, 'How are you?', questions, is_anonymous=False, allows_multiple_answers=True)
    payload = {message.poll.id: {'questions': questions, 'message_id': message.message_id, 'chat_id': update.effective_chat.id, 'answers': 0}}
    context.bot_data.update(payload)

async def receive_poll_answer(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Summarize a users poll vote"""
    answer = update.poll_answer
    answered_poll = context.bot_data[answer.poll_id]
    try:
        questions = answered_poll['questions']
    except KeyError:
        return
    selected_options = answer.option_ids
    answer_string = ''
    for question_id in selected_options:
        if question_id != selected_options[-1]:
            answer_string += questions[question_id] + ' and '
        else:
            answer_string += questions[question_id]
    await context.bot.send_message(answered_poll['chat_id'], f'{update.effective_user.mention_html()} feels {answer_string}!', parse_mode=ParseMode.HTML)
    answered_poll['answers'] += 1
    if answered_poll['answers'] == TOTAL_VOTER_COUNT:
        await context.bot.stop_poll(answered_poll['chat_id'], answered_poll['message_id'])

async def quiz(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a predefined poll"""
    questions = ['1', '2', '4', '20']
    message = await update.effective_message.reply_poll('How many eggs do you need for a cake?', questions, type=Poll.QUIZ, correct_option_id=2)
    payload = {message.poll.id: {'chat_id': update.effective_chat.id, 'message_id': message.message_id}}
    context.bot_data.update(payload)

async def receive_quiz_answer(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Close quiz after three participants took it"""
    if update.poll.is_closed:
        return
    if update.poll.total_voter_count == TOTAL_VOTER_COUNT:
        try:
            quiz_data = context.bot_data[update.poll.id]
        except KeyError:
            return
        await context.bot.stop_poll(quiz_data['chat_id'], quiz_data['message_id'])

async def preview(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Ask user to create a poll and display a preview of it"""
    button = [[KeyboardButton('Press me!', request_poll=KeyboardButtonPollType())]]
    message = 'Press the button to let the bot generate a preview for your poll'
    await update.effective_message.reply_text(message, reply_markup=ReplyKeyboardMarkup(button, one_time_keyboard=True))

async def receive_poll(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """On receiving polls, reply to it by a closed poll copying the received poll"""
    actual_poll = update.effective_message.poll
    await update.effective_message.reply_poll(question=actual_poll.question, options=[o.text for o in actual_poll.options], is_closed=True, reply_markup=ReplyKeyboardRemove())

async def help_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Display a help message"""
    await update.message.reply_text('Use /quiz, /poll or /preview to test this bot.')

def main() -> None:
    if False:
        while True:
            i = 10
    'Run bot.'
    application = Application.builder().token('TOKEN').build()
    application.add_handler(CommandHandler('start', start))
    application.add_handler(CommandHandler('poll', poll))
    application.add_handler(CommandHandler('quiz', quiz))
    application.add_handler(CommandHandler('preview', preview))
    application.add_handler(CommandHandler('help', help_handler))
    application.add_handler(MessageHandler(filters.POLL, receive_poll))
    application.add_handler(PollAnswerHandler(receive_poll_answer))
    application.add_handler(PollHandler(receive_quiz_answer))
    application.run_polling(allowed_updates=Update.ALL_TYPES)
if __name__ == '__main__':
    main()