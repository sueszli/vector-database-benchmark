"""Bot that explains Telegram's "Deep Linking Parameters" functionality.

This program is dedicated to the public domain under the CC0 license.

This Bot uses the Application class to handle the bot.

First, a few handler functions are defined. Then, those functions are passed to
the Application and registered at their respective places.
Then, the bot is started and runs until we press Ctrl-C on the command line.

Usage:
Deep Linking example. Send /start to get the link.
Press Ctrl-C on the command line or send a signal to the process to stop the
bot.
"""
import logging
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update, helpers
from telegram.constants import ParseMode
from telegram.ext import Application, CallbackQueryHandler, CommandHandler, ContextTypes, filters
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logging.getLogger('httpx').setLevel(logging.WARNING)
logger = logging.getLogger(__name__)
CHECK_THIS_OUT = 'check-this-out'
USING_ENTITIES = 'using-entities-here'
USING_KEYBOARD = 'using-keyboard-here'
SO_COOL = 'so-cool'
KEYBOARD_CALLBACKDATA = 'keyboard-callback-data'

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a deep-linked URL when the command /start is issued."""
    bot = context.bot
    url = helpers.create_deep_linked_url(bot.username, CHECK_THIS_OUT, group=True)
    text = 'Feel free to tell your friends about it:\n\n' + url
    await update.message.reply_text(text)

async def deep_linked_level_1(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Reached through the CHECK_THIS_OUT payload"""
    bot = context.bot
    url = helpers.create_deep_linked_url(bot.username, SO_COOL)
    text = "Awesome, you just accessed hidden functionality! Now let's get back to the private chat."
    keyboard = InlineKeyboardMarkup.from_button(InlineKeyboardButton(text='Continue here!', url=url))
    await update.message.reply_text(text, reply_markup=keyboard)

async def deep_linked_level_2(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Reached through the SO_COOL payload"""
    bot = context.bot
    url = helpers.create_deep_linked_url(bot.username, USING_ENTITIES)
    text = f'You can also mask the deep-linked URLs as links: <a href="{url}">▶️ CLICK HERE</a>.'
    await update.message.reply_text(text, parse_mode=ParseMode.HTML, disable_web_page_preview=True)

async def deep_linked_level_3(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Reached through the USING_ENTITIES payload"""
    await update.message.reply_text('It is also possible to make deep-linking using InlineKeyboardButtons.', reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton(text='Like this!', callback_data=KEYBOARD_CALLBACKDATA)]]))

async def deep_link_level_3_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Answers CallbackQuery with deeplinking url."""
    bot = context.bot
    url = helpers.create_deep_linked_url(bot.username, USING_KEYBOARD)
    await update.callback_query.answer(url=url)

async def deep_linked_level_4(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Reached through the USING_KEYBOARD payload"""
    payload = context.args
    await update.message.reply_text(f'Congratulations! This is as deep as it gets 👏🏻\n\nThe payload was: {payload}')

def main() -> None:
    if False:
        print('Hello World!')
    'Start the bot.'
    application = Application.builder().token('TOKEN').build()
    application.add_handler(CommandHandler('start', deep_linked_level_1, filters.Regex(CHECK_THIS_OUT)))
    application.add_handler(CommandHandler('start', deep_linked_level_2, filters.Regex(SO_COOL)))
    application.add_handler(CommandHandler('start', deep_linked_level_3, filters.Regex(USING_ENTITIES)))
    application.add_handler(CommandHandler('start', deep_linked_level_4, filters.Regex(USING_KEYBOARD)))
    application.add_handler(CallbackQueryHandler(deep_link_level_3_callback, pattern=KEYBOARD_CALLBACKDATA))
    application.add_handler(CommandHandler('start', start))
    application.run_polling(allowed_updates=Update.ALL_TYPES)
if __name__ == '__main__':
    main()