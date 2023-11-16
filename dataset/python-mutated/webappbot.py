"""
Simple example of a Telegram WebApp which displays a color picker.
The static website for this website is hosted by the PTB team for your convenience.
Currently only showcases starting the WebApp via a KeyboardButton, as all other methods would
require a bot token.
"""
import json
import logging
from telegram import KeyboardButton, ReplyKeyboardMarkup, ReplyKeyboardRemove, Update, WebAppInfo
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logging.getLogger('httpx').setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message with a button that opens a the web app."""
    await update.message.reply_text('Please press the button below to choose a color via the WebApp.', reply_markup=ReplyKeyboardMarkup.from_button(KeyboardButton(text='Open the color picker!', web_app=WebAppInfo(url='https://python-telegram-bot.org/static/webappbot'))))

async def web_app_data(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Print the received data and remove the button."""
    data = json.loads(update.effective_message.web_app_data.data)
    await update.message.reply_html(text=f"You selected the color with the HEX value <code>{data['hex']}</code>. The corresponding RGB value is <code>{tuple(data['rgb'].values())}</code>.", reply_markup=ReplyKeyboardRemove())

def main() -> None:
    if False:
        i = 10
        return i + 15
    'Start the bot.'
    application = Application.builder().token('TOKEN').build()
    application.add_handler(CommandHandler('start', start))
    application.add_handler(MessageHandler(filters.StatusUpdate.WEB_APP_DATA, web_app_data))
    application.run_polling(allowed_updates=Update.ALL_TYPES)
if __name__ == '__main__':
    main()