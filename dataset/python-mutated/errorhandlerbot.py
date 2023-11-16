"""This is a very simple example on how one could implement a custom error handler."""
import html
import json
import logging
import traceback
from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import Application, CommandHandler, ContextTypes
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logging.getLogger('httpx').setLevel(logging.WARNING)
logger = logging.getLogger(__name__)
DEVELOPER_CHAT_ID = 123456789

async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Log the error and send a telegram message to notify the developer."""
    logger.error('Exception while handling an update:', exc_info=context.error)
    tb_list = traceback.format_exception(None, context.error, context.error.__traceback__)
    tb_string = ''.join(tb_list)
    update_str = update.to_dict() if isinstance(update, Update) else str(update)
    message = f'An exception was raised while handling an update\n<pre>update = {html.escape(json.dumps(update_str, indent=2, ensure_ascii=False))}</pre>\n\n<pre>context.chat_data = {html.escape(str(context.chat_data))}</pre>\n\n<pre>context.user_data = {html.escape(str(context.user_data))}</pre>\n\n<pre>{html.escape(tb_string)}</pre>'
    await context.bot.send_message(chat_id=DEVELOPER_CHAT_ID, text=message, parse_mode=ParseMode.HTML)

async def bad_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Raise an error to trigger the error handler."""
    await context.bot.wrong_method_name()

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Displays info on how to trigger an error."""
    await update.effective_message.reply_html(f'Use /bad_command to cause an error.\nYour chat id is <code>{update.effective_chat.id}</code>.')

def main() -> None:
    if False:
        while True:
            i = 10
    'Run the bot.'
    application = Application.builder().token('TOKEN').build()
    application.add_handler(CommandHandler('start', start))
    application.add_handler(CommandHandler('bad_command', bad_command))
    application.add_error_handler(error_handler)
    application.run_polling(allowed_updates=Update.ALL_TYPES)
if __name__ == '__main__':
    main()