"""Basic example for a bot that can receive payment from user."""
import logging
from telegram import LabeledPrice, ShippingOption, Update
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, PreCheckoutQueryHandler, ShippingQueryHandler, filters
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logging.getLogger('httpx').setLevel(logging.WARNING)
logger = logging.getLogger(__name__)
PAYMENT_PROVIDER_TOKEN = 'PAYMENT_PROVIDER_TOKEN'

async def start_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Displays info on how to use the bot."""
    msg = 'Use /shipping to get an invoice for shipping-payment, or /noshipping for an invoice without shipping.'
    await update.message.reply_text(msg)

async def start_with_shipping_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Sends an invoice with shipping-payment."""
    chat_id = update.message.chat_id
    title = 'Payment Example'
    description = 'Payment Example using python-telegram-bot'
    payload = 'Custom-Payload'
    currency = 'USD'
    price = 1
    prices = [LabeledPrice('Test', price * 100)]
    await context.bot.send_invoice(chat_id, title, description, payload, PAYMENT_PROVIDER_TOKEN, currency, prices, need_name=True, need_phone_number=True, need_email=True, need_shipping_address=True, is_flexible=True)

async def start_without_shipping_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Sends an invoice without shipping-payment."""
    chat_id = update.message.chat_id
    title = 'Payment Example'
    description = 'Payment Example using python-telegram-bot'
    payload = 'Custom-Payload'
    currency = 'USD'
    price = 1
    prices = [LabeledPrice('Test', price * 100)]
    await context.bot.send_invoice(chat_id, title, description, payload, PAYMENT_PROVIDER_TOKEN, currency, prices)

async def shipping_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Answers the ShippingQuery with ShippingOptions"""
    query = update.shipping_query
    if query.invoice_payload != 'Custom-Payload':
        await query.answer(ok=False, error_message='Something went wrong...')
        return
    options = [ShippingOption('1', 'Shipping Option A', [LabeledPrice('A', 100)])]
    price_list = [LabeledPrice('B1', 150), LabeledPrice('B2', 200)]
    options.append(ShippingOption('2', 'Shipping Option B', price_list))
    await query.answer(ok=True, shipping_options=options)

async def precheckout_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Answers the PreQecheckoutQuery"""
    query = update.pre_checkout_query
    if query.invoice_payload != 'Custom-Payload':
        await query.answer(ok=False, error_message='Something went wrong...')
    else:
        await query.answer(ok=True)

async def successful_payment_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Confirms the successful payment."""
    await update.message.reply_text('Thank you for your payment!')

def main() -> None:
    if False:
        return 10
    'Run the bot.'
    application = Application.builder().token('TOKEN').build()
    application.add_handler(CommandHandler('start', start_callback))
    application.add_handler(CommandHandler('shipping', start_with_shipping_callback))
    application.add_handler(CommandHandler('noshipping', start_without_shipping_callback))
    application.add_handler(ShippingQueryHandler(shipping_callback))
    application.add_handler(PreCheckoutQueryHandler(precheckout_callback))
    application.add_handler(MessageHandler(filters.SUCCESSFUL_PAYMENT, successful_payment_callback))
    application.run_polling(allowed_updates=Update.ALL_TYPES)
if __name__ == '__main__':
    main()