"""
Simple Bot to handle '(my_)chat_member' updates.
Greets new users & keeps track of which chats the bot is in.

Usage:
Press Ctrl-C on the command line or send a signal to the process to stop the
bot.
"""
import logging
from typing import Optional, Tuple
from telegram import Chat, ChatMember, ChatMemberUpdated, Update
from telegram.constants import ParseMode
from telegram.ext import Application, ChatMemberHandler, CommandHandler, ContextTypes, MessageHandler, filters
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logging.getLogger('httpx').setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

def extract_status_change(chat_member_update: ChatMemberUpdated) -> Optional[Tuple[bool, bool]]:
    if False:
        for i in range(10):
            print('nop')
    "Takes a ChatMemberUpdated instance and extracts whether the 'old_chat_member' was a member\n    of the chat and whether the 'new_chat_member' is a member of the chat. Returns None, if\n    the status didn't change.\n    "
    status_change = chat_member_update.difference().get('status')
    (old_is_member, new_is_member) = chat_member_update.difference().get('is_member', (None, None))
    if status_change is None:
        return None
    (old_status, new_status) = status_change
    was_member = old_status in [ChatMember.MEMBER, ChatMember.OWNER, ChatMember.ADMINISTRATOR] or (old_status == ChatMember.RESTRICTED and old_is_member is True)
    is_member = new_status in [ChatMember.MEMBER, ChatMember.OWNER, ChatMember.ADMINISTRATOR] or (new_status == ChatMember.RESTRICTED and new_is_member is True)
    return (was_member, is_member)

async def track_chats(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Tracks the chats the bot is in."""
    result = extract_status_change(update.my_chat_member)
    if result is None:
        return
    (was_member, is_member) = result
    cause_name = update.effective_user.full_name
    chat = update.effective_chat
    if chat.type == Chat.PRIVATE:
        if not was_member and is_member:
            logger.info('%s unblocked the bot', cause_name)
            context.bot_data.setdefault('user_ids', set()).add(chat.id)
        elif was_member and (not is_member):
            logger.info('%s blocked the bot', cause_name)
            context.bot_data.setdefault('user_ids', set()).discard(chat.id)
    elif chat.type in [Chat.GROUP, Chat.SUPERGROUP]:
        if not was_member and is_member:
            logger.info('%s added the bot to the group %s', cause_name, chat.title)
            context.bot_data.setdefault('group_ids', set()).add(chat.id)
        elif was_member and (not is_member):
            logger.info('%s removed the bot from the group %s', cause_name, chat.title)
            context.bot_data.setdefault('group_ids', set()).discard(chat.id)
    elif not was_member and is_member:
        logger.info('%s added the bot to the channel %s', cause_name, chat.title)
        context.bot_data.setdefault('channel_ids', set()).add(chat.id)
    elif was_member and (not is_member):
        logger.info('%s removed the bot from the channel %s', cause_name, chat.title)
        context.bot_data.setdefault('channel_ids', set()).discard(chat.id)

async def show_chats(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Shows which chats the bot is in"""
    user_ids = ', '.join((str(uid) for uid in context.bot_data.setdefault('user_ids', set())))
    group_ids = ', '.join((str(gid) for gid in context.bot_data.setdefault('group_ids', set())))
    channel_ids = ', '.join((str(cid) for cid in context.bot_data.setdefault('channel_ids', set())))
    text = f'@{context.bot.username} is currently in a conversation with the user IDs {user_ids}. Moreover it is a member of the groups with IDs {group_ids} and administrator in the channels with IDs {channel_ids}.'
    await update.effective_message.reply_text(text)

async def greet_chat_members(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Greets new users in chats and announces when someone leaves"""
    result = extract_status_change(update.chat_member)
    if result is None:
        return
    (was_member, is_member) = result
    cause_name = update.chat_member.from_user.mention_html()
    member_name = update.chat_member.new_chat_member.user.mention_html()
    if not was_member and is_member:
        await update.effective_chat.send_message(f'{member_name} was added by {cause_name}. Welcome!', parse_mode=ParseMode.HTML)
    elif was_member and (not is_member):
        await update.effective_chat.send_message(f'{member_name} is no longer with us. Thanks a lot, {cause_name} ...', parse_mode=ParseMode.HTML)

async def start_private_chat(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Greets the user and records that they started a chat with the bot if it's a private chat.
    Since no `my_chat_member` update is issued when a user starts a private chat with the bot
    for the first time, we have to track it explicitly here.
    """
    user_name = update.effective_user.full_name
    chat = update.effective_chat
    if chat.type != Chat.PRIVATE or chat.id in context.bot_data.get('user_ids', set()):
        return
    logger.info('%s started a private chat with the bot', user_name)
    context.bot_data.setdefault('user_ids', set()).add(chat.id)
    await update.effective_message.reply_text(f"Welcome {user_name}. Use /show_chats to see what chats I'm in.")

def main() -> None:
    if False:
        while True:
            i = 10
    'Start the bot.'
    application = Application.builder().token('TOKEN').build()
    application.add_handler(ChatMemberHandler(track_chats, ChatMemberHandler.MY_CHAT_MEMBER))
    application.add_handler(CommandHandler('show_chats', show_chats))
    application.add_handler(ChatMemberHandler(greet_chat_members, ChatMemberHandler.CHAT_MEMBER))
    application.add_handler(MessageHandler(filters.ALL, start_private_chat))
    application.run_polling(allowed_updates=Update.ALL_TYPES)
if __name__ == '__main__':
    main()