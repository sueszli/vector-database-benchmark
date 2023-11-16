/*
 * Copyright 2004-2012, Haiku, Inc. All rights reserved.
 * Copyright 2001, Dr. Zoidberg Enterprises. All rights reserved.
 * Copyright 2011, Clemens Zeidler. All rights reserved.
 *
 * Distributed under the terms of the MIT License.
 */


//!	C-mail API - compatibility function (stubs) for the old mail kit


#include <stdlib.h>
#include <string.h>

#include <Directory.h>
#include <E-mail.h>
#include <File.h>
#include <FindDirectory.h>
#include <List.h>
#include <Path.h>

#include <crypt.h>
#include <MailDaemon.h>
#include <MailMessage.h>
#include <MailSettings.h>


_EXPORT status_t
check_for_mail(int32* _incomingCount)
{
	status_t status = BMailDaemon().CheckMail();
	if (status != B_OK)
		return status;

	if (_incomingCount != NULL)
		*_incomingCount = BMailDaemon().CountNewMessages(true);

	return B_OK;
}


_EXPORT status_t
send_queued_mail(void)
{
	return BMailDaemon().SendQueuedMail();
}


_EXPORT int32
count_pop_accounts(void)
{
	BMailAccounts accounts;
	return accounts.CountAccounts();
}


_EXPORT status_t
get_mail_notification(mail_notification *notification)
{
	notification->alert = true;
	notification->beep = false;
	return B_OK;
}


_EXPORT status_t
set_mail_notification(mail_notification *, bool)
{
	return B_NO_REPLY;
}


_EXPORT status_t
get_pop_account(mail_pop_account* account, int32 index)
{
	BMailAccounts accounts;
	BMailAccountSettings* accountSettings = accounts.AccountAt(index);
	if (accountSettings == NULL)
		return B_BAD_INDEX;

	const BMessage& settings = accountSettings->InboundSettings();
	strcpy(account->pop_name, settings.FindString("username"));
	strcpy(account->pop_host, settings.FindString("server"));
	strcpy(account->real_name, accountSettings->RealName());
	strcpy(account->reply_to, accountSettings->ReturnAddress());

	const char* encryptedPassword = get_passwd(&settings, "cpasswd");
	const char* password = encryptedPassword;
	if (password == NULL)
		password = settings.FindString("password");
	strcpy(account->pop_password, password);

	delete[] encryptedPassword;
	return B_OK;
}


_EXPORT status_t
set_pop_account(mail_pop_account *, int32, bool)
{
	return B_NO_REPLY;
}


_EXPORT status_t
get_smtp_host(char* buffer)
{
	BMailAccounts accounts;
	BMailAccountSettings* account = accounts.AccountAt(
		BMailSettings().DefaultOutboundAccount());
	if (account == NULL)
		return B_ERROR;

	const BMessage& settings = account->OutboundSettings();

	if (!settings.HasString("server"))
		return B_NAME_NOT_FOUND;

	strcpy(buffer, settings.FindString("server"));
	return B_OK;
}


_EXPORT status_t
set_smtp_host(char * /* host */, bool /* save */)
{
	return B_NO_REPLY;
}


_EXPORT status_t
forward_mail(entry_ref *ref, const char *recipients, bool now)
{
	BFile file(ref, O_RDONLY);
	status_t status = file.InitCheck();
	if (status < B_OK)
		return status;

	BEmailMessage mail(&file);
	mail.SetTo(recipients);

	return mail.Send(now);
}
