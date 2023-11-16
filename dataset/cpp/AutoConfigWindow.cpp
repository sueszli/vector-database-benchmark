/*
 * Copyright 2007-2015, Haiku, Inc. All rights reserved.
 * Copyright 2011, Clemens Zeidler <haiku@clemens-zeidler.de>
 * Distributed under the terms of the MIT License.
 */


#include "AutoConfigWindow.h"

#include "AutoConfig.h"
#include "AutoConfigView.h"

#include <Alert.h>
#include <Application.h>
#include <Catalog.h>
#include <Directory.h>
#include <File.h>
#include <FindDirectory.h>
#include <LayoutBuilder.h>
#include <MailSettings.h>
#include <Message.h>
#include <Path.h>

#include <crypt.h>


#undef B_TRANSLATION_CONTEXT
#define B_TRANSLATION_CONTEXT "AutoConfigWindow"


const float kSpacing = 10;


AutoConfigWindow::AutoConfigWindow(BRect rect, ConfigWindow *parent)
	:
	BWindow(rect, B_TRANSLATE("Create new account"), B_TITLED_WINDOW_LOOK,
		B_MODAL_APP_WINDOW_FEEL, B_NOT_ZOOMABLE | B_AVOID_FRONT
			| B_AUTO_UPDATE_SIZE_LIMITS, B_ALL_WORKSPACES),
	fParentWindow(parent),
	fAccount(NULL),
	fMainConfigState(true),
	fServerConfigState(false),
	fAutoConfigServer(true)
{
	fContainerView = new BGroupView("config container");

	fBackButton = new BButton("back", B_TRANSLATE("Back"),
		new BMessage(kBackMsg));
	fBackButton->SetEnabled(false);

	fNextButton = new BButton("next", B_TRANSLATE("Next"),
		new BMessage(kOkMsg));
	fNextButton->MakeDefault(true);

	fMainView = new AutoConfigView(fAutoConfig);
	fMainView->SetLabel(B_TRANSLATE("Account settings"));
	fContainerView->AddChild(fMainView);

	BLayoutBuilder::Group<>(this, B_VERTICAL)
		.SetInsets(B_USE_DEFAULT_SPACING)
		.Add(fContainerView)
		.AddGroup(B_HORIZONTAL)
			.AddGlue()
			.Add(fBackButton)
			.Add(fNextButton);

	// Add a shortcut to close the window using Command-W
	AddShortcut('W', B_COMMAND_KEY, new BMessage(B_QUIT_REQUESTED));
}


AutoConfigWindow::~AutoConfigWindow()
{
}


void
AutoConfigWindow::MessageReceived(BMessage* msg)
{
	status_t status = B_ERROR;
	BAlert* invalidMailAlert = NULL;

	switch (msg->what) {
		case kOkMsg:
			if (fMainConfigState) {
				fMainView->GetBasicAccountInfo(fAccountInfo);
				if (!fMainView->IsValidMailAddress(fAccountInfo.email)) {
					invalidMailAlert = new BAlert("invalidMailAlert",
						B_TRANSLATE("Enter a valid e-mail address."),
						B_TRANSLATE("OK"));
					invalidMailAlert->SetFlags(invalidMailAlert->Flags()
						| B_CLOSE_ON_ESCAPE);
					invalidMailAlert->Go();
					return;
				}
				if (fAutoConfigServer) {
					status = fAutoConfig.GetInfoFromMailAddress(
						fAccountInfo.email.String(),
						&fAccountInfo.providerInfo);
				}
				if (status == B_OK) {
					fParentWindow->Lock();
					GenerateBasicAccount();
					fParentWindow->Unlock();
					Quit();
				}
				fMainConfigState = false;
				fServerConfigState = true;
				fMainView->Hide();

				fServerView = new ServerSettingsView(fAccountInfo);
				fContainerView->AddChild(fServerView);

				fBackButton->SetEnabled(true);
				fNextButton->SetLabel(B_TRANSLATE("Finish"));
			} else {
				fServerView->GetServerInfo(fAccountInfo);
				fParentWindow->Lock();
				GenerateBasicAccount();
				fParentWindow->Unlock();
				Quit();
			}
			break;

		case kBackMsg:
			if (fServerConfigState) {
				fServerView->GetServerInfo(fAccountInfo);

				fMainConfigState = true;
				fServerConfigState = false;

				fContainerView->RemoveChild(fServerView);
				delete fServerView;

				fMainView->Show();
				fBackButton->SetEnabled(false);
			}
			break;

		case kServerChangedMsg:
			fAutoConfigServer = false;
			break;

		default:
			BWindow::MessageReceived(msg);
			break;
	}
}


bool
AutoConfigWindow::QuitRequested()
{
	return true;
}


BMailAccountSettings*
AutoConfigWindow::GenerateBasicAccount()
{
	if (!fAccount) {
		fParentWindow->Lock();
		fAccount = fParentWindow->AddAccount();
		fParentWindow->Unlock();
	}

	fAccount->SetName(fAccountInfo.accountName.String());
	fAccount->SetRealName(fAccountInfo.name.String());
	fAccount->SetReturnAddress(fAccountInfo.email.String());

	BMessage& inboundArchive = fAccount->InboundSettings();
	inboundArchive.MakeEmpty();
	BString inServerName;
	int32 authType = 0;
	int32 ssl = 0;
	if (fAccountInfo.inboundType == IMAP) {
		inServerName = fAccountInfo.providerInfo.imap_server;
		ssl = fAccountInfo.providerInfo.ssl_imap;
		fAccount->SetInboundAddOn("IMAP");
	} else {
		inServerName = fAccountInfo.providerInfo.pop_server;
		authType = fAccountInfo.providerInfo.authentification_pop;
		ssl = fAccountInfo.providerInfo.ssl_pop;
		fAccount->SetInboundAddOn("POP3");
	}
	inboundArchive.AddString("server", inServerName);
	inboundArchive.AddInt32("auth_method", authType);
	inboundArchive.AddInt32("flavor", ssl);
	inboundArchive.AddString("username", fAccountInfo.loginName);
	set_passwd(&inboundArchive, "cpasswd", fAccountInfo.password);
	inboundArchive.AddBool("leave_mail_on_server", true);
	inboundArchive.AddBool("delete_remote_when_local", true);

	BMessage& outboundArchive = fAccount->OutboundSettings();
	outboundArchive.MakeEmpty();
	fAccount->SetOutboundAddOn("SMTP");
	outboundArchive.AddString("server",
		fAccountInfo.providerInfo.smtp_server);
	outboundArchive.AddString("username", fAccountInfo.loginName);
	set_passwd(&outboundArchive, "cpasswd", fAccountInfo.password);
	outboundArchive.AddInt32("auth_method",
		fAccountInfo.providerInfo.authentification_smtp);
	outboundArchive.AddInt32("flavor",
		fAccountInfo.providerInfo.ssl_smtp);

	fParentWindow->Lock();
	fParentWindow->AccountUpdated(fAccount);
	fParentWindow->Unlock();

	return fAccount;
}
