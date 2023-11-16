/*
 * Copyright 2006-2013, Haiku, Inc. All rights reserved.
 * Distributed under the terms of the MIT License.
 *
 * Authors:
 *		Dario Casalinuovo
 *		Axel Dörfler, axeld@pinc-software.de
 *		Rene Gollent, rene@gollent.com
 *		Hugo Santos, hugosantos@gmail.com
 */


#include "NetworkStatusView.h"

#include <algorithm>
#include <set>
#include <vector>

#include <arpa/inet.h>
#include <net/if.h>
#include <stdio.h>
#include <stdlib.h>
#include <strings.h>
#include <sys/socket.h>
#include <sys/sockio.h>
#include <unistd.h>

#include <AboutWindow.h>
#include <Alert.h>
#include <Application.h>
#include <Catalog.h>
#include <Bitmap.h>
#include <Deskbar.h>
#include <Dragger.h>
#include <Drivers.h>
#include <IconUtils.h>
#include <Locale.h>
#include <MenuItem.h>
#include <MessageRunner.h>
#include <NetworkInterface.h>
#include <NetworkRoster.h>
#include <PopUpMenu.h>
#include <Resources.h>
#include <Roster.h>
#include <String.h>
#include <TextView.h>

#include "NetworkStatus.h"
#include "NetworkStatusIcons.h"
#include "RadioView.h"
#include "WirelessNetworkMenuItem.h"


#undef B_TRANSLATION_CONTEXT
#define B_TRANSLATION_CONTEXT "NetworkStatusView"


static const char *kStatusDescriptions[] = {
	B_TRANSLATE("Unknown"),
	B_TRANSLATE("No link"),
	B_TRANSLATE("No stateful configuration"),
	B_TRANSLATE("Configuring"),
	B_TRANSLATE("Ready")
};

extern "C" _EXPORT BView *instantiate_deskbar_item(float maxWidth, float maxHeight);


const uint32 kMsgShowConfiguration = 'shcf';
const uint32 kMsgOpenNetworkPreferences = 'onwp';
const uint32 kMsgJoinNetwork = 'join';

const uint32 kMinIconWidth = 16;
const uint32 kMinIconHeight = 16;


//	#pragma mark - NetworkStatusView


NetworkStatusView::NetworkStatusView(BRect frame, int32 resizingMode,
		bool inDeskbar)
	: BView(frame, kDeskbarItemName, resizingMode,
		B_WILL_DRAW | B_TRANSPARENT_BACKGROUND | B_FRAME_EVENTS),
	fInDeskbar(inDeskbar)
{
	_Init();

	if (!inDeskbar) {
		// we were obviously added to a standard window - let's add a dragger
		frame.OffsetTo(B_ORIGIN);
		frame.top = frame.bottom - 7;
		frame.left = frame.right - 7;
		BDragger* dragger = new BDragger(frame, this,
			B_FOLLOW_RIGHT | B_FOLLOW_BOTTOM);
		AddChild(dragger);
	} else
		_Update();
}


NetworkStatusView::NetworkStatusView(BMessage* archive)
	: BView(archive),
	fInDeskbar(false)
{
	app_info info;
	if (be_app->GetAppInfo(&info) == B_OK
		&& !strcasecmp(info.signature, "application/x-vnd.Be-TSKB"))
		fInDeskbar = true;

	_Init();
}


NetworkStatusView::~NetworkStatusView()
{
}


void
NetworkStatusView::_Init()
{
	for (int i = 0; i < kStatusCount; i++) {
		fTrayIcons[i] = NULL;
		fNotifyIcons[i] = NULL;
	}

	_UpdateBitmaps();
}


void
NetworkStatusView::_UpdateBitmaps()
{
	for (int i = 0; i < kStatusCount; i++) {
		delete fTrayIcons[i];
		delete fNotifyIcons[i];
		fTrayIcons[i] = NULL;
		fNotifyIcons[i] = NULL;
	}

	image_info info;
	if (our_image(info) != B_OK)
		return;

	BFile file(info.name, B_READ_ONLY);
	if (file.InitCheck() < B_OK)
		return;

	BResources resources(&file);
#ifdef HAIKU_TARGET_PLATFORM_HAIKU
	if (resources.InitCheck() < B_OK)
		return;
#endif

	for (int i = 0; i < kStatusCount; i++) {
		const void* data = NULL;
		size_t size;
		data = resources.LoadResource(B_VECTOR_ICON_TYPE,
			kNetworkStatusNoDevice + i, &size);
		if (data != NULL) {
			// Scale main tray icon
			BBitmap* trayIcon = new BBitmap(Bounds(), B_RGBA32);
			if (trayIcon->InitCheck() == B_OK
				&& BIconUtils::GetVectorIcon((const uint8 *)data,
					size, trayIcon) == B_OK) {
				fTrayIcons[i] = trayIcon;
			} else
				delete trayIcon;

			// Scale notification icon
			BBitmap* notifyIcon = new BBitmap(BRect(0, 0, 31, 31), B_RGBA32);
			if (notifyIcon->InitCheck() == B_OK
				&& BIconUtils::GetVectorIcon((const uint8 *)data,
					size, notifyIcon) == B_OK) {
				fNotifyIcons[i] = notifyIcon;
			} else
				delete notifyIcon;
		}
	}
}


void
NetworkStatusView::_Quit()
{
	if (fInDeskbar) {
		BDeskbar deskbar;
		deskbar.RemoveItem(kDeskbarItemName);
	} else
		be_app->PostMessage(B_QUIT_REQUESTED);
}


NetworkStatusView*
NetworkStatusView::Instantiate(BMessage* archive)
{
	if (!validate_instantiation(archive, "NetworkStatusView"))
		return NULL;

	return new NetworkStatusView(archive);
}


status_t
NetworkStatusView::Archive(BMessage* archive, bool deep) const
{
	status_t status = BView::Archive(archive, deep);
	if (status == B_OK)
		status = archive->AddString("add_on", kSignature);
	if (status == B_OK)
		status = archive->AddString("class", "NetworkStatusView");

	return status;
}


void
NetworkStatusView::AttachedToWindow()
{
	BView::AttachedToWindow();

	SetViewColor(B_TRANSPARENT_COLOR);

	start_watching_network(
		B_WATCH_NETWORK_INTERFACE_CHANGES | B_WATCH_NETWORK_LINK_CHANGES, this);

	_Update();
}


void
NetworkStatusView::DetachedFromWindow()
{
	stop_watching_network(this);
}


void
NetworkStatusView::MessageReceived(BMessage* message)
{
	switch (message->what) {
		case B_NETWORK_MONITOR:
			_Update();
			break;

		case kMsgShowConfiguration:
			_ShowConfiguration(message);
			break;

		case kMsgOpenNetworkPreferences:
			_OpenNetworksPreferences();
			break;

		case kMsgJoinNetwork:
		{
			const char* deviceName;
			const char* name;
			BNetworkAddress address;
			if (message->FindString("device", &deviceName) == B_OK
				&& message->FindString("name", &name) == B_OK
				&& message->FindFlat("address", &address) == B_OK) {
				BNetworkDevice device(deviceName);
				status_t status = device.JoinNetwork(address);
				if (status != B_OK) {
					BString text
						= B_TRANSLATE("Could not join wireless network:\n");
					text << strerror(status);
					BAlert* alert = new BAlert(name, text.String(),
						B_TRANSLATE("OK"), NULL, NULL, B_WIDTH_AS_USUAL,
						B_STOP_ALERT);
					alert->SetFlags(alert->Flags() | B_CLOSE_ON_ESCAPE);
					alert->Go(NULL);
				}
			}
			break;
		}

		case B_ABOUT_REQUESTED:
			_AboutRequested();
			break;

		case B_QUIT_REQUESTED:
			_Quit();
			break;

		default:
			BView::MessageReceived(message);
			break;
	}
}


void
NetworkStatusView::FrameResized(float width, float height)
{
	_UpdateBitmaps();
	Invalidate();
}


void
NetworkStatusView::Draw(BRect updateRect)
{
	int32 status = kStatusUnknown;
	for (std::map<BString, int32>::const_iterator it
		= fInterfaceStatuses.begin(); it != fInterfaceStatuses.end(); ++it) {
		if (it->second > status)
			status = it->second;
	}

	if (fTrayIcons[status] == NULL)
		return;

	SetDrawingMode(B_OP_ALPHA);
	DrawBitmap(fTrayIcons[status]);
	SetDrawingMode(B_OP_COPY);
}


void
NetworkStatusView::_ShowConfiguration(BMessage* message)
{
	const char* name;
	if (message->FindString("interface", &name) != B_OK)
		return;

	BNetworkInterface networkInterface(name);
	if (!networkInterface.Exists())
		return;

	BString text(B_TRANSLATE("%ifaceName information:\n"));
	text.ReplaceFirst("%ifaceName", name);

	size_t boldLength = text.Length();

	int32 numAddrs = networkInterface.CountAddresses();
	for (int32 i = 0; i < numAddrs; i++) {
		BNetworkInterfaceAddress address;
		networkInterface.GetAddressAt(i, address);
		switch (address.Address().Family()) {
			case AF_INET:
				text << "\n" << B_TRANSLATE("IPv4 address:") << " "
					<< address.Address().ToString()
					<< "\n" << B_TRANSLATE("Broadcast:") << " "
					<< address.Broadcast().ToString()
					<< "\n" << B_TRANSLATE("Netmask:") << " "
					<< address.Mask().ToString()
					<< "\n";
				break;
			case AF_INET6:
				text << "\n" << B_TRANSLATE("IPv6 address:") << " "
					<< address.Address().ToString()
					<< "/" << address.Mask().PrefixLength()
					<< "\n";
				break;
			default:
				break;
		}
	}

	BAlert* alert = new BAlert(name, text.String(), B_TRANSLATE("OK"));
	alert->SetFlags(alert->Flags() | B_CLOSE_ON_ESCAPE);
	BTextView* view = alert->TextView();
	BFont font;

	view->SetStylable(true);
	view->GetFont(&font);
	font.SetFace(B_BOLD_FACE);
	view->SetFontAndColor(0, boldLength, &font);

	alert->Go(NULL);
}


void
NetworkStatusView::MouseDown(BPoint point)
{
	BPopUpMenu* menu = new BPopUpMenu(B_EMPTY_STRING, false, false);
	menu->SetAsyncAutoDestruct(true);
	menu->SetFont(be_plain_font);
	BString wifiInterface;
	BNetworkDevice device;

	if (!fInterfaceStatuses.empty()) {
		for (std::map<BString, int32>::const_iterator it
				= fInterfaceStatuses.begin(); it != fInterfaceStatuses.end();
				++it) {
			const BString& name = it->first;

			// we only show network of the first wireless device we find
			if (wifiInterface.IsEmpty()) {
				device.SetTo(name);
				if (device.IsWireless())
					wifiInterface = name;
			}
		}
	}

	// Add wireless networks, if any, first so that we can sort the menu

	if (!wifiInterface.IsEmpty()) {
		std::set<BNetworkAddress> associated;
		BNetworkAddress address;
		uint32 cookie = 0;
		while (device.GetNextAssociatedNetwork(cookie, address) == B_OK)
			associated.insert(address);

		uint32 networksCount = 0;
		wireless_network* networks = NULL;
		device.GetNetworks(networks, networksCount);
		for (uint32 i = 0; i < networksCount; i++) {
			const wireless_network& network = networks[i];
			BMessage* message = new BMessage(kMsgJoinNetwork);
			message->AddString("device", wifiInterface);
			message->AddString("name", network.name);
			message->AddFlat("address", &network.address);

			BMenuItem* item = new WirelessNetworkMenuItem(network, message);
			menu->AddItem(item);
			if (associated.find(network.address) != associated.end())
				item->SetMarked(true);
		}
		delete[] networks;

		if (networksCount == 0) {
			BMenuItem* item = new BMenuItem(
				B_TRANSLATE("<no wireless networks found>"), NULL);
			item->SetEnabled(false);
			menu->AddItem(item);
		} else
			menu->SortItems(WirelessNetworkMenuItem::CompareSignalStrength);

		menu->AddSeparatorItem();
	}

	// add action menu items

	menu->AddItem(new BMenuItem(B_TRANSLATE(
		"Open network preferences" B_UTF8_ELLIPSIS),
		new BMessage(kMsgOpenNetworkPreferences)));

	if (fInDeskbar) {
		menu->AddItem(new BMenuItem(B_TRANSLATE("Quit"),
			new BMessage(B_QUIT_REQUESTED)));
	}

	// Add wired interfaces to top of menu
	if (!fInterfaceStatuses.empty()) {
		int32 wiredCount = 0;
		for (std::map<BString, int32>::const_iterator it
				= fInterfaceStatuses.begin(); it != fInterfaceStatuses.end();
				++it) {
			const BString& name = it->first;

			BString label = name;
			label += ": ";
			label += kStatusDescriptions[
				_DetermineInterfaceStatus(name.String())];

			BMessage* info = new BMessage(kMsgShowConfiguration);
			info->AddString("interface", name.String());
			menu->AddItem(new BMenuItem(label.String(), info), wiredCount);
			wiredCount++;
		}

		// add separator item between wired and wireless networks
		// (or between wired networks and actions if no wireless found)
		if (wiredCount > 0)
			menu->AddItem(new BSeparatorItem(), wiredCount);
	}

	menu->SetTargetForItems(this);

	ConvertToScreen(&point);
	menu->Go(point, true, true, true);
}


void
NetworkStatusView::_AboutRequested()
{
	BAboutWindow* window = new BAboutWindow(
		B_TRANSLATE_SYSTEM_NAME("NetworkStatus"), kSignature);

	const char* authors[] = {
		"Axel Dörfler",
		"Hugo Santos",
		NULL
	};

	window->AddCopyright(2007, "Haiku, Inc.");
	window->AddAuthors(authors);

	window->Show();
}


int32
NetworkStatusView::_DetermineInterfaceStatus(
	const BNetworkInterface& interface)
{
	uint32 flags = interface.Flags();

	if ((flags & IFF_LINK) == 0)
		return kStatusNoLink;
	if ((flags & (IFF_UP | IFF_LINK | IFF_CONFIGURING)) == IFF_LINK)
		return kStatusLinkNoConfig;
	if ((flags & IFF_CONFIGURING) == IFF_CONFIGURING)
		return kStatusConnecting;
	if ((flags & (IFF_UP | IFF_LINK)) == (IFF_UP | IFF_LINK))
		return kStatusReady;

	return kStatusUnknown;
}


void
NetworkStatusView::_Update(bool force)
{
	BNetworkRoster& roster = BNetworkRoster::Default();
	BNetworkInterface interface;
	uint32 cookie = 0;
	std::set<BString> currentInterfaces;

	while (roster.GetNextInterface(&cookie, interface) == B_OK) {
		if ((interface.Flags() & IFF_LOOPBACK) == 0) {
			currentInterfaces.insert((BString)interface.Name());
			int32 oldStatus = kStatusUnknown;
			if (fInterfaceStatuses.find(interface.Name())
				!= fInterfaceStatuses.end()) {
				oldStatus = fInterfaceStatuses[interface.Name()];
			}
			int32 status = _DetermineInterfaceStatus(interface);
			if (oldStatus != status) {
				BNotification notification(B_INFORMATION_NOTIFICATION);
				notification.SetGroup(B_TRANSLATE("Network Status"));
				notification.SetTitle(interface.Name());
				notification.SetMessageID(interface.Name());
				notification.SetIcon(fNotifyIcons[status]);
				if (status == kStatusConnecting
					|| (status == kStatusReady
						&& oldStatus == kStatusConnecting)
					|| (status == kStatusNoLink
						&& oldStatus == kStatusReady)
					|| (status == kStatusNoLink
						&& oldStatus == kStatusConnecting)) {
					// A significant state change, raise notification.
					notification.SetContent(kStatusDescriptions[status]);
					notification.Send();
				}
				Invalidate();
			}
			fInterfaceStatuses[interface.Name()] = status;
		}
	}

	// Check every element in fInterfaceStatuses against our current interface
	// list. If it's not there, then the interface is not present anymore and
	// should be removed from fInterfaceStatuses.
	std::map<BString, int32>::iterator it = fInterfaceStatuses.begin();
	while (it != fInterfaceStatuses.end()) {
		std::map<BString, int32>::iterator backupIt = it;
		if (currentInterfaces.find(it->first) == currentInterfaces.end())
			fInterfaceStatuses.erase(it);
		it = ++backupIt;
	}
}


void
NetworkStatusView::_OpenNetworksPreferences()
{
	status_t status = be_roster->Launch("application/x-vnd.Haiku-Network");
	if (status != B_OK && status != B_ALREADY_RUNNING) {
		BString errorMessage(B_TRANSLATE("Launching the network preflet "
			"failed.\n\nError: "));
		errorMessage << strerror(status);
		BAlert* alert = new BAlert("launch error", errorMessage.String(),
			B_TRANSLATE("OK"));
		alert->SetFlags(alert->Flags() | B_CLOSE_ON_ESCAPE);

		// asynchronous alert in order to not block replicant host application
		alert->Go(NULL);
	}
}


//	#pragma mark -


extern "C" _EXPORT BView *
instantiate_deskbar_item(float maxWidth, float maxHeight)
{
	return new NetworkStatusView(BRect(0, 0, maxHeight - 1, maxHeight - 1),
		B_FOLLOW_LEFT | B_FOLLOW_TOP, true);
}
