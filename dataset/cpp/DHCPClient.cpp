/*
 * Copyright 2006-2018, Haiku, Inc. All Rights Reserved.
 * Distributed under the terms of the MIT License.
 *
 * Authors:
 *		Axel Dörfler, axeld@pinc-software.de
 *		Vegard Wærp, vegarwa@online.no
 *		Philippe Houdoin, <phoudoin at haiku-os dot org>
 */


#include "DHCPClient.h"

#include <Message.h>
#include <MessageRunner.h>
#include <NetworkDevice.h>
#include <NetworkInterface.h>

#include <algorithm>
#include <arpa/inet.h>
#include <errno.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <syslog.h>
#include <sys/sockio.h>
#include <sys/time.h>
#include <unistd.h>

#include <Debug.h>
#include <Message.h>
#include <MessageRunner.h>

#include "NetServer.h"


// See RFC 2131 for DHCP, see RFC 1533 for BOOTP/DHCP options

#define DHCP_CLIENT_PORT	68
#define DHCP_SERVER_PORT	67

#define DEFAULT_TIMEOUT		0.25	// secs
#define MAX_TIMEOUT			64		// secs

#define AS_USECS(t) (1000000 * t)

#define MAX_RETRIES			5

enum message_opcode {
	BOOT_REQUEST = 1,
	BOOT_REPLY
};

enum message_option {
	OPTION_MAGIC = 0x63825363,

	// generic options
	OPTION_PAD = 0,
	OPTION_END = 255,
	OPTION_SUBNET_MASK = 1,
	OPTION_TIME_OFFSET = 2,
	OPTION_ROUTER_ADDRESS = 3,
	OPTION_DOMAIN_NAME_SERVER = 6,
	OPTION_HOST_NAME = 12,
	OPTION_DOMAIN_NAME = 15,
	OPTION_MAX_DATAGRAM_SIZE = 22,
	OPTION_INTERFACE_MTU = 26,
	OPTION_BROADCAST_ADDRESS = 28,
	OPTION_NETWORK_TIME_PROTOCOL_SERVERS = 42,
	OPTION_NETBIOS_NAME_SERVER = 44,
	OPTION_NETBIOS_SCOPE = 47,

	// DHCP specific options
	OPTION_REQUEST_IP_ADDRESS = 50,
	OPTION_ADDRESS_LEASE_TIME = 51,
	OPTION_OVERLOAD = 52,
	OPTION_MESSAGE_TYPE = 53,
	OPTION_SERVER_ADDRESS = 54,
	OPTION_REQUEST_PARAMETERS = 55,
	OPTION_ERROR_MESSAGE = 56,
	OPTION_MAX_MESSAGE_SIZE = 57,
	OPTION_RENEWAL_TIME = 58,
	OPTION_REBINDING_TIME = 59,
	OPTION_CLASS_IDENTIFIER = 60,
	OPTION_CLIENT_IDENTIFIER = 61,
};

enum message_type {
	DHCP_NONE = 0,
	DHCP_DISCOVER = 1,
	DHCP_OFFER = 2,
	DHCP_REQUEST = 3,
	DHCP_DECLINE = 4,
	DHCP_ACK = 5,
	DHCP_NACK = 6,
	DHCP_RELEASE = 7,
	DHCP_INFORM = 8
};

struct dhcp_option_cookie {
	dhcp_option_cookie()
		:
		state(0),
		file_has_options(false),
		server_name_has_options(false)
	{
	}

	const uint8* next;
	uint8	state;
	bool	file_has_options;
	bool	server_name_has_options;
};

struct dhcp_message {
	dhcp_message(message_type type);

	uint8		opcode;
	uint8		hardware_type;
	uint8		hardware_address_length;
	uint8		hop_count;
	uint32		transaction_id;
	uint16		seconds_since_start;
	uint16		flags;
	in_addr_t	client_address;
	in_addr_t	your_address;
	in_addr_t	server_address;
	in_addr_t	gateway_address;
	uint8		mac_address[16];
	uint8		server_name[64];
	uint8		file[128];
	uint32		options_magic;
	uint8		options[1260];

	size_t MinSize() const { return 576; }
	size_t Size() const;

	bool HasOptions() const;
	bool NextOption(dhcp_option_cookie& cookie, message_option& option,
		const uint8*& data, size_t& size) const;
	const uint8* FindOption(message_option which) const;
	const uint8* LastOption() const;
	message_type Type() const;

	uint8* PrepareMessage(uint8 type);
	uint8* PutOption(uint8* options, message_option option);
	uint8* PutOption(uint8* options, message_option option, uint8 data);
	uint8* PutOption(uint8* options, message_option option, uint16 data);
	uint8* PutOption(uint8* options, message_option option, uint32 data);
	uint8* PutOption(uint8* options, message_option option, const uint8* data,
		uint32 size);
	uint8* FinishOptions(uint8* options);

	static const char* TypeToString(message_type type);
} _PACKED;

struct socket_timeout {
	socket_timeout(int socket)
		:
		timeout(time_t(AS_USECS(DEFAULT_TIMEOUT))),
		tries(0)
	{
		UpdateSocket(socket);
	}

	bigtime_t timeout; // in micro secs
	uint8 tries;

	bool Shift(int socket, bigtime_t stateMaxTime, const char* device);
	void UpdateSocket(int socket) const;
};

#define DHCP_FLAG_BROADCAST		0x8000

#define ARP_HARDWARE_TYPE_ETHER	1

const uint32 kMsgLeaseTime = 'lstm';

static const uint8 kRequestParameters[] = {
	OPTION_SUBNET_MASK, OPTION_ROUTER_ADDRESS,
	OPTION_DOMAIN_NAME_SERVER, OPTION_BROADCAST_ADDRESS,
	OPTION_DOMAIN_NAME
};


dhcp_message::dhcp_message(message_type type)
{
	// ASSERT(this == offsetof(this, opcode));
	memset(this, 0, sizeof(*this));
	options_magic = htonl(OPTION_MAGIC);

	uint8* next = PrepareMessage(type);
	FinishOptions(next);
}


bool
dhcp_message::HasOptions() const
{
	return options_magic == htonl(OPTION_MAGIC);
}


bool
dhcp_message::NextOption(dhcp_option_cookie& cookie,
	message_option& option, const uint8*& data, size_t& size) const
{
	if (!HasOptions())
		return false;

	if (cookie.state == 0) {
		cookie.state++;
		cookie.next = options;
	}

	uint32 bytesLeft = 0;

	switch (cookie.state) {
		case 1:
			// options from "options"
			bytesLeft = sizeof(options) - (cookie.next - options);
			break;

		case 2:
			// options from "file"
			bytesLeft = sizeof(file) - (cookie.next - file);
			break;

		case 3:
			// options from "server_name"
			bytesLeft = sizeof(server_name) - (cookie.next - server_name);
			break;
	}

	while (true) {
		if (bytesLeft == 0) {
			cookie.state++;

			// handle option overload in file and/or server_name fields.
			switch (cookie.state) {
				case 2:
					// options from "file" field
					if (cookie.file_has_options) {
						bytesLeft = sizeof(file);
						cookie.next = file;
					}
					break;

				case 3:
					// options from "server_name" field
					if (cookie.server_name_has_options) {
						bytesLeft = sizeof(server_name);
						cookie.next = server_name;
					}
					break;

				default:
					// no more place to look for options
					// if last option is not OPTION_END,
					// there is no space left for other option!
					if (option != OPTION_END)
						cookie.next = NULL;
					return false;
			}

			if (bytesLeft == 0) {
				// no options in this state, try next one
				continue;
			}
		}

		option = (message_option)cookie.next[0];
		if (option == OPTION_END) {
			bytesLeft = 0;
			continue;
		} else if (option == OPTION_PAD) {
			bytesLeft--;
			cookie.next++;
			continue;
		}

		size = cookie.next[1];
		data = &cookie.next[2];
		cookie.next += 2 + size;
		bytesLeft -= 2 + size;

		if (option == OPTION_OVERLOAD) {
			cookie.file_has_options = data[0] & 1;
			cookie.server_name_has_options = data[0] & 2;
			continue;
		}

		return true;
	}
}


const uint8*
dhcp_message::FindOption(message_option which) const
{
	dhcp_option_cookie cookie;
	message_option option;
	const uint8* data;
	size_t size;
	while (NextOption(cookie, option, data, size)) {
		// iterate through all options
		if (option == which)
			return data;
	}

	return NULL;
}


const uint8*
dhcp_message::LastOption() const
{
	dhcp_option_cookie cookie;
	message_option option;
	const uint8* data;
	size_t size;
	while (NextOption(cookie, option, data, size)) {
		// iterate through all options
	}

	return cookie.next;
}


message_type
dhcp_message::Type() const
{
	const uint8* data = FindOption(OPTION_MESSAGE_TYPE);
	if (data)
		return (message_type)data[0];

	return DHCP_NONE;
}


size_t
dhcp_message::Size() const
{
	const uint8* last = LastOption();

	if (last < options) {
		// if last option is stored above "options" field, it means
		// the whole options field and message is already filled...
		return sizeof(dhcp_message);
	}

	return sizeof(dhcp_message) - sizeof(options) + last + 1 - options;
}


uint8*
dhcp_message::PrepareMessage(uint8 type)
{
	uint8* next = options;
	next = PutOption(next, OPTION_MESSAGE_TYPE, type);
	next = PutOption(next, OPTION_MAX_MESSAGE_SIZE,
		(uint16)htons(sizeof(dhcp_message)));
	return next;
}


uint8*
dhcp_message::PutOption(uint8* options, message_option option)
{
	options[0] = option;
	return options + 1;
}


uint8*
dhcp_message::PutOption(uint8* options, message_option option, uint8 data)
{
	return PutOption(options, option, &data, 1);
}


uint8*
dhcp_message::PutOption(uint8* options, message_option option, uint16 data)
{
	return PutOption(options, option, (uint8*)&data, sizeof(data));
}


uint8*
dhcp_message::PutOption(uint8* options, message_option option, uint32 data)
{
	return PutOption(options, option, (uint8*)&data, sizeof(data));
}


uint8*
dhcp_message::PutOption(uint8* options, message_option option,
	const uint8* data, uint32 size)
{
	// TODO: check enough space is available

	options[0] = option;
	options[1] = size;
	memcpy(&options[2], data, size);

	return options + 2 + size;
}


uint8*
dhcp_message::FinishOptions(uint8* options)
{
	return PutOption(options, OPTION_END);
}


/*static*/ const char*
dhcp_message::TypeToString(message_type type)
{
	switch (type) {
#define CASE(x) case x: return #x;
		CASE(DHCP_NONE)
		CASE(DHCP_DISCOVER)
		CASE(DHCP_OFFER)
		CASE(DHCP_REQUEST)
		CASE(DHCP_DECLINE)
		CASE(DHCP_ACK)
		CASE(DHCP_NACK)
		CASE(DHCP_RELEASE)
		CASE(DHCP_INFORM)
#undef CASE
	}

	return "<unknown>";
}


void
socket_timeout::UpdateSocket(int socket) const
{
	struct timeval value;
	value.tv_sec = timeout / 1000000;
	value.tv_usec = timeout % 1000000;
	setsockopt(socket, SOL_SOCKET, SO_RCVTIMEO, &value, sizeof(value));
}


bool
socket_timeout::Shift(int socket, bigtime_t stateMaxTime, const char* device)
{
	if (tries == UINT8_MAX)
		return false;

	tries++;

	if (tries > MAX_RETRIES) {
		bigtime_t now = system_time();
		if (stateMaxTime == -1 || stateMaxTime < now)
			return false;
		bigtime_t remaining = (stateMaxTime - now) / 2 + 1;
		timeout = std::max(remaining, bigtime_t(AS_USECS(60)));
	} else
		timeout += timeout;

	if (timeout > AS_USECS(MAX_TIMEOUT))
		timeout = AS_USECS(MAX_TIMEOUT);

	syslog(LOG_DEBUG, "%s: Timeout shift: %" B_PRIdTIME " msecs (try %" B_PRIu8 ")\n",
		device, timeout / 1000, tries);

	UpdateSocket(socket);
	return true;
}


//	#pragma mark -


DHCPClient::DHCPClient(BMessenger target, const char* device)
	:
	AutoconfigClient("dhcp", target, device),
	fConfiguration(kMsgConfigureInterface),
	fResolverConfiguration(kMsgConfigureResolver),
	fRunner(NULL),
	fAssignedAddress(0),
	fServer(AF_INET, NULL, DHCP_SERVER_PORT, B_UNCONFIGURED_ADDRESS_FAMILIES),
	fLeaseTime(0)
{
	fTransactionID = (uint32)system_time() ^ rand();

	BNetworkAddress link;
	BNetworkInterface interface(device);
	fStatus = interface.GetHardwareAddress(link);
	if (fStatus != B_OK)
		return;

	memcpy(fMAC, link.LinkLevelAddress(), sizeof(fMAC));

	if ((interface.Flags() & IFF_AUTO_CONFIGURED) != 0) {
		// Check for interface previous auto-configured address, if any.
		BNetworkInterfaceAddress interfaceAddress;
		int index = interface.FindFirstAddress(AF_INET);
		if (index >= 0
			&& interface.GetAddressAt(index, interfaceAddress) == B_OK) {
			BNetworkAddress address = interfaceAddress.Address();
			const sockaddr_in& addr = (sockaddr_in&)address.SockAddr();
			fAssignedAddress = addr.sin_addr.s_addr;

			if ((ntohl(fAssignedAddress) & IN_CLASSB_NET) == 0xa9fe0000) {
				// previous auto-configured address is a link-local one:
				// there is no point asking a DHCP server to renew such
				// server-less assigned address...
				fAssignedAddress = 0;
			}
		}
	}

	openlog_thread("DHCP", 0, LOG_DAEMON);
}


DHCPClient::~DHCPClient()
{
	if (fStatus != B_OK)
		return;

	delete fRunner;

	int socket = ::socket(AF_INET, SOCK_DGRAM, 0);
	if (socket < 0)
		return;

	// release lease

	dhcp_message release(DHCP_RELEASE);
	_PrepareMessage(release, BOUND);

	_SendMessage(socket, release, fServer);
	close(socket);

	closelog_thread();
}


status_t
DHCPClient::Initialize()
{
	fStatus = _Negotiate(fAssignedAddress == 0 ? INIT : INIT_REBOOT);
	syslog(LOG_DEBUG, "%s: DHCP status = %s\n", Device(), strerror(fStatus));
	return fStatus;
}


status_t
DHCPClient::_Negotiate(dhcp_state state)
{
	if (state == BOUND)
		return B_OK;

	fStartTime = system_time();
	fTransactionID++;

	char hostName[MAXHOSTNAMELEN];
	if (gethostname(hostName, MAXHOSTNAMELEN) == 0)
		fHostName.SetTo(hostName, MAXHOSTNAMELEN);
	else
		fHostName.Truncate(0);

	int socket = ::socket(AF_INET, SOCK_DGRAM, 0);
	if (socket < 0)
		return errno;

	// Enable reusing the port. This is needed in case there is more
	// than 1 interface that needs to be configured. Note that the only reason
	// this works is because there is code below to bind to a specific
	// interface.
	int option = 1;
	setsockopt(socket, SOL_SOCKET, SO_REUSEPORT, &option, sizeof(option));

	BNetworkAddress local;
	local.SetToWildcard(AF_INET, DHCP_CLIENT_PORT);

	option = 1;
	setsockopt(socket, SOL_SOCKET, SO_BROADCAST, &option, sizeof(option));

	if (bind(socket, local, local.Length()) < 0) {
		close(socket);
		return errno;
	}

	bigtime_t previousLeaseTime = fLeaseTime;

	status_t status = B_OK;
	while (state != BOUND) {
		status = _StateTransition(socket, state);
		if (status != B_OK && (state == SELECTING || state == REBOOTING))
			break;
	}

	close(socket);

	if (fLeaseTime == 0)
		fLeaseTime = previousLeaseTime;
	if (fLeaseTime == 0)
		fLeaseTime = 60;

	if (fRenewalTime == 0)
		fRenewalTime = fLeaseTime / 2;
	if (fRebindingTime == 0)
		fRebindingTime = fLeaseTime * 7 / 8;
	fLeaseTime += fRequestTime;
	fRenewalTime += fRequestTime;
	fRebindingTime += fRequestTime;
	_RestartLease(fRenewalTime);

	fStatus = status;
	if (status)
		return status;

	// configure interface
	BMessage reply;
	status = Target().SendMessage(&fConfiguration, &reply);
	if (status == B_OK)
		status = reply.FindInt32("status", &fStatus);

	// configure resolver
	reply.MakeEmpty();
	fResolverConfiguration.AddString("device", Device());
	status = Target().SendMessage(&fResolverConfiguration, &reply);
	if (status == B_OK)
		status = reply.FindInt32("status", &fStatus);
	return status;
}


status_t
DHCPClient::_GotMessage(dhcp_state& state, dhcp_message* message)
{
	switch (state) {
		case SELECTING:
			if (message->Type() == DHCP_OFFER) {
				state = REQUESTING;

				fAssignedAddress = message->your_address;
				syslog(LOG_INFO, "  your_address: %s\n",
						_AddressToString(fAssignedAddress).String());

				fConfiguration.MakeEmpty();
				fConfiguration.AddString("device", Device());
				fConfiguration.AddBool("auto_configured", true);

				BMessage address;
				address.AddString("family", "inet");
				address.AddString("address", _AddressToString(fAssignedAddress));
				fResolverConfiguration.MakeEmpty();
				_ParseOptions(*message, address, fResolverConfiguration);

				fConfiguration.AddMessage("address", &address);
				return B_OK;
			}

			return B_BAD_VALUE;

		case REBOOTING:
		case REBINDING:
		case RENEWING:
		case REQUESTING:
			if (message->Type() == DHCP_ACK) {
				// TODO: we might want to configure the stuff, don't we?
				BMessage address;
				fResolverConfiguration.MakeEmpty();
				_ParseOptions(*message, address, fResolverConfiguration);
					// TODO: currently, only lease time and DNS is updated this
					// way

				// our address request has been acknowledged
				state = BOUND;

				return B_OK;
			}

			if (message->Type() == DHCP_NACK) {
				// server reject our request on previous assigned address
				// back to square one...
				fAssignedAddress = 0;
				state = INIT;
				return B_OK;
			}

		default:
			return B_BAD_VALUE;
	}
}


status_t
DHCPClient::_StateTransition(int socket, dhcp_state& state)
{
	if (state == INIT) {
		// The local interface does not have an address yet, bind the socket
		// to the device directly.
		BNetworkDevice device(Device());
		int index = device.Index();

		setsockopt(socket, SOL_SOCKET, SO_BINDTODEVICE, &index, sizeof(int));
	}

	BNetworkAddress broadcast;
	broadcast.SetToBroadcast(AF_INET, DHCP_SERVER_PORT);

	socket_timeout timeout(socket);

	dhcp_message discover(DHCP_DISCOVER);
	_PrepareMessage(discover, state);

	dhcp_message request(DHCP_REQUEST);
	_PrepareMessage(request, state);

	bool skipRequest = false;
	dhcp_state originalState = state;
	fRequestTime = system_time();
	while (true) {
		if (!skipRequest) {
			_SendMessage(socket, originalState == INIT ? discover : request,
				originalState == RENEWING ? fServer : broadcast);

			if (originalState == INIT)
				state = SELECTING;
			else if (originalState == INIT_REBOOT)
				state = REBOOTING;
		}

		char buffer[2048];
		struct sockaddr_in from;
		socklen_t fromLength = sizeof(from);
		ssize_t bytesReceived = recvfrom(socket, buffer, sizeof(buffer),
			0, (struct sockaddr*)&from, &fromLength);
		if (bytesReceived < 0 && errno == B_TIMED_OUT) {
			// depending on the state, we'll just try again
			if (!_TimeoutShift(socket, state, timeout))
				return B_TIMED_OUT;
			skipRequest = false;
			continue;
		} else if (bytesReceived < 0)
			return errno;

		skipRequest = true;
		dhcp_message* message = (dhcp_message*)buffer;
		if (message->transaction_id != htonl(fTransactionID)
			|| !message->HasOptions()
			|| memcmp(message->mac_address, discover.mac_address,
				discover.hardware_address_length)) {
			// this message is not for us
			syslog(LOG_DEBUG, "%s: Ignoring %s not for us from %s\n",
				Device(), dhcp_message::TypeToString(message->Type()),
				_AddressToString(from.sin_addr.s_addr).String());
			continue;
		}

		syslog(LOG_DEBUG, "%s: Received %s from %s\n",
			Device(), dhcp_message::TypeToString(message->Type()),
			_AddressToString(from.sin_addr.s_addr).String());

		if (_GotMessage(state, message) == B_OK)
			break;
	}

	return B_OK;
}


void
DHCPClient::_RestartLease(bigtime_t leaseTime)
{
	if (leaseTime == 0)
		return;

	BMessage lease(kMsgLeaseTime);
	fRunner = new BMessageRunner(this, &lease, leaseTime - system_time(), 1);
}


void
DHCPClient::_ParseOptions(dhcp_message& message, BMessage& address,
	BMessage& resolverConfiguration)
{
	dhcp_option_cookie cookie;
	message_option option;
	const uint8* data;
	size_t size;
	while (message.NextOption(cookie, option, data, size)) {
		// iterate through all options
		switch (option) {
			case OPTION_ROUTER_ADDRESS:
				syslog(LOG_DEBUG, "  gateway: %s\n",
					_AddressToString(data).String());
				address.AddString("gateway", _AddressToString(data));
				break;
			case OPTION_SUBNET_MASK:
				syslog(LOG_DEBUG, "  subnet: %s\n",
					_AddressToString(data).String());
				address.AddString("mask", _AddressToString(data));
				break;
			case OPTION_BROADCAST_ADDRESS:
				syslog(LOG_DEBUG, "  broadcast: %s\n",
					_AddressToString(data).String());
				address.AddString("broadcast", _AddressToString(data));
				break;
			case OPTION_DOMAIN_NAME_SERVER:
			{
				for (uint32 i = 0; i < size / 4; i++) {
					syslog(LOG_DEBUG, "  nameserver[%d]: %s\n", i,
						_AddressToString(&data[i * 4]).String());
					resolverConfiguration.AddString("nameserver",
						_AddressToString(&data[i * 4]).String());
				}
				resolverConfiguration.AddInt32("nameserver_count",
					size / 4);
				break;
			}
			case OPTION_SERVER_ADDRESS:
			{
				syslog(LOG_DEBUG, "  server: %s\n",
					_AddressToString(data).String());
				status_t status = fServer.SetAddress(*(in_addr_t*)data);
				if (status != B_OK) {
					syslog(LOG_ERR, "   BNetworkAddress::SetAddress failed with %s!\n",
						strerror(status));
					fServer.Unset();
				}
				break;
			}

			case OPTION_ADDRESS_LEASE_TIME:
				syslog(LOG_DEBUG, "  lease time: %lu seconds\n",
					ntohl(*(uint32*)data));
				fLeaseTime = ntohl(*(uint32*)data) * 1000000LL;
				break;
			case OPTION_RENEWAL_TIME:
				syslog(LOG_DEBUG, "  renewal time: %lu seconds\n",
					ntohl(*(uint32*)data));
				fRenewalTime = ntohl(*(uint32*)data) * 1000000LL;
				break;
			case OPTION_REBINDING_TIME:
				syslog(LOG_DEBUG, "  rebinding time: %lu seconds\n",
					ntohl(*(uint32*)data));
				fRebindingTime = ntohl(*(uint32*)data) * 1000000LL;
				break;

			case OPTION_HOST_NAME:
				syslog(LOG_DEBUG, "  host name: \"%.*s\"\n", (int)size,
					(const char*)data);
				break;

			case OPTION_DOMAIN_NAME:
			{
				char domain[256];
				strlcpy(domain, (const char*)data,
					min_c(size + 1, sizeof(domain)));

				syslog(LOG_DEBUG, "  domain name: \"%s\"\n", domain);

				resolverConfiguration.AddString("domain", domain);
				break;
			}

			case OPTION_MESSAGE_TYPE:
				break;

			case OPTION_ERROR_MESSAGE:
				syslog(LOG_INFO, "  error message: \"%.*s\"\n", (int)size,
					(const char*)data);
				break;

			default:
				syslog(LOG_DEBUG, "  UNKNOWN OPTION %" B_PRIu32 " (0x%" B_PRIx32 ")\n",
					(uint32)option, (uint32)option);
				break;
		}
	}
}


void
DHCPClient::_PrepareMessage(dhcp_message& message, dhcp_state state)
{
	message.opcode = BOOT_REQUEST;
	message.hardware_type = ARP_HARDWARE_TYPE_ETHER;
	message.hardware_address_length = 6;
	message.transaction_id = htonl(fTransactionID);
	message.seconds_since_start = htons(min_c((system_time() - fStartTime)
		/ 1000000LL, 65535));
	memcpy(message.mac_address, fMAC, 6);

	message_type type = message.Type();

	uint8 *next = message.PrepareMessage(type);

	switch (type) {
		case DHCP_DISCOVER:
			next = message.PutOption(next, OPTION_REQUEST_PARAMETERS,
				kRequestParameters, sizeof(kRequestParameters));

			if (fHostName.Length() > 0) {
				next = message.PutOption(next, OPTION_HOST_NAME,
					reinterpret_cast<const uint8*>(fHostName.String()),
					fHostName.Length());
			}
			break;

		case DHCP_REQUEST:
			next = message.PutOption(next, OPTION_REQUEST_PARAMETERS,
				kRequestParameters, sizeof(kRequestParameters));

			if (fHostName.Length() > 0) {
				next = message.PutOption(next, OPTION_HOST_NAME,
					reinterpret_cast<const uint8*>(fHostName.String()),
					fHostName.Length());
			}

			if (state == REQUESTING) {
				const sockaddr_in& server = (sockaddr_in&)fServer.SockAddr();
				next = message.PutOption(next, OPTION_SERVER_ADDRESS,
					(uint32)server.sin_addr.s_addr);
			}

			if (state == INIT || state == INIT_REBOOT
				|| state == REQUESTING) {
				next = message.PutOption(next, OPTION_REQUEST_IP_ADDRESS,
					(uint32)fAssignedAddress);
			} else
				message.client_address = fAssignedAddress;
			break;

		case DHCP_RELEASE: {
			const sockaddr_in& server = (sockaddr_in&)fServer.SockAddr();
			next = message.PutOption(next, OPTION_SERVER_ADDRESS,
				(uint32)server.sin_addr.s_addr);

			message.client_address = fAssignedAddress;
			break;
		}

		default:
			break;
	}

	message.FinishOptions(next);
}


bool
DHCPClient::_TimeoutShift(int socket, dhcp_state& state,
	socket_timeout& timeout)
{
	bigtime_t stateMaxTime = -1;

	// Compute the date at which we must consider the DHCP negociation failed.
	// This varies depending on the current state. In renewing and rebinding
	// states, it is based on the lease expiration.
	// We can stay for up to 1 minute in the selecting and requesting states
	// (these correspond to sending DHCP_DISCOVER and DHCP_REQUEST,
	// respectively).
	// All other states time out immediately after a single try.
	// If this timeout expires, the DHCP negociation is aborted and starts
	// over. As long as the timeout is not expired, we repeat the message with
	// increasing delays (the delay is computed in timeout.shift below, and is
	// at most equal to the timeout, but usually much shorter).
	if (state == RENEWING)
		stateMaxTime = fRebindingTime;
	else if (state == REBINDING)
		stateMaxTime = fLeaseTime;
	else if (state == SELECTING || state == REQUESTING)
		stateMaxTime = fRequestTime + AS_USECS(MAX_TIMEOUT);

	if (system_time() > stateMaxTime) {
		state = state == REBINDING ? INIT : REBINDING;
		return false;
	}

	return timeout.Shift(socket, stateMaxTime, Device());
}


/*static*/ BString
DHCPClient::_AddressToString(const uint8* data)
{
	BString target = inet_ntoa(*(in_addr*)data);
	return target;
}


/*static*/ BString
DHCPClient::_AddressToString(in_addr_t address)
{
	BString target = inet_ntoa(*(in_addr*)&address);
	return target;
}


status_t
DHCPClient::_SendMessage(int socket, dhcp_message& message,
	const BNetworkAddress& address) const
{
	message_type type = message.Type();
	BString text;
	text << dhcp_message::TypeToString(type);

	const uint8* requestAddress = message.FindOption(OPTION_REQUEST_IP_ADDRESS);
	if (type == DHCP_REQUEST && requestAddress != NULL)
		text << " for " << _AddressToString(requestAddress).String();

	syslog(LOG_DEBUG, "%s: Send %s to %s\n", Device(), text.String(),
		address.ToString().String());

	ssize_t bytesSent = sendto(socket, &message, message.Size(),
		address.IsBroadcast() ? MSG_BCAST : 0, address, address.Length());
	if (bytesSent < 0)
		return errno;

	return B_OK;
}


dhcp_state
DHCPClient::_CurrentState() const
{
	bigtime_t now = system_time();

	if (now > fLeaseTime || fStatus != B_OK)
		return INIT;
	if (now >= fRebindingTime)
		return REBINDING;
	if (now >= fRenewalTime)
		return RENEWING;
	return BOUND;
}


void
DHCPClient::MessageReceived(BMessage* message)
{
	switch (message->what) {
		case kMsgLeaseTime:
			_Negotiate(_CurrentState());
			break;

		default:
			BHandler::MessageReceived(message);
			break;
	}
}
