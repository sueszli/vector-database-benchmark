/*  PCSX2 - PS2 Emulator for PCs
 *  Copyright (C) 2002-2022  PCSX2 Dev Team
 *
 *  PCSX2 is free software: you can redistribute it and/or modify it under the terms
 *  of the GNU Lesser General Public License as published by the Free Software Found-
 *  ation, either version 3 of the License, or (at your option) any later version.
 *
 *  PCSX2 is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
 *  without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 *  PURPOSE.  See the GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License along with PCSX2.
 *  If not, see <http://www.gnu.org/licenses/>.
 */

#include "PrecompiledHeader.h"

#ifdef __POSIX__
#include <vector>
#include <fstream>
#include <net/if.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

#include "common/StringUtil.h"

#ifdef __linux__
#include <unistd.h>
#include <sys/ioctl.h>
#endif

#if defined(__FreeBSD__) || (__APPLE__)
#include <sys/param.h>
#include <sys/sysctl.h>
#include <net/route.h>

#include "common/Assertions.h"

#endif
#endif

#include "AdapterUtils.h"

using namespace PacketReader;
using namespace PacketReader::IP;

#ifdef _WIN32
bool AdapterUtils::GetAdapter(const std::string& name, Adapter* adapter, AdapterBuffer* buffer)
{
	int neededSize = 128;
	std::unique_ptr<IP_ADAPTER_ADDRESSES[]> adapterInfo = std::make_unique<IP_ADAPTER_ADDRESSES[]>(neededSize);
	ULONG dwBufLen = sizeof(IP_ADAPTER_ADDRESSES) * neededSize;

	PIP_ADAPTER_ADDRESSES pAdapterInfo;

	DWORD dwStatus = GetAdaptersAddresses(
		AF_UNSPEC,
		GAA_FLAG_INCLUDE_PREFIX | GAA_FLAG_INCLUDE_GATEWAYS,
		NULL,
		adapterInfo.get(),
		&dwBufLen);

	if (dwStatus == ERROR_BUFFER_OVERFLOW)
	{
		DevCon.WriteLn("DEV9: GetWin32Adapter() buffer too small, resizing");
		neededSize = dwBufLen / sizeof(IP_ADAPTER_ADDRESSES) + 1;
		adapterInfo = std::make_unique<IP_ADAPTER_ADDRESSES[]>(neededSize);
		dwBufLen = sizeof(IP_ADAPTER_ADDRESSES) * neededSize;
		DevCon.WriteLn("DEV9: New size %i", neededSize);

		dwStatus = GetAdaptersAddresses(
			AF_UNSPEC,
			GAA_FLAG_INCLUDE_PREFIX | GAA_FLAG_INCLUDE_GATEWAYS,
			NULL,
			adapterInfo.get(),
			&dwBufLen);
	}
	if (dwStatus != ERROR_SUCCESS)
		return false;

	pAdapterInfo = adapterInfo.get();

	do
	{
		if (strcmp(pAdapterInfo->AdapterName, name.c_str()) == 0)
		{
			*adapter = *pAdapterInfo;
			buffer->swap(adapterInfo);
			return true;
		}

		pAdapterInfo = pAdapterInfo->Next;
	} while (pAdapterInfo);

	return false;
}
bool AdapterUtils::GetAdapterAuto(Adapter* adapter, AdapterBuffer* buffer)
{
	int neededSize = 128;
	std::unique_ptr<IP_ADAPTER_ADDRESSES[]> adapterInfo = std::make_unique<IP_ADAPTER_ADDRESSES[]>(neededSize);
	ULONG dwBufLen = sizeof(IP_ADAPTER_ADDRESSES) * neededSize;

	PIP_ADAPTER_ADDRESSES pAdapter;

	DWORD dwStatus = GetAdaptersAddresses(
		AF_UNSPEC,
		GAA_FLAG_INCLUDE_PREFIX | GAA_FLAG_INCLUDE_GATEWAYS,
		NULL,
		adapterInfo.get(),
		&dwBufLen);

	if (dwStatus == ERROR_BUFFER_OVERFLOW)
	{
		DevCon.WriteLn("DEV9: PCAPGetWin32Adapter() buffer too small, resizing");
		//
		neededSize = dwBufLen / sizeof(IP_ADAPTER_ADDRESSES) + 1;
		adapterInfo = std::make_unique<IP_ADAPTER_ADDRESSES[]>(neededSize);
		dwBufLen = sizeof(IP_ADAPTER_ADDRESSES) * neededSize;
		DevCon.WriteLn("DEV9: New size %i", neededSize);

		dwStatus = GetAdaptersAddresses(
			AF_UNSPEC,
			GAA_FLAG_INCLUDE_PREFIX | GAA_FLAG_INCLUDE_GATEWAYS,
			NULL,
			adapterInfo.get(),
			&dwBufLen);
	}

	if (dwStatus != ERROR_SUCCESS)
		return 0;

	pAdapter = adapterInfo.get();

	do
	{
		if (pAdapter->IfType != IF_TYPE_SOFTWARE_LOOPBACK &&
			pAdapter->OperStatus == IfOperStatusUp)
		{
			// Search for an adapter with;
			// IPv4 Address,
			// DNS,
			// Gateway.

			bool hasIPv4 = false;
			bool hasDNS = false;
			bool hasGateway = false;

			// IPv4.
			if (GetAdapterIP(pAdapter).has_value())
				hasIPv4 = true;

			// DNS.
			if (GetDNS(pAdapter).size() > 0)
				hasDNS = true;

			// Gateway.
			if (GetGateways(pAdapter).size() > 0)
				hasGateway = true;

			if (hasIPv4 && hasDNS && hasGateway)
			{
				*adapter = *pAdapter;
				buffer->swap(adapterInfo);
				return true;
			}
		}

		pAdapter = pAdapter->Next;
	} while (pAdapter);

	return false;
}
#elif defined(__POSIX__)
bool AdapterUtils::GetAdapter(const std::string& name, Adapter* adapter, AdapterBuffer* buffer)
{
	ifaddrs* ifa;
	ifaddrs* pAdapter;

	int error = getifaddrs(&ifa);
	if (error)
		return false;

	std::unique_ptr<ifaddrs, IfAdaptersDeleter> adapterInfo(ifa, IfAdaptersDeleter());

	pAdapter = adapterInfo.get();

	do
	{
		if (pAdapter->ifa_addr != nullptr &&
			pAdapter->ifa_addr->sa_family == AF_INET &&
			strcmp(pAdapter->ifa_name, name.c_str()) == 0)
			break;

		pAdapter = pAdapter->ifa_next;
	} while (pAdapter);

	if (pAdapter != nullptr)
	{
		*adapter = *pAdapter;
		buffer->swap(adapterInfo);
		return true;
	}

	return false;
}
bool AdapterUtils::GetAdapterAuto(Adapter* adapter, AdapterBuffer* buffer)
{
	ifaddrs* ifa;
	ifaddrs* pAdapter;

	int error = getifaddrs(&ifa);
	if (error)
		return false;

	std::unique_ptr<ifaddrs, IfAdaptersDeleter> adapterInfo(ifa, IfAdaptersDeleter());

	pAdapter = adapterInfo.get();

	do
	{
		if ((pAdapter->ifa_flags & IFF_LOOPBACK) == 0 &&
			(pAdapter->ifa_flags & IFF_UP) != 0)
		{
			// Search for an adapter with;
			// IPv4 Address,
			// Gateway.

			bool hasIPv4 = false;
			bool hasGateway = false;

			if (GetAdapterIP(pAdapter).has_value())
				hasIPv4 = true;

			if (GetGateways(pAdapter).size() > 0)
				hasGateway = true;

			if (hasIPv4 && hasGateway)
			{
				*adapter = *pAdapter;
				buffer->swap(adapterInfo);
				return true;
			}
		}

		pAdapter = pAdapter->ifa_next;
	} while (pAdapter);

	return false;
}
#endif

// AdapterMAC.
#ifdef _WIN32
std::optional<MAC_Address> AdapterUtils::GetAdapterMAC(Adapter* adapter)
{
	if (adapter != nullptr && adapter->PhysicalAddressLength == 6)
		return *(MAC_Address*)adapter->PhysicalAddress;

	return std::nullopt;
}
#elif defined(__POSIX__)
#ifdef __linux__
std::optional<MAC_Address> AdapterUtils::GetAdapterMAC(Adapter* adapter)
{
	struct ifreq ifr;
	strcpy(ifr.ifr_name, adapter->ifa_name);

	int fd = socket(AF_INET, SOCK_DGRAM, 0);
	int ret = ioctl(fd, SIOCGIFHWADDR, &ifr);
	close(fd);

	if (ret == 0)
		return *(MAC_Address*)ifr.ifr_hwaddr.sa_data;

	return std::nullopt;
}
#else
std::optional<MAC_Address> AdapterUtils::GetAdapterMAC(Adapter* adapter)
{
	Console.Error("DEV9: Unsupported OS, can't get MAC address");
	return std::nullopt;
}
#endif
#endif

// AdapterIP.
#ifdef _WIN32
std::optional<IP_Address> AdapterUtils::GetAdapterIP(Adapter* adapter)
{
	PIP_ADAPTER_UNICAST_ADDRESS address = nullptr;
	if (adapter != nullptr)
	{
		address = adapter->FirstUnicastAddress;
		while (address != nullptr && address->Address.lpSockaddr->sa_family != AF_INET)
			address = address->Next;
	}

	if (address != nullptr)
	{
		sockaddr_in* sockaddr = (sockaddr_in*)address->Address.lpSockaddr;
		return *(IP_Address*)&sockaddr->sin_addr;
	}
	return std::nullopt;
}
#elif defined(__POSIX__)
std::optional<IP_Address> AdapterUtils::GetAdapterIP(Adapter* adapter)
{
	sockaddr* address = nullptr;
	if (adapter != nullptr)
	{
		if (adapter->ifa_addr != nullptr && adapter->ifa_addr->sa_family == AF_INET)
			address = adapter->ifa_addr;
	}

	if (address != nullptr)
	{
		sockaddr_in* sockaddr = (sockaddr_in*)address;
		return *(IP_Address*)&sockaddr->sin_addr;
	}
	return std::nullopt;
}
#endif

// Gateways.
#ifdef _WIN32
std::vector<IP_Address> AdapterUtils::GetGateways(Adapter* adapter)
{
	if (adapter == nullptr)
		return {};

	std::vector<IP_Address> collection;

	PIP_ADAPTER_GATEWAY_ADDRESS address = adapter->FirstGatewayAddress;
	while (address != nullptr)
	{
		if (address->Address.lpSockaddr->sa_family == AF_INET)
		{
			sockaddr_in* sockaddr = (sockaddr_in*)address->Address.lpSockaddr;
			collection.push_back(*(IP_Address*)&sockaddr->sin_addr);
		}
		address = address->Next;
	}

	return collection;
}
#elif defined(__POSIX__)
#ifdef __linux__
std::vector<IP_Address> AdapterUtils::GetGateways(Adapter* adapter)
{
	// /proc/net/route contains some information about gateway addresses,
	// and separates the information about by each interface.
	if (adapter == nullptr)
		return {};

	std::vector<IP_Address> collection;
	std::vector<std::string> routeLines;
	std::fstream route("/proc/net/route", std::ios::in);
	if (route.fail())
	{
		route.close();
		Console.Error("DEV9: Failed to open /proc/net/route");
		return collection;
	}

	std::string line;
	while (std::getline(route, line))
		routeLines.push_back(line);
	route.close();

	// Columns are as follows (first-line header):
	// Iface  Destination  Gateway  Flags  RefCnt  Use  Metric  Mask  MTU  Window  IRTT.
	for (size_t i = 1; i < routeLines.size(); i++)
	{
		std::string line = routeLines[i];
		if (line.rfind(adapter->ifa_name, 0) == 0)
		{
			std::vector<std::string_view> split = StringUtil::SplitString(line, '\t', true);
			std::string gatewayIPHex{split[2]};
			// stoi assumes hex values are unsigned, but tries to store it in a signed int,
			// this results in a std::out_of_range exception for addresses ending in a number > 128.
			// We don't have a stoui for (unsigned int), so instead use stoul for (unsigned long).
			u32 addressValue = static_cast<u32>(std::stoul(gatewayIPHex, 0, 16));
			// Skip device routes without valid NextHop IP address.
			if (addressValue != 0)
			{
				IP_Address gwIP = *(IP_Address*)&addressValue;
				collection.push_back(gwIP);
			}
		}
	}
	return collection;
}
#elif defined(__FreeBSD__) || (__APPLE__)
std::vector<IP_Address> AdapterUtils::GetGateways(Adapter* adapter)
{
	if (adapter == nullptr)
		return {};

	std::vector<IP_Address> collection;

	// Get index for our adapter by matching the adapter name.
	int ifIndex = -1;

	struct if_nameindex* ifNI;
	ifNI = if_nameindex();
	if (ifNI == nullptr)
	{
		Console.Error("DEV9: if_nameindex Failed");
		return collection;
	}

	struct if_nameindex* i = ifNI;
	while (i->if_index != 0 && i->if_name != nullptr)
	{
		if (strcmp(i->if_name, adapter->ifa_name) == 0)
		{
			ifIndex = i->if_index;
			break;
		}
		i++;
	}
	if_freenameindex(ifNI);

	// Check if we found the adapter.
	if (ifIndex == -1)
	{
		Console.Error("DEV9: Failed to get index for adapter");
		return collection;
	}

	// Find the gateway by looking though the routing information.
	int name[] = {CTL_NET, PF_ROUTE, 0, AF_INET, NET_RT_DUMP, 0};
	size_t bufferLen = 0;

	if (sysctl(name, 6, NULL, &bufferLen, NULL, 0) != 0)
	{
		Console.Error("DEV9: Failed to perform NET_RT_DUMP");
		return collection;
	}

	// bufferLen is an estimate, double it to be safe.
	bufferLen *= 2;
	std::unique_ptr<u8[]> buffer = std::make_unique<u8[]>(bufferLen);

	if (sysctl(name, 6, buffer.get(), &bufferLen, NULL, 0) != 0)
	{
		Console.Error("DEV9: Failed to perform NET_RT_DUMP");
		return collection;
	}

	rt_msghdr* hdr;
	for (size_t i = 0; i < bufferLen; i += hdr->rtm_msglen)
	{
		hdr = (rt_msghdr*)&buffer[i];

		if (hdr->rtm_flags & RTF_GATEWAY && hdr->rtm_addrs & RTA_GATEWAY && (hdr->rtm_index == ifIndex))
		{
			sockaddr* sockaddrs = (sockaddr*)(hdr + 1);
			pxAssert(sockaddrs[RTAX_DST].sa_family == AF_INET);

			// Default gateway has no destination address.
			sockaddr_in* sockaddr = (sockaddr_in*)&sockaddrs[RTAX_DST];
			if (sockaddr->sin_addr.s_addr != 0)
				continue;

			sockaddr = (sockaddr_in*)&sockaddrs[RTAX_GATEWAY];
			IP_Address gwIP = *(IP_Address*)&sockaddr->sin_addr;
			collection.push_back(gwIP);
		}
	}
	return collection;
}
#else
std::vector<IP_Address> AdapterUtils::GetGateways(Adapter* adapter)
{
	Console.Error("DEV9: Unsupported OS, can't find Gateway");
	return {};
}
#endif
#endif

// DNS.
#ifdef _WIN32
std::vector<IP_Address> AdapterUtils::GetDNS(Adapter* adapter)
{
	if (adapter == nullptr)
		return {};

	std::vector<IP_Address> collection;

	PIP_ADAPTER_DNS_SERVER_ADDRESS address = adapter->FirstDnsServerAddress;
	while (address != nullptr)
	{
		if (address->Address.lpSockaddr->sa_family == AF_INET)
		{
			sockaddr_in* sockaddr = (sockaddr_in*)address->Address.lpSockaddr;
			collection.push_back(*(IP_Address*)&sockaddr->sin_addr);
		}
		address = address->Next;
	}

	return collection;
}
#elif defined(__POSIX__)
std::vector<IP_Address> AdapterUtils::GetDNS(Adapter* adapter)
{
	// On Linux and OSX, DNS is system wide, not adapter specific, so we can ignore the adapter parameter.

	// Parse /etc/resolv.conf for all of the "nameserver" entries.
	// These are the DNS servers the machine is configured to use.
	// On OSX, this file is not directly used by most processes for DNS
	// queries/routing, but it is automatically generated instead, with
	// the machine's DNS servers listed in it.
	if (adapter == nullptr)
		return {};

	std::vector<IP_Address> collection;

	std::fstream servers("/etc/resolv.conf", std::ios::in);
	if (servers.fail())
	{
		servers.close();
		Console.Error("DEV9: Failed to open /etc/resolv.conf");
		return collection;
	}

	std::string line;
	std::vector<std::string> serversLines;
	while (std::getline(servers, line))
		serversLines.push_back(line);
	servers.close();

	const IP_Address systemdDNS{127, 0, 0, 53};
	for (size_t i = 1; i < serversLines.size(); i++)
	{
		std::string line = serversLines[i];
		if (line.rfind("nameserver", 0) == 0)
		{
			std::vector<std::string_view> split = StringUtil::SplitString(line, '\t', true);
			if (split.size() == 1)
				split = StringUtil::SplitString(line, ' ', true);
			std::string dns{split[1]};

			IP_Address address;
			if (inet_pton(AF_INET, dns.c_str(), &address) != 1)
				continue;

			if (address == systemdDNS)
				Console.Error("DEV9: systemd-resolved DNS server is not supported");

			collection.push_back(address);
		}
	}
	return collection;
}
#endif
