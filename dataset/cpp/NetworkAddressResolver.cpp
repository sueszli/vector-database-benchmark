/*
 * Copyright 2010-2011, Axel Dörfler, axeld@pinc-software.de.
 * Copyright 2015-2017, Adrien Destugues, pulkomandy@pulkomandy.tk.
 * Distributed under the terms of the MIT License.
 */


#include <NetworkAddressResolver.h>

#include <errno.h>
#include <netdb.h>

#include <Autolock.h>
#include <NetworkAddress.h>


static bool
strip_port(BString& host, BString& port)
{
	int32 first = host.FindFirst(':');
	int32 separator = host.FindLast(':');
	if (separator != first
			&& (separator == 0 || host.ByteAt(separator - 1) != ']')) {
		return false;
	}

	if (separator != -1) {
		// looks like there is a port
		host.CopyInto(port, separator + 1, -1);
		host.Truncate(separator);

		return true;
	}

	return false;
}


// #pragma mark -


BNetworkAddressResolver::BNetworkAddressResolver()
	:
	BReferenceable(),
	fInfo(NULL),
	fStatus(B_NO_INIT)
{
}


BNetworkAddressResolver::BNetworkAddressResolver(const char* address,
	uint16 port, uint32 flags)
	:
	BReferenceable(),
	fInfo(NULL),
	fStatus(B_NO_INIT)
{
	SetTo(address, port, flags);
}

BNetworkAddressResolver::BNetworkAddressResolver(const char* address,
	const char* service, uint32 flags)
	:
	BReferenceable(),
	fInfo(NULL),
	fStatus(B_NO_INIT)
{
	SetTo(address, service, flags);
}


BNetworkAddressResolver::BNetworkAddressResolver(int family,
	const char* address, uint16 port, uint32 flags)
	:
	BReferenceable(),
	fInfo(NULL),
	fStatus(B_NO_INIT)
{
	SetTo(family, address, port, flags);
}


BNetworkAddressResolver::BNetworkAddressResolver(int family,
	const char* address, const char* service, uint32 flags)
	:
	BReferenceable(),
	fInfo(NULL),
	fStatus(B_NO_INIT)
{
	SetTo(family, address, service, flags);
}


BNetworkAddressResolver::~BNetworkAddressResolver()
{
	Unset();
}


status_t
BNetworkAddressResolver::InitCheck() const
{
	return fStatus;
}


void
BNetworkAddressResolver::Unset()
{
	if (fInfo != NULL) {
		freeaddrinfo(fInfo);
		fInfo = NULL;
	}
	fStatus = B_NO_INIT;
}


status_t
BNetworkAddressResolver::SetTo(const char* address, uint16 port, uint32 flags)
{
	return SetTo(AF_UNSPEC, address, port, flags);
}


status_t
BNetworkAddressResolver::SetTo(const char* address, const char* service,
	uint32 flags)
{
	return SetTo(AF_UNSPEC, address, service, flags);
}


status_t
BNetworkAddressResolver::SetTo(int family, const char* address, uint16 port,
	uint32 flags)
{
	BString service;
	service << port;

	return SetTo(family, address, port != 0 ? service.String() : NULL, flags);
}


status_t
BNetworkAddressResolver::SetTo(int family, const char* host,
	const char* service, uint32 flags)
{
	Unset();

	// Check if the address contains a port

	BString hostString(host);

	BString portString;
	if (!strip_port(hostString, portString) && service != NULL)
		portString = service;

	// Resolve address

	addrinfo hint = {0};
	hint.ai_family = family;
	if ((flags & B_NO_ADDRESS_RESOLUTION) != 0)
		hint.ai_flags |= AI_NUMERICHOST;
	else if ((flags & B_UNCONFIGURED_ADDRESS_FAMILIES) == 0)
		hint.ai_flags |= AI_ADDRCONFIG;

	if (host == NULL && portString.Length() == 0) {
		portString = "0";
		hint.ai_flags |= AI_PASSIVE;
	}

	int status = getaddrinfo(host != NULL ? hostString.String() : NULL,
		portString.Length() != 0 ? portString.String() : NULL, &hint, &fInfo);
	if (status == 0)
		return fStatus = B_OK;

	// Map errors
	// TODO: improve error reporting, maybe add specific error codes?

	switch (status) {
		case EAI_ADDRFAMILY:
		case EAI_BADFLAGS:
		case EAI_PROTOCOL:
		case EAI_BADHINTS:
		case EAI_SOCKTYPE:
		case EAI_SERVICE:
		case EAI_NONAME:
		case EAI_FAMILY:
			fStatus = B_BAD_VALUE;
			break;

		case EAI_SYSTEM:
			fStatus = errno;
			break;

		case EAI_OVERFLOW:
		case EAI_MEMORY:
			fStatus = B_NO_MEMORY;
			break;

		case EAI_AGAIN:
			// TODO: better error code to denote temporary failure?
			fStatus = B_TIMED_OUT;
			break;

		default:
			fStatus = B_ERROR;
			break;
	}

	return fStatus;
}


status_t
BNetworkAddressResolver::GetNextAddress(uint32* cookie,
	BNetworkAddress& address) const
{
	if (fStatus != B_OK)
		return fStatus;

	// Skip previous info entries

	addrinfo* info = fInfo;
	int32 first = *cookie;
	for (int32 index = 0; index < first && info != NULL; index++) {
		info = info->ai_next;
	}

	if (info == NULL)
		return B_BAD_VALUE;

	// Return current

	address.SetTo(*info->ai_addr, info->ai_addrlen);
	(*cookie)++;

	return B_OK;
}


status_t
BNetworkAddressResolver::GetNextAddress(int family, uint32* cookie,
	BNetworkAddress& address) const
{
	if (fStatus != B_OK)
		return fStatus;

	// Skip previous info entries, and those that have a non-matching family

	addrinfo* info = fInfo;
	int32 first = *cookie;
	for (int32 index = 0; index < first && info != NULL; index++)
		info = info->ai_next;

	while (info != NULL && info->ai_family != family)
		info = info->ai_next;

	if (info == NULL)
		return B_BAD_VALUE;

	// Return current

	address.SetTo(*info->ai_addr, info->ai_addrlen);
	(*cookie)++;

	return B_OK;
}


/*static*/ BReference<const BNetworkAddressResolver>
BNetworkAddressResolver::Resolve(const char* address, const char* service,
	uint32 flags)
{
	return Resolve(AF_UNSPEC, address, service, flags);
}


/*static*/ BReference<const BNetworkAddressResolver>
BNetworkAddressResolver::Resolve(const char* address, uint16 port, uint32 flags)
{
	return Resolve(AF_UNSPEC, address, port, flags);
}


/*static*/ BReference<const BNetworkAddressResolver>
BNetworkAddressResolver::Resolve(int family, const char* address,
	uint16 port, uint32 flags)
{
	BString service;
	service << port;

	return Resolve(family, address, port == 0 ? NULL : service.String(), flags);
}


/*static*/ BReference<const BNetworkAddressResolver>
BNetworkAddressResolver::Resolve(int family, const char* address,
	const char* service, uint32 flags)
{
	BAutolock locker(&sCacheLock);

	// TODO it may be faster to use an hash map to have faster lookup of the
	// cache. However, we also need to access the cache by LRU, and for that
	// a doubly-linked list is better. We should have these two share the same
	// items, so it's easy to remove the LRU from the map, or insert a new
	// item in both structures.
	for (int i = 0; i < sCacheMap.CountItems(); i++) {
		CacheEntry* entry = sCacheMap.ItemAt(i);
		if (entry->Matches(family, address, service, flags)) {
			// This entry is now the MRU, move to end of list.
			// TODO if the item is old (more than 1 minute), it should be
			// dropped and a new request made.
			sCacheMap.MoveItem(i, sCacheMap.CountItems());
			return entry->fResolver;
		}
	}

	// Cache miss! Unlock the cache while we perform the costly address
	// resolution
	locker.Unlock();

	BNetworkAddressResolver* resolver = new(std::nothrow)
		BNetworkAddressResolver(family, address, service, flags);

	if (resolver != NULL && resolver->InitCheck() == B_OK) {
		CacheEntry* entry = new(std::nothrow) CacheEntry(family, address,
			service, flags, resolver);

		locker.Lock();
		// TODO adjust capacity. Chrome uses 256 entries with a timeout of
		// 1 minute, IE uses 1000 entries with a timeout of 30 seconds.
		if (sCacheMap.CountItems() > 255)
			delete sCacheMap.RemoveItemAt(0);

		if (entry)
			sCacheMap.AddItem(entry, sCacheMap.CountItems());
	}

	return BReference<const BNetworkAddressResolver>(resolver, true);
}

BLocker BNetworkAddressResolver::sCacheLock("DNS cache");
BObjectList<BNetworkAddressResolver::CacheEntry>
	BNetworkAddressResolver::sCacheMap;
