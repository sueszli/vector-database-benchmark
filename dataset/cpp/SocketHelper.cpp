/*
* Copyright: JessMA Open Source (ldcsaa@gmail.com)
*
* Author	: Bruce Liang
* Website	: https://github.com/ldcsaa
* Project	: https://github.com/ldcsaa/HP-Socket
* Blog		: http://www.cnblogs.com/ldcsaa
* Wiki		: http://www.oschina.net/p/hp-socket
* QQ Group	: 44636872, 75375912
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*      http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/

#include "SocketHelper.h"

#include <sys/ioctl.h>
#include <arpa/inet.h>
#include <netinet/tcp.h>
#include <netinet/udp.h>

#ifdef _ICONV_SUPPORT
#include <iconv.h>
#endif

#ifndef SO_REUSEPORT
	#define SO_REUSEPORT	15
#endif

///////////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////////////////

static const BYTE s_szUdpCloseNotify[] = {0xBE, 0xB6, 0x1F, 0xEB, 0xDA, 0x52, 0x46, 0xBA, 0x92, 0x33, 0x59, 0xDB, 0xBF, 0xE6, 0xC8, 0xE4};
static const int s_iUdpCloseNotifySize = ARRAY_SIZE(s_szUdpCloseNotify);

const hp_addr hp_addr::ANY_ADDR4(AF_INET, TRUE);
const hp_addr hp_addr::ANY_ADDR6(AF_INET6, TRUE);

LPCTSTR GetSocketErrorDesc(EnSocketError enCode)
{
	switch(enCode)
	{
	case SE_OK:						return _T("SUCCESS");
	case SE_ILLEGAL_STATE:			return _T("Illegal State");
	case SE_INVALID_PARAM:			return _T("Invalid Parameter");
	case SE_SOCKET_CREATE:			return _T("Create SOCKET Fail");
	case SE_SOCKET_BIND:			return _T("Bind SOCKET Fail");
	case SE_SOCKET_PREPARE:			return _T("Prepare SOCKET Fail");
	case SE_SOCKET_LISTEN:			return _T("Listen SOCKET Fail");
	case SE_CP_CREATE:				return _T("Create IOCP Fail");
	case SE_WORKER_THREAD_CREATE:	return _T("Create Worker Thread Fail");
	case SE_DETECT_THREAD_CREATE:	return _T("Create Detector Thread Fail");
	case SE_SOCKE_ATTACH_TO_CP:		return _T("Attach SOCKET to IOCP Fail");
	case SE_CONNECT_SERVER:			return _T("Connect to Server Fail");
	case SE_NETWORK:				return _T("Network Error");
	case SE_DATA_PROC:				return _T("Process Data Error");
	case SE_DATA_SEND:				return _T("Send Data Fail");

	case SE_SSL_ENV_NOT_READY:		return _T("SSL environment not ready");

	default: ASSERT(FALSE);			return _T("UNKNOWN ERROR");
	}
}

ADDRESS_FAMILY DetermineAddrFamily(LPCTSTR lpszAddress)
{
	if (!lpszAddress || lpszAddress[0] == 0)
		return AF_UNSPEC;

	if(::StrChr(lpszAddress, IPV6_ADDR_SEPARATOR_CHAR))
		return AF_INET6;

	TCHAR c;
	int arr[4];

	if(stscanf(lpszAddress, _T("%d.%d.%d.%d%c"), &arr[0], &arr[1], &arr[2], &arr[3], &c) != 4)
		return AF_UNSPEC;

	for(int i = 0; i < 4; i++)
	{
		if(arr[i] < 0 || arr[i] > 255)
			return AF_UNSPEC;
	}

	return AF_INET;
}

BOOL GetInAddr(LPCTSTR lpszAddress, HP_ADDR& addr)
{
	addr.family = DetermineAddrFamily(lpszAddress);

	if (addr.family == AF_UNSPEC)
		return FALSE;

	return (::InetPton(addr.family, lpszAddress, addr.Addr()) == TRUE);
}

BOOL GetSockAddr(LPCTSTR lpszAddress, USHORT usPort, HP_SOCKADDR& addr)
{
	if(addr.family != AF_INET && addr.family != AF_INET6)
	{
		::WSASetLastError(ERROR_ADDRNOTAVAIL);
		return FALSE;
	}

	if(addr.family == AF_INET6 && StrChr(lpszAddress, IPV6_ZONE_INDEX_CHAR))
		return GetSockAddrByHostNameDirectly(lpszAddress, usPort, addr);

	addr.ZeroAddr();

	int rs = ::InetPton(addr.family, lpszAddress, addr.SinAddr());

	if(rs != 1)
	{
		if(rs == 0) ::WSASetLastError(ERROR_INVALID_PARAMETER);

		return FALSE;
	}

	if(usPort != 0)
		addr.SetPort(usPort);

	return TRUE;
}

BOOL IsIPAddress(LPCTSTR lpszAddress, EnIPAddrType* penType)
{
	HP_ADDR addr;

	BOOL isOK = GetInAddr(lpszAddress, addr);

	if(isOK && penType)
		*penType = addr.IsIPv4() ? IPT_IPV4 : IPT_IPV6;

	return isOK;
}

BOOL GetIPAddress(LPCTSTR lpszHost, LPTSTR lpszIP, int& iIPLen, EnIPAddrType& enType)
{
	HP_SOCKADDR addr;

	if(!GetSockAddrByHostName(lpszHost, 0, addr))
		return FALSE;

	enType = addr.IsIPv4() ? IPT_IPV4 : IPT_IPV6;

	USHORT usPort;
	ADDRESS_FAMILY usFamily;
	return sockaddr_IN_2_A(addr, usFamily, lpszIP, iIPLen, usPort);
}

BOOL GetSockAddrByHostName(LPCTSTR lpszHost, USHORT usPort, HP_SOCKADDR& addr)
{
	addr.family = DetermineAddrFamily(lpszHost);

	if(addr.family != AF_UNSPEC)
		return GetSockAddr(lpszHost, usPort, addr);

	return GetSockAddrByHostNameDirectly(lpszHost, usPort, addr);
}

BOOL GetSockAddrByHostNameDirectly(LPCTSTR lpszHost, USHORT usPort, HP_SOCKADDR& addr)
{
	addr.ZeroAddr();

	addrinfo* pInfo	= nullptr;
	addrinfo hints	= {0};

#if defined(__ANDROID__)
	hints.ai_flags		= 0;
#else
	hints.ai_flags		= (AI_V4MAPPED | AI_ADDRCONFIG);
#endif
	hints.ai_family		= addr.family;
	hints.ai_socktype	= SOCK_STREAM;

	int rs = ::getaddrinfo(CT2A(lpszHost), nullptr, &hints, &pInfo);

	if(!IS_NO_ERROR(rs))
	{
		::WSASetLastError(ERROR_HOSTUNREACH);
		return FALSE;
	}

	BOOL isOK = FALSE;

	for(addrinfo* pCur = pInfo; pCur != nullptr; pCur = pCur->ai_next)
	{
		if(pCur->ai_family == AF_INET || pCur->ai_family == AF_INET6)
		{
			memcpy(addr.Addr(), pCur->ai_addr, pCur->ai_addrlen);
			isOK = TRUE;

			break;
		}
	}

	EXECUTE_RESTORE_ERROR(::freeaddrinfo(pInfo));

	if(isOK)
		addr.SetPort(usPort);
	else
		::WSASetLastError(ERROR_HOSTUNREACH);

	return isOK;
}

BOOL EnumHostIPAddresses(LPCTSTR lpszHost, EnIPAddrType enType, LPTIPAddr** lpppIPAddr, int& iIPAddrCount)
{
	*lpppIPAddr	 = nullptr;
	iIPAddrCount = 0;

	ADDRESS_FAMILY usFamily =	(enType		== IPT_ALL				?
								AF_UNSPEC	: (enType == IPT_IPV4	?
								AF_INET		: (enType == IPT_IPV6	?
								AF_INET6	: 0xFF)));

	if(usFamily == 0xFF)
	{
		::WSASetLastError(ERROR_AFNOSUPPORT);
		return FALSE;
	}

	vector<HP_PSOCKADDR> vt;

	ADDRESS_FAMILY usFamily2 = DetermineAddrFamily(lpszHost);

	if(usFamily2 != AF_UNSPEC)
	{
		if(usFamily != AF_UNSPEC && usFamily != usFamily2)
		{
			::WSASetLastError(ERROR_HOSTUNREACH);
			return FALSE;
		}

		HP_SOCKADDR addr(usFamily2);

		if(!GetSockAddr(lpszHost, 0, addr))
			return FALSE;

		vt.emplace_back(&addr);

		return RetrieveSockAddrIPAddresses(vt, lpppIPAddr, iIPAddrCount);
	}

	addrinfo* pInfo	= nullptr;
	addrinfo hints	= {0};

#if defined(__ANDROID__)
	hints.ai_flags		= 0;
#else
	hints.ai_flags		= AI_ALL;
#endif
	hints.ai_family		= usFamily;
	hints.ai_socktype	= SOCK_STREAM;

	int rs = ::getaddrinfo(CT2A(lpszHost), nullptr, &hints, &pInfo);

	if(rs != NO_ERROR)
	{
		::WSASetLastError(rs);
		return FALSE;
	}

	for(addrinfo* pCur = pInfo; pCur != nullptr; pCur = pCur->ai_next)
	{
		if(pCur->ai_family == AF_INET || pCur->ai_family == AF_INET6)
			vt.emplace_back((HP_PSOCKADDR)pCur->ai_addr);
	}

	BOOL isOK = RetrieveSockAddrIPAddresses(vt, lpppIPAddr, iIPAddrCount);

	::freeaddrinfo(pInfo);

	if(!isOK) ::WSASetLastError(EHOSTUNREACH);

	return isOK;
}

BOOL RetrieveSockAddrIPAddresses(const vector<HP_PSOCKADDR>& vt, LPTIPAddr** lpppIPAddr, int& iIPAddrCount)
{
	iIPAddrCount = (int)vt.size();

	if(iIPAddrCount == 0) return FALSE;

	HP_PSOCKADDR	pSockAddr;
	ADDRESS_FAMILY	usFamily;
	USHORT			usPort;
	int				iAddrLength;
	LPTSTR			lpszAddr;
	LPTIPAddr		lpItem;

	(*lpppIPAddr) = new LPTIPAddr[iIPAddrCount + 1];
	(*lpppIPAddr)[iIPAddrCount] = nullptr;

	for(int i = 0; i < iIPAddrCount; i++)
	{
		pSockAddr	= vt[i];
		iAddrLength	= HP_SOCKADDR::AddrMinStrLength(pSockAddr->family);
		lpszAddr	= new TCHAR[iAddrLength];

		VERIFY(sockaddr_IN_2_A(*vt[i], usFamily, lpszAddr, iAddrLength, usPort));

		lpItem			= new TIPAddr;
		lpItem->type	= pSockAddr->IsIPv4() ? IPT_IPV4 : IPT_IPV6;
		lpItem->address	= lpszAddr;

		(*lpppIPAddr)[i] = lpItem;
	}

	return TRUE;
}

BOOL FreeHostIPAddresses(LPTIPAddr* lppIPAddr)
{
	if(!lppIPAddr) return FALSE;

	LPTIPAddr p;
	LPTIPAddr* lppCur = lppIPAddr;

	while((p = *lppCur++) != nullptr)
	{
		delete[] p->address;
		delete p;
	}

	delete[] lppIPAddr;

	return TRUE;
}

BOOL sockaddr_IN_2_A(const HP_SOCKADDR& addr, ADDRESS_FAMILY& usFamily, LPTSTR lpszAddress, int& iAddressLen, USHORT& usPort)
{
	BOOL isOK	= FALSE;

	usFamily	= addr.family;
	usPort		= addr.Port();

	if(::InetNtop(addr.family, addr.SinAddr(), lpszAddress, iAddressLen))
	{
		iAddressLen	= (int)lstrlen(lpszAddress) + 1;
		isOK		= TRUE;
	}
	else
	{
		if(::WSAGetLastError() == ENOSPC)
			iAddressLen = HP_SOCKADDR::AddrMinStrLength(usFamily);
	}

	return isOK;
}

BOOL sockaddr_A_2_IN(LPCTSTR lpszAddress, USHORT usPort, HP_SOCKADDR& addr)
{
	addr.family = DetermineAddrFamily(lpszAddress);
	return GetSockAddr(lpszAddress, usPort, addr);
}

BOOL GetSocketAddress(SOCKET socket, LPTSTR lpszAddress, int& iAddressLen, USHORT& usPort, BOOL bLocal)
{
	HP_SOCKADDR addr;

	int addr_len = addr.AddrSize();
	int result	 = bLocal ? getsockname(socket, addr.Addr(), (socklen_t*)&addr_len) : getpeername(socket, addr.Addr(), (socklen_t*)&addr_len);

	if(result != NO_ERROR)
		return FALSE;

	ADDRESS_FAMILY usFamily;
	return sockaddr_IN_2_A(addr, usFamily, lpszAddress, iAddressLen, usPort);
}

BOOL GetSocketLocalAddress(SOCKET socket, LPTSTR lpszAddress, int& iAddressLen, USHORT& usPort)
{
	return GetSocketAddress(socket, lpszAddress, iAddressLen, usPort, TRUE);
}

BOOL GetSocketRemoteAddress(SOCKET socket, LPTSTR lpszAddress, int& iAddressLen, USHORT& usPort)
{
	return GetSocketAddress(socket, lpszAddress, iAddressLen, usPort, FALSE);
}

BOOL SetMultiCastSocketOptions(SOCKET sock, const HP_SOCKADDR& bindAddr, const HP_SOCKADDR& castAddr, int iMCTtl, BOOL bMCLoop)
{
	if(castAddr.IsIPv4())
	{
		BYTE ttl  = (BYTE)iMCTtl;
		BYTE loop = (BYTE)bMCLoop;

		VERIFY(::SSO_SetSocketOption(sock, IPPROTO_IP, IP_MULTICAST_TTL, &ttl, sizeof(ttl)) != SOCKET_ERROR);
		VERIFY(::SSO_SetSocketOption(sock, IPPROTO_IP, IP_MULTICAST_LOOP, &loop, sizeof(loop)) != SOCKET_ERROR);

		ip_mreq mcast;
		::ZeroMemory(&mcast, sizeof(mcast));

		mcast.imr_multiaddr = castAddr.addr4.sin_addr;
		mcast.imr_interface = bindAddr.addr4.sin_addr;

		if(::SSO_SetSocketOption(sock, IPPROTO_IP, IP_ADD_MEMBERSHIP, &mcast, sizeof(mcast)) == SOCKET_ERROR)
			return FALSE;
		if(::SSO_SetSocketOption(sock, IPPROTO_IP, IP_MULTICAST_IF, bindAddr.SinAddr(), sizeof(IN_ADDR)) == SOCKET_ERROR)
			return FALSE;
	}
	else
	{
		INT ttl	  = (INT)iMCTtl;
		UINT loop = (UINT)bMCLoop;

		VERIFY(::SSO_SetSocketOption(sock, IPPROTO_IPV6, IPV6_MULTICAST_HOPS, &ttl, sizeof(ttl)) != SOCKET_ERROR);
		VERIFY(::SSO_SetSocketOption(sock, IPPROTO_IPV6, IPV6_MULTICAST_LOOP, &loop, sizeof(loop)) != SOCKET_ERROR);

		ipv6_mreq mcast;
		::ZeroMemory(&mcast, sizeof(mcast));

		mcast.ipv6mr_multiaddr = castAddr.addr6.sin6_addr;
		mcast.ipv6mr_interface = bindAddr.addr6.sin6_scope_id;

		if(::SSO_SetSocketOption(sock, IPPROTO_IPV6, IPV6_ADD_MEMBERSHIP, &mcast, sizeof(mcast)) == SOCKET_ERROR)
			return FALSE;
		if(::SSO_SetSocketOption(sock, IPPROTO_IPV6, IPV6_MULTICAST_IF, (PVOID)(&bindAddr.addr6.sin6_scope_id), sizeof(UINT)) == SOCKET_ERROR)
			return FALSE;
	}

	return TRUE;
}

ULONGLONG NToH64(ULONGLONG value)
{
	return (((ULONGLONG)ntohl((UINT)((value << 32) >> 32))) << 32) | ntohl((UINT)(value >> 32));
}

ULONGLONG HToN64(ULONGLONG value)
{
	return (((ULONGLONG)htonl((UINT)((value << 32) >> 32))) << 32) | htonl((UINT)(value >> 32));
}

BOOL IsLittleEndian()
{
	static const USHORT _s_endian_test_value = 0x0102;
	static const BOOL _s_bLE = (*((BYTE*)&_s_endian_test_value) == 0x02);

	return _s_bLE;
}

USHORT HToLE16(USHORT value)
{
	return IsLittleEndian() ? value : ENDIAN_SWAP_16(value);
}

USHORT HToBE16(USHORT value)
{
	return IsLittleEndian() ? ENDIAN_SWAP_16(value) : value;
}

DWORD HToLE32(DWORD value)
{
	return IsLittleEndian() ? value : ENDIAN_SWAP_32(value);
}

DWORD HToBE32(DWORD value)
{
	return IsLittleEndian() ? ENDIAN_SWAP_32(value) : value;
}

HRESULT ReadSmallFile(LPCTSTR lpszFileName, CFile& file, CFileMapping& fmap, DWORD dwMaxFileSize)
{
	ASSERT(lpszFileName != nullptr);

	if(file.Open(lpszFileName, O_RDONLY))
	{
		SIZE_T dwSize;
		if(file.GetSize(dwSize))
		{
			if(dwSize > 0 && dwSize <= dwMaxFileSize)
			{
				if(fmap.Map(file, dwSize))
					return NO_ERROR;
			}
			else if(dwSize == 0)
				::SetLastError(ERROR_EMPTY);
			else
				::SetLastError(ERROR_FILE_TOO_LARGE);
		}
	}

	HRESULT rs = ::GetLastError();

	return (!IS_NO_ERROR(rs) ? rs : ERROR_UNKNOWN);
}

HRESULT MakeSmallFilePackage(LPCTSTR lpszFileName, CFile& file, CFileMapping& fmap, WSABUF szBuf[3], const LPWSABUF pHead, const LPWSABUF pTail)
{
	DWORD dwMaxFileSize = MAX_SMALL_FILE_SIZE - (pHead ? pHead->len : 0) - (pTail ? pTail->len : 0);
	ASSERT(dwMaxFileSize <= MAX_SMALL_FILE_SIZE);

	HRESULT hr = ReadSmallFile(lpszFileName, file, fmap, dwMaxFileSize);

	if(IS_NO_ERROR(hr))
	{
		szBuf[1].len = (UINT)fmap.Size();
		szBuf[1].buf = fmap;

		if(pHead) memcpy(&szBuf[0], pHead, sizeof(WSABUF));
		else	  memset(&szBuf[0], 0, sizeof(WSABUF));

		if(pTail) memcpy(&szBuf[2], pTail, sizeof(WSABUF));
		else	  memset(&szBuf[2], 0, sizeof(WSABUF));
	}

	return hr;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////

int SSO_SetSocketOption(SOCKET sock, int level, int name, LPVOID val, int len)
{
	return setsockopt(sock, level, name, val, (socklen_t)len);
}

int SSO_GetSocketOption(SOCKET sock, int level, int name, LPVOID val, int* len)
{
	return getsockopt(sock, level, name, val, (socklen_t*)len);
}

int SSO_IoctlSocket(SOCKET sock, long cmd, PVOID arg)
{
	return ioctl(sock, cmd, arg);
}

int SSO_NoBlock(SOCKET sock, BOOL bNoBlock)
{
	return fcntl_SETFL(sock, O_NONBLOCK, bNoBlock) ? NO_ERROR : SOCKET_ERROR;
}

int SSO_NoDelay(SOCKET sock, BOOL bNoDelay)
{
	int val = bNoDelay ? 1 : 0;
	return setsockopt(sock, IPPROTO_TCP, TCP_NODELAY, &val, sizeof(int));
}

int SSO_DontLinger(SOCKET sock, BOOL bDont)
{
	return SSO_Linger(sock, 0, 0);
}

int SSO_Linger(SOCKET sock, int l_onoff, int l_linger)
{
	linger ln = {l_onoff, l_linger};
	return setsockopt(sock, SOL_SOCKET, SO_LINGER, &ln, sizeof(linger));
}

int SSO_KeepAlive(SOCKET sock, BOOL bKeepAlive)
{
	int val = bKeepAlive ? 1 : 0;
	return setsockopt(sock, SOL_SOCKET, SO_KEEPALIVE, &val, sizeof(int));
}

int SSO_KeepAliveVals(SOCKET sock, BOOL bOnOff, DWORD dwIdle, DWORD dwInterval, DWORD dwCount)
{
	if(bOnOff)
	{
		dwIdle		/= 1000;
		dwInterval	/= 1000;

		if(dwIdle == 0 || dwInterval == 0 || dwCount == 0)
		{
			::WSASetLastError(ERROR_INVALID_PARAMETER);
			return SOCKET_ERROR;
		}
	}

	BOOL isOK = IS_NO_ERROR(SSO_KeepAlive(sock, bOnOff));

	if(isOK && bOnOff)
	{
		isOK &= IS_NO_ERROR(setsockopt(sock, SOL_TCP, TCP_KEEPIDLE, &dwIdle, sizeof(DWORD)));
		isOK &= IS_NO_ERROR(setsockopt(sock, SOL_TCP, TCP_KEEPINTVL, &dwInterval, sizeof(DWORD)));
		isOK &= IS_NO_ERROR(setsockopt(sock, SOL_TCP, TCP_KEEPCNT, &dwCount, sizeof(DWORD)));
	}

	return isOK ? NO_ERROR : SOCKET_ERROR;
}

int SSO_ReuseAddress(SOCKET sock, EnReuseAddressPolicy opt)
{
	int iSet	= 1;
	int iUnSet	= 0;
	int rs		= NO_ERROR;

	BOOL bReusePortSupported =
#if defined(__linux) || defined(__linux__)
		::IsKernelVersionAbove(3, 9, 0);
#elif defined(__FreeBSD__) || defined(__NetBSD__) || defined(__OpenBSD__) || defined(__bsdi__) || defined(__APPLE__) || defined(__MACH__)
		TRUE;
#else
		FALSE;
#endif

	if(opt == RAP_NONE)
	{
		rs  = setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, &iUnSet, sizeof(int));
		if(bReusePortSupported)
		rs |= setsockopt(sock, SOL_SOCKET, SO_REUSEPORT, &iUnSet, sizeof(int));
	}
	else if(opt == RAP_ADDR_ONLY)
	{
		rs  = setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, &iSet, sizeof(int));
		if(bReusePortSupported)
		rs |= setsockopt(sock, SOL_SOCKET, SO_REUSEPORT, &iUnSet, sizeof(int));
	}
	else if(opt == RAP_ADDR_AND_PORT)
	{
		rs  = setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, &iSet, sizeof(int));
		if(bReusePortSupported)
		rs |= setsockopt(sock, SOL_SOCKET, SO_REUSEPORT, &iSet, sizeof(int));
	}
	else
	{
		::SetLastError(ERROR_INVALID_PARAMETER);
		rs = -1;
	}

	return rs;
}

int SSO_RecvBuffSize(SOCKET sock, int size)
{
	return setsockopt(sock, SOL_SOCKET, SO_RCVBUF, &size, sizeof(int));
}

int SSO_SendBuffSize(SOCKET sock, int size)
{
	return setsockopt(sock, SOL_SOCKET, SO_SNDBUF, &size, sizeof(int));
}

int SSO_RecvTimeOut(SOCKET sock, int ms)
{
	timeval tv;
	::MillisecondToTimeval(ms, tv);
	
	return setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(timeval));
}

int SSO_SendTimeOut(SOCKET sock, int ms)
{
	timeval tv;
	::MillisecondToTimeval(ms, tv);

	return setsockopt(sock, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(timeval));
}

int SSO_GetError(SOCKET sock)
{
	int e;
	socklen_t len = sizeof(e);

	if(IS_NO_ERROR(getsockopt(sock, SOL_SOCKET, SO_ERROR, &e, &len)))
		return e;

	return SOCKET_ERROR;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////

CONNID GenerateConnectionID()
{
	static volatile CONNID s_dwConnID = 0;

	CONNID dwConnID	= ::InterlockedIncrement(&s_dwConnID);
	
	if(dwConnID == 0)
		dwConnID = ::InterlockedIncrement(&s_dwConnID);

	return dwConnID;
}

int IsUdpCloseNotify(const BYTE* pData, int iLength)
{
	return (iLength == s_iUdpCloseNotifySize								&&
			memcmp(pData, s_szUdpCloseNotify, s_iUdpCloseNotifySize) == 0)	;
}

int SendUdpCloseNotify(SOCKET sock)
{
	return (int)send(sock, (LPCSTR)s_szUdpCloseNotify, s_iUdpCloseNotifySize, 0);
}

int SendUdpCloseNotify(SOCKET sock, const HP_SOCKADDR& remoteAddr)
{
	return (int)sendto(sock, (LPCSTR)s_szUdpCloseNotify, s_iUdpCloseNotifySize, 0, remoteAddr.Addr(), remoteAddr.AddrSize());
}

int ManualCloseSocket(SOCKET sock, int iShutdownFlag, BOOL bGraceful)
{
	if(!bGraceful)
		SSO_Linger(sock, 1, 0);

	if(iShutdownFlag != 0xFF)
		shutdown(sock, iShutdownFlag);

	return closesocket(sock);
}

DWORD GuessBase64EncodeBound(DWORD dwSrcLen)
{
	return 4 * ((dwSrcLen + 2) / 3);
}

DWORD GuessBase64DecodeBound(const BYTE* lpszSrc, DWORD dwSrcLen)
{
	if(dwSrcLen < 2)
		return 0;

	if(lpszSrc[dwSrcLen - 2] == '=')
		dwSrcLen -= 2;
	else if(lpszSrc[dwSrcLen - 1] == '=')
			--dwSrcLen;

	DWORD dwMod = dwSrcLen % 4;
	DWORD dwAdd = dwMod == 2 ? 1 : (dwMod == 3 ? 2 : 0);

	return 3 * (dwSrcLen / 4) + dwAdd;
}

int Base64Encode(const BYTE* lpszSrc, DWORD dwSrcLen, BYTE* lpszDest, DWORD& dwDestLen)
{
	static const BYTE CODES[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

	DWORD dwRealLen = GuessBase64EncodeBound(dwSrcLen);

	if(lpszDest == nullptr || dwDestLen < dwRealLen)
	{
		dwDestLen = dwRealLen;
		return -5;
	}

	BYTE* p		= lpszDest;
	DWORD leven	= 3 * (dwSrcLen / 3);
	DWORD i		= 0;

	for (; i < leven; i += 3)
	{
		*p++ = CODES[lpszSrc[0] >> 2];
		*p++ = CODES[((lpszSrc[0] & 3) << 4) + (lpszSrc[1] >> 4)];
		*p++ = CODES[((lpszSrc[1] & 0xf) << 2) + (lpszSrc[2] >> 6)];
		*p++ = CODES[lpszSrc[2] & 0x3f];

		lpszSrc += 3;
	}

	if(i < dwSrcLen)
	{
		BYTE a = lpszSrc[0];
		BYTE b = (i + 1 < dwSrcLen) ? lpszSrc[1] : 0;

		*p++ = CODES[a >> 2];
		*p++ = CODES[((a & 3) << 4) + (b >> 4)];
		*p++ = (i + 1 < dwSrcLen) ? CODES[((b & 0xf) << 2)] : '=';
		*p++ = '=';
	}  

	ASSERT(dwRealLen == (DWORD)(p - lpszDest));

	if(dwDestLen > dwRealLen)
	{
		*p			= 0;
		dwDestLen	= dwRealLen;
	}

	return 0;  
}

int Base64Decode(const BYTE* lpszSrc, DWORD dwSrcLen, BYTE* lpszDest, DWORD& dwDestLen)
{
	static const BYTE MAP[256]	=
	{ 
		255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 253, 255,
		255, 253, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
		255, 255, 255, 255, 255, 255, 255, 255, 253, 255, 255, 255,
		255, 255, 255, 255, 255, 255, 255,  62, 255, 255, 255,  63,
		 52,  53,  54,  55,  56,  57,  58,  59,  60,  61, 255, 255,
		255, 254, 255, 255, 255,   0,   1,   2,   3,   4,   5,   6,
		  7,   8,   9,  10,  11,  12,  13,  14,  15,  16,  17,  18,
		 19,  20,  21,  22,  23,  24,  25, 255, 255, 255, 255, 255,
		255,  26,  27,  28,  29,  30,  31,  32,  33,  34,  35,  36,
		 37,  38,  39,  40,  41,  42,  43,  44,  45,  46,  47,  48,
		 49,  50,  51, 255, 255, 255, 255, 255, 255, 255, 255, 255,
		255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
		255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
		255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
		255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
		255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
		255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
		255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
		255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
		255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
		255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
		255, 255, 255, 255
	};

	DWORD dwRealLen = GuessBase64DecodeBound(lpszSrc, dwSrcLen);

	if(lpszDest == nullptr || dwDestLen < dwRealLen)
	{
		dwDestLen = dwRealLen;
		return -5;
	}

	BYTE c;
	int g = 3;
	DWORD i, x, y, z;

	for(i = x = y = z = 0; i < dwSrcLen || x != 0;)
	{
		c = i < dwSrcLen ? MAP[lpszSrc[i++]] : 254;

		if(c == 255) {dwDestLen = 0; return -3;}
		else if(c == 254) {c = 0; g--;}
		else if(c == 253) continue;

		z = (z << 6) | c;

		if(++x == 4)
		{
			lpszDest[y++] = (BYTE)((z >> 16) & 255);
			if (g > 1) lpszDest[y++] = (BYTE)((z >> 8) & 255);
			if (g > 2) lpszDest[y++] = (BYTE)(z & 255);

			x = z = 0;
		}
	}

	BOOL isOK = (y == dwRealLen);

	if(!isOK)
		dwDestLen = 0;
	else
	{
		if(dwDestLen > dwRealLen)
		{
			lpszDest[dwRealLen]	= 0;
			dwDestLen			= dwRealLen;
		}
	}

	return isOK ? 0 : -3;
}

DWORD GuessUrlEncodeBound(const BYTE* lpszSrc, DWORD dwSrcLen)
{
	DWORD dwAdd = 0;

	for(DWORD i = 0; i < dwSrcLen; i++)
	{
		BYTE c	= lpszSrc[i];

		if(!(isalnum(c) || c == ' ' || c == '.' || c == '-' || c == '_' || c == '*'))
			dwAdd += 2;
	}

	return dwSrcLen + dwAdd;
}

DWORD GuessUrlDecodeBound(const BYTE* lpszSrc, DWORD dwSrcLen)
{
	DWORD dwPercent = 0;

	for(DWORD i = 0; i < dwSrcLen; i++)
	{
		if(lpszSrc[i] == '%')
		{
			++dwPercent;
			i += 2;
		}
	}

	DWORD dwSub = dwPercent * 2;

	if(dwSrcLen < dwSub)
		return 0;

	return dwSrcLen - dwSub;
}

int UrlEncode(BYTE* lpszSrc, DWORD dwSrcLen, BYTE* lpszDest, DWORD& dwDestLen)
{
	BYTE c;
	DWORD j = 0;

	if(lpszDest == nullptr || dwDestLen == 0)
		goto ERROR_DEST_LEN;

	for(DWORD i = 0; i < dwSrcLen; i++)
	{
		if(j >= dwDestLen)
			goto ERROR_DEST_LEN;

		c = lpszSrc[i];

		if (isalnum(c) || c == '.' || c == '-' || c == '_' || c == '*')
			lpszDest[j++] = c;
		else if(c == ' ')
			lpszDest[j++] = '+';
		else
		{
			if(j + 3 >= dwDestLen)
				goto ERROR_DEST_LEN;

			lpszDest[j++] = '%';
			HEX_VALUE_TO_DOUBLE_CHAR(lpszDest + j, c);
			j += 2;
			
		}
	}

	if(dwDestLen > j)
	{
		lpszDest[j]	= 0;
		dwDestLen	= j;
	}

	return 0;

ERROR_DEST_LEN:
	dwDestLen = GuessUrlEncodeBound(lpszSrc, dwSrcLen);
	return -5;
}

int UrlDecode(BYTE* lpszSrc, DWORD dwSrcLen, BYTE* lpszDest, DWORD& dwDestLen)
{
	char c;
	DWORD j = 0;

	if(lpszDest == nullptr || dwDestLen == 0)
		goto ERROR_DEST_LEN;

	for(DWORD i = 0; i < dwSrcLen; i++)
	{
		if(j >= dwDestLen)
			goto ERROR_DEST_LEN;

		c = lpszSrc[i];

		if(c == '+')
			lpszDest[j++] = ' ';
		else if(c != '%')
			lpszDest[j++] = c;
		else
		{
			if(i + 2 >= dwSrcLen)
				goto ERROR_SRC_DATA;

			lpszDest[j++] = HEX_DOUBLE_CHAR_TO_VALUE(lpszSrc + i + 1);
			i += 2;
		}
	}

	if(dwDestLen > j)
	{
		lpszDest[j]	= 0;
		dwDestLen	= j;
	}

	return 0;

ERROR_SRC_DATA:
	dwDestLen = 0;
	return -3;

ERROR_DEST_LEN:
	dwDestLen = GuessUrlDecodeBound(lpszSrc, dwSrcLen);
	return -5;
}

void DestroyCompressor(IHPCompressor* pCompressor)
{
	delete pCompressor;
}

void DestroyDecompressor(IHPDecompressor* pDecompressor)
{
	delete pDecompressor;
}

#ifdef _ZLIB_SUPPORT

CHPZLibCompressor::CHPZLibCompressor(Fn_CompressDataCallback fnCallback, int iWindowBits, int iLevel, int iMethod, int iMemLevel, int iStrategy, DWORD dwBuffSize)
: m_fnCallback	(fnCallback)
, m_dwBuffSize	(dwBuffSize)
, m_bValid		(FALSE)
{
	ASSERT(m_fnCallback != nullptr);

	::ZeroObject(m_Stream);

	m_bValid = (::deflateInit2(&m_Stream, iLevel, iMethod, iWindowBits, iMemLevel, iStrategy) == Z_OK);
}
CHPZLibCompressor::~CHPZLibCompressor()
{
	if(m_bValid) ::deflateEnd(&m_Stream);
}

BOOL CHPZLibCompressor::Reset()
{
	return (m_bValid = (::deflateReset(&m_Stream) == Z_OK));
}

BOOL CHPZLibCompressor::Process(const BYTE* pData, int iLength, BOOL bLast, PVOID pContext)
{
	return ProcessEx(pData, iLength, bLast, FALSE, pContext);
}

BOOL CHPZLibCompressor::ProcessEx(const BYTE* pData, int iLength, BOOL bLast, BOOL bFlush, PVOID pContext)
{
	ASSERT(IsValid() && iLength > 0);

	if(!IsValid())
	{
		::SetLastError(ERROR_INVALID_STATE);
		return FALSE;
	}

	unique_ptr<BYTE[]> szBuff = make_unique<BYTE[]>(m_dwBuffSize);

	m_Stream.next_in	= (z_const Bytef*)pData;
	m_Stream.avail_in	= iLength;
	
	BOOL isOK	= TRUE;
	int rs		= Z_OK;
	int flush	= bLast ? Z_FINISH : (bFlush ? Z_SYNC_FLUSH : Z_NO_FLUSH);

	while(m_Stream.avail_in > 0)
	{
		do
		{
			m_Stream.next_out  = szBuff.get();
			m_Stream.avail_out = m_dwBuffSize;

			rs = ::deflate(&m_Stream, flush);

			if(rs == Z_STREAM_ERROR)
			{
				::SetLastError(ERROR_INVALID_DATA);
				isOK = FALSE;
				
				goto ZLIB_COMPRESS_END;
			}

			int iRead = (int)(m_dwBuffSize - m_Stream.avail_out);

			if(iRead == 0)
				break;

			if(!m_fnCallback(szBuff.get(), iRead, pContext))
			{
				::SetLastError(ERROR_CANCELLED);
				isOK = FALSE;

				goto ZLIB_COMPRESS_END;
			}
		} while(m_Stream.avail_out == 0);
	}

ZLIB_COMPRESS_END:

	ASSERT(!isOK || (rs == Z_OK && !bLast) || (rs == Z_STREAM_END && bLast));

	if(!isOK || bLast) Reset();

	return isOK;
}

CHPZLibDecompressor::CHPZLibDecompressor(Fn_DecompressDataCallback fnCallback, int iWindowBits, DWORD dwBuffSize)
: m_fnCallback	(fnCallback)
, m_dwBuffSize	(dwBuffSize)
, m_bValid		(FALSE)
{
	ASSERT(m_fnCallback != nullptr);

	::ZeroObject(m_Stream);

	m_bValid = (::inflateInit2(&m_Stream, iWindowBits) == Z_OK);
}
CHPZLibDecompressor::~CHPZLibDecompressor()
{
	if(m_bValid) ::inflateEnd(&m_Stream);
}

BOOL CHPZLibDecompressor::Reset()
{
	return (m_bValid = (::inflateReset(&m_Stream) == Z_OK));
}

BOOL CHPZLibDecompressor::Process(const BYTE* pData, int iLength, PVOID pContext)
{
	ASSERT(IsValid() && iLength > 0);

	if(!IsValid())
	{
		::SetLastError(ERROR_INVALID_STATE);
		return FALSE;
	}

	unique_ptr<BYTE[]> szBuff = make_unique<BYTE[]>(m_dwBuffSize);

	m_Stream.next_in	= (z_const Bytef*)pData;
	m_Stream.avail_in	= iLength;

	BOOL isOK	= TRUE;
	int rs		= Z_OK;

	while(m_Stream.avail_in > 0)
	{
		do
		{
			m_Stream.next_out  = szBuff.get();
			m_Stream.avail_out = m_dwBuffSize;

			rs = ::inflate(&m_Stream, Z_NO_FLUSH);

			if(rs != Z_OK && rs != Z_STREAM_END)
			{
				::SetLastError(ERROR_INVALID_DATA);
				isOK = FALSE;

				goto ZLIB_DECOMPRESS_END;
			}

			int iRead = (int)(m_dwBuffSize - m_Stream.avail_out);

			if(iRead == 0)
				break;

			if(!m_fnCallback(szBuff.get(), iRead, pContext))
			{
				::SetLastError(ERROR_CANCELLED);
				isOK = FALSE;

				goto ZLIB_DECOMPRESS_END;
			}
		} while(m_Stream.avail_out == 0);

		if(rs == Z_STREAM_END)
			break;
	}

ZLIB_DECOMPRESS_END:

	ASSERT(!isOK || rs == Z_OK || rs == Z_STREAM_END);

	if(!isOK || rs == Z_STREAM_END) Reset();

	return isOK;
}

IHPCompressor* CreateZLibCompressor(Fn_CompressDataCallback fnCallback, int iWindowBits, int iLevel, int iMethod, int iMemLevel, int iStrategy, DWORD dwBuffSize)
{
	return new CHPZLibCompressor(fnCallback, iWindowBits, iLevel, iMethod, iMemLevel, iStrategy, dwBuffSize);
}

IHPCompressor* CreateGZipCompressor(Fn_CompressDataCallback fnCallback, int iLevel, int iMethod, int iMemLevel, int iStrategy, DWORD dwBuffSize)
{
	return new CHPZLibCompressor(fnCallback, MAX_WBITS + 16, iLevel, iMethod, iMemLevel, iStrategy, dwBuffSize);
}

IHPDecompressor* CreateZLibDecompressor(Fn_DecompressDataCallback fnCallback, int iWindowBits, DWORD dwBuffSize)
{
	return new CHPZLibDecompressor(fnCallback, iWindowBits, dwBuffSize);
}

IHPDecompressor* CreateGZipDecompressor(Fn_DecompressDataCallback fnCallback, DWORD dwBuffSize)
{
	return new CHPZLibDecompressor(fnCallback, MAX_WBITS + 32, dwBuffSize);
}

int Compress(const BYTE* lpszSrc, DWORD dwSrcLen, BYTE* lpszDest, DWORD& dwDestLen)
{
	return CompressEx(lpszSrc, dwSrcLen, lpszDest, dwDestLen);
}

int CompressEx(const BYTE* lpszSrc, DWORD dwSrcLen, BYTE* lpszDest, DWORD& dwDestLen, int iLevel, int iMethod, int iWindowBits, int iMemLevel, int iStrategy)
{
	z_stream stream;

	stream.next_in	 = (z_const Bytef*)lpszSrc;
	stream.avail_in	 = dwSrcLen;
	stream.next_out	 = lpszDest;
	stream.avail_out = dwDestLen;
	stream.zalloc	 = nullptr;
	stream.zfree	 = nullptr;
	stream.opaque	 = nullptr;

	int err = ::deflateInit2(&stream, iLevel, iMethod, iWindowBits, iMemLevel, iStrategy);

	if(err != Z_OK) return err;

	err = ::deflate(&stream, Z_FINISH);

	if(err != Z_STREAM_END)
	{
		::deflateEnd(&stream);
		return err == Z_OK ? Z_BUF_ERROR : err;
	}

	if(dwDestLen > stream.total_out)
	{
		lpszDest[stream.total_out]	= 0;
		dwDestLen					= (DWORD)stream.total_out;
	}

	return ::deflateEnd(&stream);
}

int Uncompress(const BYTE* lpszSrc, DWORD dwSrcLen, BYTE* lpszDest, DWORD& dwDestLen)
{
	return UncompressEx(lpszSrc, dwSrcLen, lpszDest, dwDestLen);
}

int UncompressEx(const BYTE* lpszSrc, DWORD dwSrcLen, BYTE* lpszDest, DWORD& dwDestLen, int iWindowBits)
{
	z_stream stream;

	stream.next_in	 = (z_const Bytef*)lpszSrc;
	stream.avail_in	 = (uInt)dwSrcLen;
	stream.next_out	 = lpszDest;
	stream.avail_out = dwDestLen;
	stream.zalloc	 = nullptr;
	stream.zfree	 = nullptr;

	int err = ::inflateInit2(&stream, iWindowBits);

	if(err != Z_OK) return err;

	err = ::inflate(&stream, Z_FINISH);

	if(err != Z_STREAM_END)
	{
		::inflateEnd(&stream);
		return (err == Z_NEED_DICT || (err == Z_BUF_ERROR && stream.avail_in == 0)) ? Z_DATA_ERROR : err;
	}

	if(dwDestLen > stream.total_out)
	{
		lpszDest[stream.total_out]	= 0;
		dwDestLen					= (DWORD)stream.total_out;
	}

	return inflateEnd(&stream);
}

DWORD GuessCompressBound(DWORD dwSrcLen, BOOL bGZip)
{
	DWORD dwBound = (DWORD)::compressBound(dwSrcLen);
	
	if(bGZip) dwBound += 16;

	return dwBound;
}

int GZipCompress(const BYTE* lpszSrc, DWORD dwSrcLen, BYTE* lpszDest, DWORD& dwDestLen)
{
	return CompressEx(lpszSrc, dwSrcLen, lpszDest, dwDestLen, Z_DEFAULT_COMPRESSION, Z_DEFLATED, MAX_WBITS + 16);
}

int GZipUncompress(const BYTE* lpszSrc, DWORD dwSrcLen, BYTE* lpszDest, DWORD& dwDestLen)
{
	return UncompressEx(lpszSrc, dwSrcLen, lpszDest, dwDestLen, MAX_WBITS + 32);
}

DWORD GZipGuessUncompressBound(const BYTE* lpszSrc, DWORD dwSrcLen)
{
	if(dwSrcLen < 20 || *(USHORT*)lpszSrc != 0x8B1F)
		return 0;

	return *(DWORD*)(lpszSrc + dwSrcLen - 4);
}

#endif

#ifdef _BROTLI_SUPPORT

CHPBrotliCompressor::CHPBrotliCompressor(Fn_CompressDataCallback fnCallback, int iQuality, int iWindow, int iMode, DWORD dwBuffSize)
: m_fnCallback	(fnCallback)
, m_iQuality	(iQuality)
, m_iWindow		(iWindow)
, m_iMode		(iMode)
, m_dwBuffSize	(dwBuffSize)
, m_bValid		(FALSE)
{
	ASSERT(m_fnCallback != nullptr);

	Reset();
}

CHPBrotliCompressor::~CHPBrotliCompressor()
{
	if(m_bValid) ::BrotliEncoderDestroyInstance(m_pState);
}

BOOL CHPBrotliCompressor::Reset()
{
	if(m_bValid) ::BrotliEncoderDestroyInstance(m_pState);
	m_pState =   ::BrotliEncoderCreateInstance(nullptr, nullptr, nullptr);

	if(m_pState != nullptr)
	{
		::BrotliEncoderSetParameter(m_pState, BROTLI_PARAM_QUALITY	, (UINT)m_iQuality);
		::BrotliEncoderSetParameter(m_pState, BROTLI_PARAM_LGWIN	, (UINT)m_iWindow);
		::BrotliEncoderSetParameter(m_pState, BROTLI_PARAM_MODE		, (UINT)m_iMode);

		if (m_iWindow > BROTLI_MAX_WINDOW_BITS)
			::BrotliEncoderSetParameter(m_pState, BROTLI_PARAM_LARGE_WINDOW, BROTLI_TRUE);
	}

	return (m_bValid = (m_pState != nullptr));
}

BOOL CHPBrotliCompressor::Process(const BYTE* pData, int iLength, BOOL bLast, PVOID pContext)
{
	return ProcessEx(pData, iLength, bLast, FALSE, pContext);
}

BOOL CHPBrotliCompressor::ProcessEx(const BYTE* pData, int iLength, BOOL bLast, BOOL bFlush, PVOID pContext)
{
	ASSERT(IsValid() && iLength > 0);

	if(!IsValid())
	{
		::SetLastError(ERROR_INVALID_STATE);
		return FALSE;
	}

	unique_ptr<BYTE[]> szBuff = make_unique<BYTE[]>(m_dwBuffSize);

	const BYTE* pNextInData	= pData;
	size_t iAvlInLen		= (SIZE_T)iLength;
	BYTE* pNextOutData		= nullptr;
	size_t iAvlOutLen		= 0;

	BOOL isOK				  = TRUE;
	BrotliEncoderOperation op = bLast ? BROTLI_OPERATION_FINISH : (bFlush ? BROTLI_OPERATION_FLUSH : BROTLI_OPERATION_PROCESS);

	while(iAvlInLen > 0)
	{
		do
		{
			pNextOutData = szBuff.get();
			iAvlOutLen	 = m_dwBuffSize;

			if(!::BrotliEncoderCompressStream(m_pState, op, &iAvlInLen, &pNextInData, &iAvlOutLen, &pNextOutData, nullptr))
			{
				::SetLastError(ERROR_INVALID_DATA);
				isOK = FALSE;

				goto BROTLI_COMPRESS_END;
			}

			int iRead = (int)(m_dwBuffSize - iAvlOutLen);

			if(iRead == 0)
				break;

			if(!m_fnCallback(szBuff.get(), iRead, pContext))
			{
				::SetLastError(ERROR_CANCELLED);
				isOK = FALSE;
				
				goto BROTLI_COMPRESS_END;
			}
		} while (iAvlOutLen == 0);
	}

BROTLI_COMPRESS_END:

	if(!isOK || bLast) Reset();

	return isOK;
}

CHPBrotliDecompressor::CHPBrotliDecompressor(Fn_DecompressDataCallback fnCallback, DWORD dwBuffSize)
: m_fnCallback	(fnCallback)
, m_dwBuffSize	(dwBuffSize)
, m_bValid		(FALSE)
{
	ASSERT(m_fnCallback != nullptr);

	Reset();
}

CHPBrotliDecompressor::~CHPBrotliDecompressor()
{
	if(m_bValid) ::BrotliDecoderDestroyInstance(m_pState);
}

BOOL CHPBrotliDecompressor::Reset()
{
	if(m_bValid) ::BrotliDecoderDestroyInstance(m_pState);
	m_pState =   ::BrotliDecoderCreateInstance(nullptr, nullptr, nullptr);

	return (m_bValid = (m_pState != nullptr));
}

BOOL CHPBrotliDecompressor::Process(const BYTE* pData, int iLength, PVOID pContext)
{
	ASSERT(IsValid() && iLength > 0);

	if(!IsValid())
	{
		::SetLastError(ERROR_INVALID_STATE);
		return FALSE;
	}

	unique_ptr<BYTE[]> szBuff = make_unique<BYTE[]>(m_dwBuffSize);

	const BYTE* pNextInData	= pData;
	size_t iAvlInLen		= (SIZE_T)iLength;
	BYTE* pNextOutData		= nullptr;
	size_t iAvlOutLen		= 0;

	BOOL isOK				= TRUE;
	BrotliDecoderResult rs	= BROTLI_DECODER_RESULT_NEEDS_MORE_OUTPUT;

	do
	{
		do
		{
			pNextOutData = szBuff.get();
			iAvlOutLen	 = m_dwBuffSize;

			rs = ::BrotliDecoderDecompressStream(m_pState, &iAvlInLen, &pNextInData, &iAvlOutLen, &pNextOutData, nullptr);

			if(rs == BROTLI_DECODER_RESULT_ERROR)
			{
				::SetLastError(ERROR_INVALID_DATA);
				isOK = FALSE;

				goto BROTLI_DECOMPRESS_END;
			}

			int iRead = (int)(m_dwBuffSize - iAvlOutLen);

			if(iRead == 0)
				break;

			if(!m_fnCallback(szBuff.get(), iRead, pContext))
			{
				::SetLastError(ERROR_CANCELLED);
				isOK = FALSE;

				goto BROTLI_DECOMPRESS_END;
			}
		} while (iAvlOutLen == 0);

		if(rs == BROTLI_DECODER_RESULT_SUCCESS)
			break;

	} while(rs == BROTLI_DECODER_RESULT_NEEDS_MORE_OUTPUT);

BROTLI_DECOMPRESS_END:

	if(!isOK || rs == BROTLI_DECODER_RESULT_SUCCESS) Reset();

	return isOK;
}

IHPCompressor* CreateBrotliCompressor(Fn_CompressDataCallback fnCallback, int iQuality, int iWindow, int iMode, DWORD dwBuffSize)
{
	return new CHPBrotliCompressor(fnCallback, iQuality, iWindow, iMode, dwBuffSize);
}

IHPDecompressor* CreateBrotliDecompressor(Fn_DecompressDataCallback fnCallback, DWORD dwBuffSize)
{
	return new CHPBrotliDecompressor(fnCallback, dwBuffSize);
}

int BrotliCompress(const BYTE* lpszSrc, DWORD dwSrcLen, BYTE* lpszDest, DWORD& dwDestLen)
{
	return BrotliCompressEx(lpszSrc, dwSrcLen, lpszDest, dwDestLen, BROTLI_DEFAULT_QUALITY, BROTLI_DEFAULT_WINDOW, BROTLI_DEFAULT_MODE);
}

int BrotliCompressEx(const BYTE* lpszSrc, DWORD dwSrcLen, BYTE* lpszDest, DWORD& dwDestLen, int iQuality, int iWindow, int iMode)
{
	size_t stDestLen = (size_t)dwDestLen;
	int rs = ::BrotliEncoderCompress(iQuality, iWindow, (BrotliEncoderMode)iMode, (size_t)dwSrcLen, lpszSrc, &stDestLen, lpszDest);
	dwDestLen = (DWORD)stDestLen;

	return (rs == 1) ? 0 : ((rs == 3) ? -5 : -3);
}

int BrotliUncompress(const BYTE* lpszSrc, DWORD dwSrcLen, BYTE* lpszDest, DWORD& dwDestLen)
{
	size_t stDestLen = (size_t)dwDestLen;
	BrotliDecoderResult rs = ::BrotliDecoderDecompress((size_t)dwSrcLen, lpszSrc, &stDestLen, lpszDest);
	dwDestLen = (DWORD)stDestLen;

	return (rs == BROTLI_DECODER_RESULT_SUCCESS) ? 0 : ((rs == BROTLI_DECODER_RESULT_NEEDS_MORE_OUTPUT) ? -5 : -3);
}

DWORD BrotliGuessCompressBound(DWORD dwSrcLen)
{
	return (DWORD)::BrotliEncoderMaxCompressedSize((size_t)dwSrcLen);
}

#endif

#ifdef _ICONV_SUPPORT

BOOL CharsetConvert(LPCSTR lpszFromCharset, LPCSTR lpszToCharset, LPCSTR lpszInBuf, int iInBufLen, LPSTR lpszOutBuf, int& iOutBufLen)
{
	ASSERT(lpszInBuf != nullptr);

	SIZE_T nInBufLeft	= iInBufLen;
	SIZE_T nOutBufLeft	= iOutBufLen;
	int iOutBufSize		= iOutBufLen;
	iOutBufLen			= 0;

	if(lpszInBuf == nullptr)
	{
		SetLastError(ERROR_INVALID_PARAMETER);
		return FALSE;
	}

	iconv_t ic = iconv_open(lpszToCharset, lpszFromCharset);

	if(IS_INVALID_PVOID(ic))
		return FALSE;

	SIZE_T rs	= iconv(ic, (LPSTR*)&lpszInBuf, &nInBufLeft, &lpszOutBuf, &nOutBufLeft);
	iOutBufLen	= iOutBufSize - (int)nOutBufLeft;

	EXECUTE_RESTORE_ERROR(iconv_close(ic));

	return !IS_HAS_ERROR(rs);
}

BOOL GbkToUnicodeEx(const char szSrc[], int iSrcLength, WCHAR szDest[], int& iDestLength)
{
	int iInBufLen	= (int)((iSrcLength > 0) ? iSrcLength : ((szSrc != nullptr) ? strlen(szSrc) + 1 : 0));
	int iOutBufLen	= (int)(iDestLength * sizeof(WCHAR));

	BOOL isOK	= CharsetConvert(CHARSET_GBK, SYSTEM_CHARSET_UNICODE, szSrc, iInBufLen, (char*)szDest, iOutBufLen);
	iDestLength	= (int)(iOutBufLen / sizeof(WCHAR));

	return isOK;
}

BOOL UnicodeToGbkEx(const WCHAR szSrc[], int iSrcLength, char szDest[], int& iDestLength)
{
	int iInBufLen = (int)(((iSrcLength > 0) ? iSrcLength : ((szSrc != nullptr) ? wcslen(szSrc) + 1 : 0)) * sizeof(WCHAR));

	return CharsetConvert(SYSTEM_CHARSET_UNICODE, CHARSET_GBK, (LPCSTR)szSrc, iInBufLen, szDest, iDestLength);
}

BOOL Utf8ToUnicodeEx(const char szSrc[], int iSrcLength, WCHAR szDest[], int& iDestLength)
{
	int iInBufLen	= (int)((iSrcLength > 0) ? iSrcLength : ((szSrc != nullptr) ? strlen(szSrc) + 1 : 0));
	int iOutBufLen	= (int)(iDestLength * sizeof(WCHAR));

	BOOL isOK	= CharsetConvert(CHARSET_UTF_8, SYSTEM_CHARSET_UNICODE, szSrc, iInBufLen, (char*)szDest, iOutBufLen);
	iDestLength	= (int)(iOutBufLen / sizeof(WCHAR));

	return isOK;
}

BOOL UnicodeToUtf8Ex(const WCHAR szSrc[], int iSrcLength, char szDest[], int& iDestLength)
{
	int iInBufLen = (int)(((iSrcLength > 0) ? iSrcLength : ((szSrc != nullptr) ? wcslen(szSrc) + 1 : 0)) * sizeof(WCHAR));

	return CharsetConvert(SYSTEM_CHARSET_UNICODE, CHARSET_UTF_8, (LPCSTR)szSrc, iInBufLen, szDest, iDestLength);
}

BOOL GbkToUtf8Ex(const char szSrc[], int iSrcLength, char szDest[], int& iDestLength)
{
	int iInBufLen = (int)((iSrcLength > 0) ? iSrcLength : ((szSrc != nullptr) ? strlen(szSrc) + 1 : 0));

	return CharsetConvert(CHARSET_GBK, CHARSET_UTF_8, szSrc, iInBufLen, szDest, iDestLength);
}

BOOL Utf8ToGbkEx(const char szSrc[], int iSrcLength, char szDest[], int& iDestLength)
{
	int iInBufLen = (int)((iSrcLength > 0) ? iSrcLength : ((szSrc != nullptr) ? strlen(szSrc) + 1 : 0));

	return CharsetConvert(CHARSET_UTF_8, CHARSET_GBK, szSrc, iInBufLen, szDest, iDestLength);
}

BOOL GbkToUnicode(const char szSrc[], WCHAR szDest[], int& iDestLength)
{
	return GbkToUnicodeEx(szSrc, -1, szDest, iDestLength);
}

BOOL UnicodeToGbk(const WCHAR szSrc[], char szDest[], int& iDestLength)
{
	return UnicodeToGbkEx(szSrc, -1, szDest, iDestLength);
}

BOOL Utf8ToUnicode(const char szSrc[], WCHAR szDest[], int& iDestLength)
{
	return Utf8ToUnicodeEx(szSrc, -1, szDest, iDestLength);
}

BOOL UnicodeToUtf8(const WCHAR szSrc[], char szDest[], int& iDestLength)
{
	return UnicodeToUtf8Ex(szSrc, -1, szDest, iDestLength);
}

BOOL GbkToUtf8(const char szSrc[], char szDest[], int& iDestLength)
{
	return GbkToUtf8Ex(szSrc, -1, szDest, iDestLength);
}

BOOL Utf8ToGbk(const char szSrc[], char szDest[], int& iDestLength)
{
	return Utf8ToGbkEx(szSrc, -1, szDest, iDestLength);
}

#endif
