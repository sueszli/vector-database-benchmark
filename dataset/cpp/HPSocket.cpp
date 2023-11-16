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
 
#include "../include/hpsocket/HPSocket.h"
#include "TcpServer.h"
#include "TcpAgent.h"
#include "TcpClient.h"
#include "TcpPullServer.h"
#include "TcpPullClient.h"
#include "TcpPullAgent.h"
#include "TcpPackServer.h"
#include "TcpPackClient.h"
#include "TcpPackAgent.h"
#include "HPThreadPool.h"

#ifdef _UDP_SUPPORT
#include "UdpServer.h"
#include "UdpClient.h"
#include "UdpCast.h"
#include "UdpNode.h"
#include "UdpArqServer.h"
#include "UdpArqClient.h"
#endif

#ifdef _HTTP_SUPPORT
#include "HttpServer.h"
#include "HttpAgent.h"
#include "HttpClient.h"
#endif

/*****************************************************************************************************************************************************/
/****************************************************************** TCP/UDP Exports ******************************************************************/
/*****************************************************************************************************************************************************/

HPSOCKET_API ITcpServer* HP_Create_TcpServer(ITcpServerListener* pListener)
{
	return new CTcpServer(pListener);
}

HPSOCKET_API ITcpAgent* HP_Create_TcpAgent(ITcpAgentListener* pListener)
{
	return new CTcpAgent(pListener);
}

HPSOCKET_API ITcpClient* HP_Create_TcpClient(ITcpClientListener* pListener)
{
	return new CTcpClient(pListener);
}

HPSOCKET_API ITcpPullServer* HP_Create_TcpPullServer(ITcpServerListener* pListener)
{
	return (ITcpPullServer*)(new CTcpPullServer(pListener));
}

HPSOCKET_API ITcpPullAgent* HP_Create_TcpPullAgent(ITcpAgentListener* pListener)
{
	return (ITcpPullAgent*)(new CTcpPullAgent(pListener));
}

HPSOCKET_API ITcpPullClient* HP_Create_TcpPullClient(ITcpClientListener* pListener)
{
	return (ITcpPullClient*)(new CTcpPullClient(pListener));
}

HPSOCKET_API ITcpPackServer* HP_Create_TcpPackServer(ITcpServerListener* pListener)
{
	return (ITcpPackServer*)(new CTcpPackServer(pListener));
}

HPSOCKET_API ITcpPackAgent* HP_Create_TcpPackAgent(ITcpAgentListener* pListener)
{
	return (ITcpPackAgent*)(new CTcpPackAgent(pListener));
}

HPSOCKET_API ITcpPackClient* HP_Create_TcpPackClient(ITcpClientListener* pListener)
{
	return (ITcpPackClient*)(new CTcpPackClient(pListener));
}

HPSOCKET_API void HP_Destroy_TcpServer(ITcpServer* pServer)
{
	delete pServer;
}

HPSOCKET_API void HP_Destroy_TcpAgent(ITcpAgent* pAgent)
{
	delete pAgent;
}

HPSOCKET_API void HP_Destroy_TcpClient(ITcpClient* pClient)
{
	delete pClient;
}

HPSOCKET_API void HP_Destroy_TcpPullServer(ITcpPullServer* pServer)
{
	delete pServer;
}

HPSOCKET_API void HP_Destroy_TcpPullAgent(ITcpPullAgent* pAgent)
{
	delete pAgent;
}

HPSOCKET_API void HP_Destroy_TcpPullClient(ITcpPullClient* pClient)
{
	delete pClient;
}

HPSOCKET_API void HP_Destroy_TcpPackServer(ITcpPackServer* pServer)
{
	delete pServer;
}

HPSOCKET_API void HP_Destroy_TcpPackAgent(ITcpPackAgent* pAgent)
{
	delete pAgent;
}

HPSOCKET_API void HP_Destroy_TcpPackClient(ITcpPackClient* pClient)
{
	delete pClient;
}

#ifdef _UDP_SUPPORT

HPSOCKET_API IUdpServer* HP_Create_UdpServer(IUdpServerListener* pListener)
{
	return new CUdpServer(pListener);
}

HPSOCKET_API IUdpClient* HP_Create_UdpClient(IUdpClientListener* pListener)
{
	return new CUdpClient(pListener);
}

HPSOCKET_API IUdpCast* HP_Create_UdpCast(IUdpCastListener* pListener)
{
	return new CUdpCast(pListener);
}

HPSOCKET_API IUdpNode* HP_Create_UdpNode(IUdpNodeListener* pListener)
{
	return new CUdpNode(pListener);
}

HPSOCKET_API IUdpArqServer* HP_Create_UdpArqServer(IUdpServerListener* pListener)
{
	return (IUdpArqServer*)(new CUdpArqServer(pListener));
}

HPSOCKET_API IUdpArqClient* HP_Create_UdpArqClient(IUdpClientListener* pListener)
{
	return (IUdpArqClient*)(new CUdpArqClient(pListener));
}

HPSOCKET_API void HP_Destroy_UdpServer(IUdpServer* pServer)
{
	delete pServer;
}

HPSOCKET_API void HP_Destroy_UdpClient(IUdpClient* pClient)
{
	delete pClient;
}

HPSOCKET_API void HP_Destroy_UdpCast(IUdpCast* pCast)
{
	delete pCast;
}

HPSOCKET_API void HP_Destroy_UdpNode(IUdpNode* pNode)
{
	delete pNode;
}

HPSOCKET_API void HP_Destroy_UdpArqServer(IUdpArqServer* pServer)
{
	delete pServer;
}

HPSOCKET_API void HP_Destroy_UdpArqClient(IUdpArqClient* pClient)
{
	delete pClient;
}

#endif

/*****************************************************************************************************************************************************/
/*************************************************************** Global Function Exports *************************************************************/
/*****************************************************************************************************************************************************/

HPSOCKET_API DWORD HP_GetHPSocketVersion()
{
	return ::GetHPSocketVersion();
}

HPSOCKET_API LPCTSTR HP_GetSocketErrorDesc(EnSocketError enCode)
{
	return ::GetSocketErrorDesc(enCode);
}

HPSOCKET_API DWORD SYS_GetLastError()
{
	return ::GetLastError();
}

HPSOCKET_API LPCSTR SYS_GetLastErrorStr()
{
	return ::GetLastErrorStr();
}

HPSOCKET_API int SYS_SetSocketOption(SOCKET sock, int level, int name, LPVOID val, int len)
{
	return ::SSO_SetSocketOption(sock, level, name, val, len);
}

HPSOCKET_API int SYS_GetSocketOption(SOCKET sock, int level, int name, LPVOID val, int* len)
{
	return ::SSO_GetSocketOption(sock, level, name, val, len);
}

HPSOCKET_API int SYS_IoctlSocket(SOCKET sock, long cmd, ULONG* arg)
{
	return ::SSO_IoctlSocket(sock, cmd, arg);
}

HPSOCKET_API BOOL SYS_fcntl_SETFL(FD fd, INT fl, BOOL bSet)
{
	return ::fcntl_SETFL(fd, fl, bSet);
}

HPSOCKET_API int SYS_SSO_NoBlock(SOCKET sock, BOOL bNoBlock)
{
	return ::SSO_NoBlock(sock, bNoBlock);
}

HPSOCKET_API int SYS_SSO_NoDelay(SOCKET sock, BOOL bNoDelay)
{
	return ::SSO_NoDelay(sock, bNoDelay);
}

HPSOCKET_API int SYS_SSO_DontLinger(SOCKET sock, BOOL bDont)
{
	return ::SSO_DontLinger(sock, bDont);
}

HPSOCKET_API int SYS_SSO_Linger(SOCKET sock, USHORT l_onoff, USHORT l_linger)
{
	return ::SSO_Linger(sock, l_onoff, l_linger);
}

HPSOCKET_API int SYS_SSO_RecvBuffSize(SOCKET sock, int size)
{
	return ::SSO_RecvBuffSize(sock, size);
}

HPSOCKET_API int SYS_SSO_SendBuffSize(SOCKET sock, int size)
{
	return ::SSO_SendBuffSize(sock, size);
}

HPSOCKET_API int SYS_SSO_RecvTimeOut(SOCKET sock, int ms)
{
	return ::SSO_RecvTimeOut(sock, ms);
}

HPSOCKET_API int SYS_SSO_SendTimeOut(SOCKET sock, int ms)
{
	return ::SSO_SendTimeOut(sock, ms);
}

HPSOCKET_API int SYS_SSO_ReuseAddress(SOCKET sock, EnReuseAddressPolicy opt)
{
	return ::SSO_ReuseAddress(sock, opt);
}

HPSOCKET_API BOOL SYS_GetSocketLocalAddress(SOCKET socket, TCHAR lpszAddress[], int& iAddressLen, USHORT& usPort)
{
	return ::GetSocketLocalAddress(socket, lpszAddress, iAddressLen, usPort);
}

HPSOCKET_API BOOL SYS_GetSocketRemoteAddress(SOCKET socket, TCHAR lpszAddress[], int& iAddressLen, USHORT& usPort)
{
	return ::GetSocketRemoteAddress(socket, lpszAddress, iAddressLen, usPort);
}

HPSOCKET_API BOOL SYS_EnumHostIPAddresses(LPCTSTR lpszHost, EnIPAddrType enType, LPTIPAddr** lpppIPAddr, int& iIPAddrCount)
{
	return ::EnumHostIPAddresses(lpszHost, enType, lpppIPAddr, iIPAddrCount);
}

HPSOCKET_API BOOL SYS_FreeHostIPAddresses(LPTIPAddr* lppIPAddr)
{
	return ::FreeHostIPAddresses(lppIPAddr);
}

HPSOCKET_API BOOL SYS_IsIPAddress(LPCTSTR lpszAddress, EnIPAddrType* penType)
{
	return ::IsIPAddress(lpszAddress, penType);
}

HPSOCKET_API BOOL SYS_GetIPAddress(LPCTSTR lpszHost, TCHAR lpszIP[], int& iIPLenth, EnIPAddrType& enType)
{
	return ::GetIPAddress(lpszHost, lpszIP, iIPLenth, enType);
}

HPSOCKET_API ULONGLONG SYS_NToH64(ULONGLONG value)
{
	return ::NToH64(value);
}

HPSOCKET_API ULONGLONG SYS_HToN64(ULONGLONG value)
{
	return ::HToN64(value);
}

HPSOCKET_API USHORT SYS_SwapEndian16(USHORT value)
{
	return ENDIAN_SWAP_16(value);
}

HPSOCKET_API DWORD SYS_SwapEndian32(DWORD value)
{
	return ENDIAN_SWAP_32(value);
}

HPSOCKET_API BOOL SYS_IsLittleEndian()
{
	return ::IsLittleEndian();
}

HPSOCKET_API LPBYTE SYS_Malloc(int size)
{
	return MALLOC(BYTE, size);
}

HPSOCKET_API LPBYTE SYS_Realloc(LPBYTE p, int size)
{
	return REALLOC(BYTE, p, size);
}

HPSOCKET_API VOID SYS_Free(LPBYTE p)
{
	FREE(p);
}

HPSOCKET_API LPVOID SYS_Calloc(int number, int size)
{
	return CALLOC(number, size);
}

HPSOCKET_API DWORD SYS_GuessBase64EncodeBound(DWORD dwSrcLen)
{
	return ::GuessBase64EncodeBound(dwSrcLen);
}

HPSOCKET_API DWORD SYS_GuessBase64DecodeBound(const BYTE* lpszSrc, DWORD dwSrcLen)
{
	return ::GuessBase64DecodeBound(lpszSrc, dwSrcLen);
}

HPSOCKET_API int SYS_Base64Encode(const BYTE* lpszSrc, DWORD dwSrcLen, BYTE* lpszDest, DWORD& dwDestLen)
{
	return ::Base64Encode(lpszSrc, dwSrcLen, lpszDest, dwDestLen);
}

HPSOCKET_API int SYS_Base64Decode(const BYTE* lpszSrc, DWORD dwSrcLen, BYTE* lpszDest, DWORD& dwDestLen)
{
	return ::Base64Decode(lpszSrc, dwSrcLen, lpszDest, dwDestLen);
}

HPSOCKET_API DWORD SYS_GuessUrlEncodeBound(const BYTE* lpszSrc, DWORD dwSrcLen)
{
	return ::GuessUrlEncodeBound(lpszSrc, dwSrcLen);
}

HPSOCKET_API DWORD SYS_GuessUrlDecodeBound(const BYTE* lpszSrc, DWORD dwSrcLen)
{
	return ::GuessUrlDecodeBound(lpszSrc, dwSrcLen);
}

HPSOCKET_API int SYS_UrlEncode(BYTE* lpszSrc, DWORD dwSrcLen, BYTE* lpszDest, DWORD& dwDestLen)
{
	return ::UrlEncode(lpszSrc, dwSrcLen, lpszDest, dwDestLen);
}

HPSOCKET_API int SYS_UrlDecode(BYTE* lpszSrc, DWORD dwSrcLen, BYTE* lpszDest, DWORD& dwDestLen)
{
	return ::UrlDecode(lpszSrc, dwSrcLen, lpszDest, dwDestLen);
}

#ifdef _ZLIB_SUPPORT

HPSOCKET_API int SYS_Compress(const BYTE* lpszSrc, DWORD dwSrcLen, BYTE* lpszDest, DWORD& dwDestLen)
{
	return ::Compress(lpszSrc, dwSrcLen, lpszDest, dwDestLen);
}

HPSOCKET_API int SYS_CompressEx(const BYTE* lpszSrc, DWORD dwSrcLen, BYTE* lpszDest, DWORD& dwDestLen, int iLevel, int iMethod, int iWindowBits, int iMemLevel, int iStrategy)
{
	return ::CompressEx(lpszSrc, dwSrcLen, lpszDest, dwDestLen, iLevel, iMethod, iWindowBits, iMemLevel, iStrategy);
}

HPSOCKET_API int SYS_Uncompress(const BYTE* lpszSrc, DWORD dwSrcLen, BYTE* lpszDest, DWORD& dwDestLen)
{
	return ::Uncompress(lpszSrc, dwSrcLen, lpszDest, dwDestLen);
}

HPSOCKET_API int SYS_UncompressEx(const BYTE* lpszSrc, DWORD dwSrcLen, BYTE* lpszDest, DWORD& dwDestLen, int iWindowBits)
{
	return ::UncompressEx(lpszSrc, dwSrcLen, lpszDest, dwDestLen, iWindowBits);
}

HPSOCKET_API DWORD SYS_GuessCompressBound(DWORD dwSrcLen, BOOL bGZip)
{
	return ::GuessCompressBound(dwSrcLen, bGZip);
}

HPSOCKET_API int SYS_GZipCompress(const BYTE* lpszSrc, DWORD dwSrcLen, BYTE* lpszDest, DWORD& dwDestLen)
{
	return ::GZipCompress(lpszSrc, dwSrcLen, lpszDest, dwDestLen);
}

HPSOCKET_API int SYS_GZipUncompress(const BYTE* lpszSrc, DWORD dwSrcLen, BYTE* lpszDest, DWORD& dwDestLen)
{
	return ::GZipUncompress(lpszSrc, dwSrcLen, lpszDest, dwDestLen);
}

HPSOCKET_API DWORD SYS_GZipGuessUncompressBound(const BYTE* lpszSrc, DWORD dwSrcLen)
{
	return ::GZipGuessUncompressBound(lpszSrc, dwSrcLen);
}

#endif

#ifdef _BROTLI_SUPPORT

HPSOCKET_API int SYS_BrotliCompress(const BYTE* lpszSrc, DWORD dwSrcLen, BYTE* lpszDest, DWORD& dwDestLen)
{
	return ::BrotliCompress(lpszSrc, dwSrcLen, lpszDest, dwDestLen);
}

HPSOCKET_API int SYS_BrotliCompressEx(const BYTE* lpszSrc, DWORD dwSrcLen, BYTE* lpszDest, DWORD& dwDestLen, int iQuality, int iWindow, int iMode)
{
	return ::BrotliCompressEx(lpszSrc, dwSrcLen, lpszDest, dwDestLen, iQuality, iWindow, (BrotliEncoderMode)iMode);
}

HPSOCKET_API int SYS_BrotliUncompress(const BYTE* lpszSrc, DWORD dwSrcLen, BYTE* lpszDest, DWORD& dwDestLen)
{
	return ::BrotliUncompress(lpszSrc, dwSrcLen, lpszDest, dwDestLen);
}

HPSOCKET_API DWORD SYS_BrotliGuessCompressBound(DWORD dwSrcLen)
{
	return ::BrotliGuessCompressBound(dwSrcLen);
}

#endif

#ifdef _ICONV_SUPPORT

HPSOCKET_API BOOL SYS_CharsetConvert(LPCSTR lpszFromCharset, LPCSTR lpszToCharset, LPCSTR lpszInBuf, int iInBufLen, LPSTR lpszOutBuf, int& iOutBufLen)
{
	return ::CharsetConvert(lpszFromCharset, lpszToCharset, lpszInBuf, iInBufLen, lpszOutBuf, iOutBufLen);
}

HPSOCKET_API BOOL SYS_GbkToUnicodeEx(const char szSrc[], int iSrcLength, WCHAR szDest[], int& iDestLength)
{
	return ::GbkToUnicodeEx(szSrc, iSrcLength, szDest, iDestLength);
}

HPSOCKET_API BOOL SYS_UnicodeToGbkEx(const WCHAR szSrc[], int iSrcLength, char szDest[], int& iDestLength)
{
	return ::UnicodeToGbkEx(szSrc, iSrcLength, szDest, iDestLength);
}

HPSOCKET_API BOOL SYS_Utf8ToUnicodeEx(const char szSrc[], int iSrcLength, WCHAR szDest[], int& iDestLength)
{
	return ::Utf8ToUnicodeEx(szSrc, iSrcLength, szDest, iDestLength);
}

HPSOCKET_API BOOL SYS_UnicodeToUtf8Ex(const WCHAR szSrc[], int iSrcLength, char szDest[], int& iDestLength)
{
	return ::UnicodeToUtf8Ex(szSrc, iSrcLength, szDest, iDestLength);
}

HPSOCKET_API BOOL SYS_GbkToUtf8Ex(const char szSrc[], int iSrcLength, char szDest[], int& iDestLength)
{
	return ::GbkToUtf8Ex(szSrc, iSrcLength, szDest, iDestLength);
}

HPSOCKET_API BOOL SYS_Utf8ToGbkEx(const char szSrc[], int iSrcLength, char szDest[], int& iDestLength)
{
	return ::Utf8ToGbkEx(szSrc, iSrcLength, szDest, iDestLength);
}

HPSOCKET_API BOOL SYS_GbkToUnicode(const char szSrc[], WCHAR szDest[], int& iDestLength)
{
	return ::GbkToUnicode(szSrc, szDest, iDestLength);
}

HPSOCKET_API BOOL SYS_UnicodeToGbk(const WCHAR szSrc[], char szDest[], int& iDestLength)
{
	return ::UnicodeToGbk(szSrc, szDest, iDestLength);
}

HPSOCKET_API BOOL SYS_Utf8ToUnicode(const char szSrc[], WCHAR szDest[], int& iDestLength)
{
	return ::Utf8ToUnicode(szSrc, szDest, iDestLength);
}

HPSOCKET_API BOOL SYS_UnicodeToUtf8(const WCHAR szSrc[], char szDest[], int& iDestLength)
{
	return ::UnicodeToUtf8(szSrc, szDest, iDestLength);
}

HPSOCKET_API BOOL SYS_GbkToUtf8(const char szSrc[], char szDest[], int& iDestLength)
{
	return ::GbkToUtf8(szSrc, szDest, iDestLength);
}

HPSOCKET_API BOOL SYS_Utf8ToGbk(const char szSrc[], char szDest[], int& iDestLength)
{
	return ::Utf8ToGbk(szSrc, szDest, iDestLength);
}

#endif

/*****************************************************************************************************************************************************/
/******************************************************************** HTTP Exports *******************************************************************/
/*****************************************************************************************************************************************************/

#ifdef _HTTP_SUPPORT

HPSOCKET_API IHttpServer* HP_Create_HttpServer(IHttpServerListener* pListener)
{
	return (IHttpServer*)(new CHttpServer(pListener));
}

HPSOCKET_API IHttpAgent* HP_Create_HttpAgent(IHttpAgentListener* pListener)
{
	return (IHttpAgent*)(new CHttpAgent(pListener));
}

HPSOCKET_API IHttpClient* HP_Create_HttpClient(IHttpClientListener* pListener)
{
	return (IHttpClient*)(new CHttpClient(pListener));
}

HPSOCKET_API IHttpSyncClient* HP_Create_HttpSyncClient(IHttpClientListener* pListener)
{
	return (IHttpSyncClient*)(new CHttpSyncClient(pListener));
}

HPSOCKET_API void HP_Destroy_HttpServer(IHttpServer* pServer)
{
	delete pServer;
}

HPSOCKET_API void HP_Destroy_HttpAgent(IHttpAgent* pAgent)
{
	delete pAgent;
}

HPSOCKET_API void HP_Destroy_HttpClient(IHttpClient* pClient)
{
	delete pClient;
}

HPSOCKET_API void HP_Destroy_HttpSyncClient(IHttpSyncClient* pClient)
{
	delete pClient;
}

/**************************************************************************/
/*************************** HTTP Cookie 管理方法 **************************/

HPSOCKET_API BOOL HP_HttpCookie_MGR_LoadFromFile(LPCSTR lpszFile, BOOL bKeepExists)
{
	return g_CookieMgr.LoadFromFile(lpszFile, bKeepExists);
}

HPSOCKET_API BOOL HP_HttpCookie_MGR_SaveToFile(LPCSTR lpszFile, BOOL bKeepExists)
{
	return g_CookieMgr.SaveToFile(lpszFile, bKeepExists);
}

HPSOCKET_API BOOL HP_HttpCookie_MGR_ClearCookies(LPCSTR lpszDomain, LPCSTR lpszPath)
{
	return g_CookieMgr.ClearCookies(lpszDomain, lpszPath);
}

HPSOCKET_API BOOL HP_HttpCookie_MGR_RemoveExpiredCookies(LPCSTR lpszDomain, LPCSTR lpszPath)
{
	return g_CookieMgr.RemoveExpiredCookies(lpszDomain, lpszPath);
}

HPSOCKET_API BOOL HP_HttpCookie_MGR_SetCookie(LPCSTR lpszName, LPCSTR lpszValue, LPCSTR lpszDomain, LPCSTR lpszPath, int iMaxAge, BOOL bHttpOnly, BOOL bSecure, int enSameSite, BOOL bOnlyUpdateValueIfExists)
{
	return g_CookieMgr.SetCookie(lpszName, lpszValue, lpszDomain, lpszPath, iMaxAge, bHttpOnly, bSecure, (CCookie::EnSameSite)enSameSite, bOnlyUpdateValueIfExists);
}

HPSOCKET_API BOOL HP_HttpCookie_MGR_DeleteCookie(LPCSTR lpszDomain, LPCSTR lpszPath, LPCSTR lpszName)
{
	return g_CookieMgr.DeleteCookie(lpszDomain, lpszPath, lpszName);
}

HPSOCKET_API void HP_HttpCookie_MGR_SetEnableThirdPartyCookie(BOOL bEnableThirdPartyCookie)
{
	g_CookieMgr.SetEnableThirdPartyCookie(bEnableThirdPartyCookie);
}

HPSOCKET_API BOOL HP_HttpCookie_MGR_IsEnableThirdPartyCookie()
{
	return g_CookieMgr.IsEnableThirdPartyCookie();
}

HPSOCKET_API BOOL HP_HttpCookie_HLP_ParseExpires(LPCSTR lpszExpires, __time64_t& tmExpires)
{
	return CCookie::ParseExpires(lpszExpires, tmExpires);
}

HPSOCKET_API BOOL HP_HttpCookie_HLP_MakeExpiresStr(char lpszBuff[], int& iBuffLen, __time64_t tmExpires)
{
	return CCookie::MakeExpiresStr(lpszBuff, iBuffLen, tmExpires);
}

HPSOCKET_API BOOL HP_HttpCookie_HLP_ToString(char lpszBuff[], int& iBuffLen, LPCSTR lpszName, LPCSTR lpszValue, LPCSTR lpszDomain, LPCSTR lpszPath, int iMaxAge, BOOL bHttpOnly, BOOL bSecure, int enSameSite)
{
	return CCookie::ToString(lpszBuff, iBuffLen, lpszName, lpszValue, lpszDomain, lpszPath, iMaxAge, bHttpOnly, bSecure, (CCookie::EnSameSite)enSameSite);
}

HPSOCKET_API __time64_t HP_HttpCookie_HLP_CurrentUTCTime()
{
	return CCookie::CurrentUTCTime();
}

HPSOCKET_API __time64_t HP_HttpCookie_HLP_MaxAgeToExpires(int iMaxAge)
{
	return CCookie::MaxAgeToExpires(iMaxAge);
}

HPSOCKET_API int HP_HttpCookie_HLP_ExpiresToMaxAge(__time64_t tmExpires)
{
	return CCookie::ExpiresToMaxAge(tmExpires);
}

/*****************************************************************************************************************************************************/
/************************************************************ HTTP Global Function Exports ***********************************************************/
/*****************************************************************************************************************************************************/

#endif

/*****************************************************************************************************************************************************/
/**************************************************************** Thread Pool Exports ****************************************************************/
/*****************************************************************************************************************************************************/

HPSOCKET_API IHPThreadPool* HP_Create_ThreadPool(IHPThreadPoolListener* pListener)
{
	return new CHPThreadPool(pListener);
}

HPSOCKET_API void HP_Destroy_ThreadPool(IHPThreadPool* pThreadPool)
{
	delete pThreadPool;
}

HPSOCKET_API LPTSocketTask HP_Create_SocketTaskObj(Fn_SocketTaskProc fnTaskProc, PVOID pSender, CONNID dwConnID, LPCBYTE pBuffer, INT iBuffLen, EnTaskBufferType enBuffType, WPARAM wParam, LPARAM lParam)
{
	return ::CreateSocketTaskObj(fnTaskProc, pSender, dwConnID, pBuffer, iBuffLen, enBuffType, wParam, lParam);
}

HPSOCKET_API void HP_Destroy_SocketTaskObj(LPTSocketTask pTask)
{
	::DestroySocketTaskObj(pTask);
}

/*****************************************************************************************************************************************************/
/********************************************************* Compressor / Decompressor Exports *********************************************************/
/*****************************************************************************************************************************************************/

HPSOCKET_API void HP_Destroy_Compressor(IHPCompressor* pCompressor)
{
	::DestroyCompressor(pCompressor);
}

HPSOCKET_API void HP_Destroy_Decompressor(IHPDecompressor* pDecompressor)
{
	::DestroyDecompressor(pDecompressor);
}

#ifdef _ZLIB_SUPPORT

HPSOCKET_API IHPCompressor* HP_Create_ZLibCompressor(Fn_CompressDataCallback fnCallback, int iWindowBits, int iLevel, int iMethod, int iMemLevel, int iStrategy, DWORD dwBuffSize)
{
	return ::CreateZLibCompressor(fnCallback, iWindowBits, iLevel, iMethod, iMemLevel, iStrategy, dwBuffSize);
}

HPSOCKET_API IHPCompressor* HP_Create_GZipCompressor(Fn_CompressDataCallback fnCallback, int iLevel, int iMethod, int iMemLevel, int iStrategy, DWORD dwBuffSize)
{
	return ::CreateGZipCompressor(fnCallback, iLevel, iMethod, iMemLevel, iStrategy, dwBuffSize);
}

HPSOCKET_API IHPDecompressor* HP_Create_ZLibDecompressor(Fn_DecompressDataCallback fnCallback, int iWindowBits, DWORD dwBuffSize)
{
	return ::CreateZLibDecompressor(fnCallback, iWindowBits, dwBuffSize);
}

HPSOCKET_API IHPDecompressor* HP_Create_GZipDecompressor(Fn_DecompressDataCallback fnCallback, DWORD dwBuffSize)
{
	return ::CreateGZipDecompressor(fnCallback, dwBuffSize);
}

#endif

#ifdef _BROTLI_SUPPORT

HPSOCKET_API IHPCompressor* HP_Create_BrotliCompressor(Fn_CompressDataCallback fnCallback, int iQuality, int iWindow, int iMode, DWORD dwBuffSize)
{
	return ::CreateBrotliCompressor(fnCallback, iQuality, iWindow, iMode, dwBuffSize);
}

HPSOCKET_API IHPDecompressor* HP_Create_BrotliDecompressor(Fn_DecompressDataCallback fnCallback, DWORD dwBuffSize)
{
	return ::CreateBrotliDecompressor(fnCallback, dwBuffSize);
}

#endif
