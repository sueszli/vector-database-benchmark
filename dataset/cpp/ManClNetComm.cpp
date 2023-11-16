// ManClNetComm.cpp : 구현 파일입니다.
//

#include "stdafx.h"
#include "ManClNetComm.h"

CManClNetComm::CManClNetComm()
{
	m_pSockClient = NULL;
	m_bConnected = FALSE;
}

CManClNetComm::~CManClNetComm()
{
}

BOOL CManClNetComm::Connect (LPCTSTR strServAddr, int nPort)
{
	if (m_pSockClient != NULL)
	{
		Close ();
	}

	m_pSockClient = new CAsyncSocket;
	m_pSockClient->Create ();

	DWORD idxErr, timeConnStart;

	timeConnStart = ::timeGetTime ();
	while (m_pSockClient->Connect (strServAddr, nPort) == FALSE)
	{
		idxErr = GetLastError ();

		if (idxErr == WSAEISCONN)
		{
			break;
		}
		else if (idxErr == WSAEWOULDBLOCK)
		{
			::Sleep (50);
		}
		else
		{
			Close ();
			return FALSE;
		}
	}

	m_bConnected = TRUE;
	return TRUE;
}

void CManClNetComm::Close ()
{
	m_bConnected = FALSE;

	if (m_pSockClient != NULL)
	{
		delete m_pSockClient;
	}
	m_pSockClient = NULL;
}

int CManClNetComm::ReceiveData (BYTE *bufData, DWORD szData)
{
	if (m_pSockClient == NULL)
	{
		return -1;
	}

	int numRcv;

	numRcv = m_pSockClient->Receive (bufData, szData);

	if (numRcv == SOCKET_ERROR)
	{
		DWORD idxErr = GetLastError ();
		LPVOID lpMsgBuf;

		switch (idxErr)
		{
		case WSANOTINITIALISED:
		case WSAENETDOWN:
		case WSAENOTCONN:
		case WSAEINPROGRESS:
		case WSAENOTSOCK:
		case WSAEOPNOTSUPP:
		case WSAESHUTDOWN:
		case WSAEMSGSIZE:
		case WSAEINVAL:
		case WSAECONNABORTED:
		case WSAECONNRESET:
			FormatMessage (FORMAT_MESSAGE_ALLOCATE_BUFFER |FORMAT_MESSAGE_FROM_SYSTEM |FORMAT_MESSAGE_IGNORE_INSERTS,
				NULL, idxErr, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), (LPTSTR)&lpMsgBuf, 0, NULL);
//			AddReport ("%s", (LPTSTR)lpMsgBuf);
			LocalFree (lpMsgBuf);
			Close ();
			break;
		case WSAEWOULDBLOCK:
			break;
		}
	}

	return numRcv;
}

int CManClNetComm::SendData (BYTE *bufData, DWORD szData)
{
	if (m_pSockClient == NULL)
	{
		return -1;
	}

	int numSend;

	numSend = m_pSockClient->Send (bufData, szData);

	if (numSend == SOCKET_ERROR)
	{
		DWORD idxErr = GetLastError ();
		LPVOID lpMsgBuf;

		switch (idxErr)
		{
		case WSANOTINITIALISED:
		case WSAENETDOWN:
		case WSAENOTCONN:
		case WSAEINPROGRESS:
		case WSAENOTSOCK:
		case WSAEOPNOTSUPP:
		case WSAESHUTDOWN:
		case WSAEMSGSIZE:
		case WSAEINVAL:
		case WSAECONNABORTED:
		case WSAECONNRESET:
			FormatMessage (FORMAT_MESSAGE_ALLOCATE_BUFFER |FORMAT_MESSAGE_FROM_SYSTEM |FORMAT_MESSAGE_IGNORE_INSERTS,
				NULL, idxErr, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), (LPTSTR)&lpMsgBuf, 0, NULL);
			//			AddReport ("%s", (LPTSTR)lpMsgBuf);
			LocalFree (lpMsgBuf);
			Close ();
			break;
		case WSAEWOULDBLOCK:
			break;
		}
	}

	return numSend;
}
