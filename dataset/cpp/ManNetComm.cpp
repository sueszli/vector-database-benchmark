#include "StdAfx.h"
#include "ManNetComm.h"

enum {
	IDX_SOCK_ERR_NO_ERROR = 1,
	IDX_SOCK_ERR_SOCK_INIT,
	IDX_SOCK_ERR_CREATE,
	IDX_SOCK_ERR_LISTEN,
};

#define COMM_IDX_REQ_CMD_READY		0xffff

#define SZ_BUF_NET_REQ_PKT			(256 *1024)
#define SZ_BUF_NET_RES_PKT			(256 *1024)

#define TIME_SOCK_LISTEN_TIMEOUT	1000

UINT TH_NetCommProc (LPVOID pParam)
{
	CManNetComm *pMan = (CManNetComm *)pParam;

	AddReport ("[%2d] TH_NetCommMan_CCM: START\n", pMan->m_nDevID_CCM);

	int i, idxErr;
	DWORD timeStart;
	BOOL bNAck, bRcvTimeout;
	INFO_NET_COMM_SETTINGS iNCS;

	idxErr = IDX_SOCK_ERR_NO_ERROR;

/*	// Socket initialization[S]
	if (!AfxSocketInit ())
	{
		idxErr = IDX_SOCK_ERR_SOCK_INIT;
		goto END_OF_THREAD;
	}
	// Socket initialization[E]
*/
	pMan->m_pSockListen = new CAsyncSocket;
	pMan->m_pSockComm = new CAsyncSocket;
	pMan->m_pSockDummy = new CAsyncSocket;

	// Socket creation[S]
	if (pMan->m_pSockListen->Create (NUM_ICCS_NET_COMM_BASE_PORT +pMan->m_nDevID_CCM) == FALSE)
	{
		idxErr = IDX_SOCK_ERR_CREATE;
		goto END_OF_THREAD;
	}
	// Socket creation[E]

	// Socket listen[S]
	timeStart = ::timeGetTime ();
	while (1)
	{
		if (pMan->m_pSockListen->Listen () == TRUE)
		{
			break;
		}

		if (::timeGetTime () >= timeStart +TIME_SOCK_LISTEN_TIMEOUT)
		{
			idxErr = IDX_SOCK_ERR_LISTEN;
			goto END_OF_THREAD;
		}

		Sleep (10);
	}
	// Socket listen[E]

	pMan->m_bSockReady = TRUE;
	pMan->m_idxReqCmd = COMM_IDX_REQ_CMD_READY;

	while (pMan->m_bEndThread == FALSE)
	{
		if (pMan->m_bConnected == FALSE)
		{
			if (pMan->m_pSockListen->Accept (*pMan->m_pSockComm) == TRUE)
			{
				pMan->m_bConnected = TRUE;

				AddReport ("[%2d] CONNECTED\n", pMan->m_nDevID_CCM);
			}
			else
			{
				if (pMan->m_idxReqCmd != COMM_IDX_REQ_CMD_READY)
				{
					pMan->m_bCommOk = FALSE;

					::SetEvent (pMan->m_hEvtWaitNetComm);
					pMan->m_idxReqCmd = COMM_IDX_REQ_CMD_READY;
				}

				::Sleep (50);
			}
		}
		else
		{
			if (pMan->m_pSockListen->Accept (*pMan->m_pSockDummy) == TRUE)
			{
				pMan->m_pSockDummy->Close ();
				AddReport ("[%2d] Close more than 1 connection.\n", pMan->m_nDevID_CCM);
			}

			if (pMan->m_idxReqCmd != COMM_IDX_REQ_CMD_READY)
			{
				memcpy (&iNCS, &pMan->m_iNCS, sizeof(INFO_NET_COMM_SETTINGS));

				for (i=0; i<iNCS.numRetry; i++)
				{
					if (pMan->PktSnd () == FALSE ||
						pMan->PktRcv (iNCS.timeRcvTimeout, bNAck, bRcvTimeout) == FALSE)
					{
						AddReport ("[%2d] DISCONNECTED\n", pMan->m_nDevID_CCM);

						pMan->m_pSockComm->Close ();
						pMan->m_bConnected = FALSE;

						break;
					}

					if (bNAck == FALSE && bRcvTimeout == FALSE)
					{
						break;
					}
				}

				if (pMan->m_bConnected == FALSE || i == iNCS.numRetry)
				{
					pMan->m_bCommOk = FALSE;
				}
				else
				{
					pMan->m_bCommOk = TRUE;
				}

				::SetEvent (pMan->m_hEvtWaitNetComm);
				pMan->m_idxReqCmd = COMM_IDX_REQ_CMD_READY;
			}
			else
			{
				::Sleep (50);
			}
		}
	}
	pMan->m_pSockComm->Close ();

END_OF_THREAD:
	delete pMan->m_pSockListen;
	delete pMan->m_pSockComm;
	delete pMan->m_pSockDummy;

	pMan->m_bEndThread = FALSE;
	pMan->m_bSockReady = idxErr;

	AddReport ("[%2d] TH_NetCommMan_CCM(): END: idxErr: %d\n", pMan->m_nDevID_CCM, idxErr);

	return 0;
}

CManNetComm::CManNetComm(int nDevID_CCM, INFO_NET_COMM_SETTINGS *pINCS)
{
	m_nDevID_CCM = nDevID_CCM;
	memcpy (&m_iNCS, pINCS, sizeof(INFO_NET_COMM_SETTINGS));

	m_bufSnd = new BYTE[SZ_BUF_NET_REQ_PKT];
	m_bufRcv = new BYTE[SZ_BUF_NET_RES_PKT];
	m_hEvtWaitNetComm = ::CreateEvent (NULL, TRUE, FALSE, NULL);

	m_bEndThread = TRUE;

	memcpy (&m_bufSnd[0], GL_ICCS_BUF_STX, SZ_ICCS_PKT_ITEM_STX);
}

CManNetComm::~CManNetComm(void)
{
	DestroyNetCommManager ();

	delete[] m_bufSnd;
	delete[] m_bufRcv;
	::CloseHandle (m_hEvtWaitNetComm);
}

BOOL CManNetComm::InitNetCommManager ()
{
	if (m_bEndThread == TRUE)
	{
		m_bEndThread = FALSE;
		m_bSockReady = FALSE;
		m_bConnected = FALSE;

		AfxBeginThread (TH_NetCommProc, this);
		while (m_bSockReady == FALSE)
		{
			Sleep (10);
		}

		if (m_bSockReady != TRUE)
		{
			m_bEndThread = TRUE;
			return FALSE;
		}
	}

	return TRUE;
}

void CManNetComm::DestroyNetCommManager ()
{
	if (m_bEndThread == FALSE)
	{
		m_bEndThread = TRUE;
		while (m_bEndThread == TRUE)
		{
			Sleep (10);
		}
		m_bEndThread = TRUE;
		m_bSockReady = FALSE;
		m_bConnected = FALSE;
	}
}

BOOL CManNetComm::IsNetCommReady ()
{
	return m_bSockReady;
}

BOOL CManNetComm::IsClientConnected ()
{
	return m_bConnected;
}

BOOL CManNetComm::ProcessNetReq (INFO_IN_PROC_NET_REQ *pIIn, INFO_OUT_PROC_NET_REQ *pIOut)
{
#ifdef __USE_FAKE_CCM__
	pIOut->idxNC_Err = IDX_NC_ERR_NO_ERROR;
	pIOut->statSCM = FLAG_ICCS_SCM_SENSING_ON;

	if (pIIn->idxDT_Req == IDX_ICCS_DT_REQ_USM_LGM_STAT)
	{
		int i;
		pIOut->pIResM_USM_LGM_STAT->statSCM = FLAG_ICCS_SCM_SENSING_ON;
		for (i=0; i<MAX_NUM_USM; i++)
		{
			switch (rand () %6)
			{
			case 0: pIOut->pIResM_USM_LGM_STAT->bufUSM_Stat[i] = FLAG_ICCS_USM_USED |IDX_OPM_SENS_LED_OFF;			break;
			case 1: pIOut->pIResM_USM_LGM_STAT->bufUSM_Stat[i] = FLAG_ICCS_USM_USED |IDX_OPM_SENS_LED_ON_GREEN;		break;
			case 2: pIOut->pIResM_USM_LGM_STAT->bufUSM_Stat[i] = FLAG_ICCS_USM_USED |IDX_OPM_SENS_LED_ON_RED;		break;
			case 3: pIOut->pIResM_USM_LGM_STAT->bufUSM_Stat[i] = FLAG_ICCS_USM_USED |IDX_OPM_FORC_LED_ON_GREEN;		break;
			case 4: pIOut->pIResM_USM_LGM_STAT->bufUSM_Stat[i] = FLAG_ICCS_USM_USED |IDX_OPM_FORC_LED_ON_RED;		break;
			case 5: pIOut->pIResM_USM_LGM_STAT->bufUSM_Stat[i] = FLAG_ICCS_USM_USED |IDX_OPM_FORC_LED_BLINKING;		break;
			}
		}
		for (i=0; i<MAX_NUM_LGM; i++)
		{
			switch (rand () %3)
			{
			case 0: pIOut->pIResM_USM_LGM_STAT->bufLGM_Stat[i] = FLAG_ICCS_LGM_USED |IDX_OPM_FORC_LED_ON_GREEN;		break;
			case 1: pIOut->pIResM_USM_LGM_STAT->bufLGM_Stat[i] = FLAG_ICCS_LGM_USED |IDX_OPM_FORC_LED_ON_RED;		break;
			case 2: pIOut->pIResM_USM_LGM_STAT->bufLGM_Stat[i] = FLAG_ICCS_LGM_USED |IDX_OPM_FORC_LED_BLINKING;		break;
			}
		}
	}

	return TRUE;
#endif

	if (m_bSockReady == FALSE || m_bConnected == FALSE || m_idxReqCmd != COMM_IDX_REQ_CMD_READY)
	{
		return FALSE;
	}

	m_crtAccSock.Lock ();

	int nLength;
	WORD nCRC16;

	ITEM_HDR_REQM *pIHdrReqM = (ITEM_HDR_REQM *)&m_bufSnd[0];
	ITEM_TRL_REQM *pITrlReqM;
	
	switch (pIIn->idxDT_Req)
	{
	case IDX_ICCS_DT_REQ_USM_LGM_STAT:
		nLength = 3;
		pITrlReqM = (ITEM_TRL_REQM *)&m_bufSnd[sizeof(ITEM_HDR_REQM)];
		break;
	case IDX_ICCS_DT_REQ_SET_USM_ADDR:
		nLength = 3 +(pIIn->numItem *sizeof(ITEM_REQM_SET_ADDR));
		memcpy (&m_bufSnd[sizeof(ITEM_HDR_REQM)], pIIn->pIReqM_SET_ADDR, pIIn->numItem *sizeof(ITEM_REQM_SET_ADDR));
		pITrlReqM = (ITEM_TRL_REQM *)&m_bufSnd[sizeof(ITEM_HDR_REQM) +(pIIn->numItem *sizeof(ITEM_REQM_SET_ADDR))];
		break;
	case IDX_ICCS_DT_REQ_SET_LGM_ADDR:
		nLength = 3 +(pIIn->numItem *sizeof(ITEM_REQM_SET_ADDR));
		memcpy (&m_bufSnd[sizeof(ITEM_HDR_REQM)], pIIn->pIReqM_SET_ADDR, pIIn->numItem *sizeof(ITEM_REQM_SET_ADDR));
		pITrlReqM = (ITEM_TRL_REQM *)&m_bufSnd[sizeof(ITEM_HDR_REQM) +(pIIn->numItem *sizeof(ITEM_REQM_SET_ADDR))];
		break;
	case IDX_ICCS_DT_REQ_SET_USM_OP_MODE:
		nLength = 3 +(pIIn->numItem *sizeof(ITEM_REQM_SET_OP_MODE));
		memcpy (&m_bufSnd[sizeof(ITEM_HDR_REQM)], pIIn->pIReqM_SET_OP_MODE, pIIn->numItem *sizeof(ITEM_REQM_SET_OP_MODE));
		pITrlReqM = (ITEM_TRL_REQM *)&m_bufSnd[sizeof(ITEM_HDR_REQM) +(pIIn->numItem *sizeof(ITEM_REQM_SET_OP_MODE))];
		break;
	case IDX_ICCS_DT_REQ_SET_LGM_OP_MODE:
		nLength = 3 +(pIIn->numItem *sizeof(ITEM_REQM_SET_OP_MODE));
		memcpy (&m_bufSnd[sizeof(ITEM_HDR_REQM)], pIIn->pIReqM_SET_OP_MODE, pIIn->numItem *sizeof(ITEM_REQM_SET_OP_MODE));
		pITrlReqM = (ITEM_TRL_REQM *)&m_bufSnd[sizeof(ITEM_HDR_REQM) +(pIIn->numItem *sizeof(ITEM_REQM_SET_OP_MODE))];
		break;
	case IDX_ICCS_DT_REQ_SET_USM_PARAM:
		nLength = 3 +(pIIn->numItem *sizeof(ITEM_REQM_SET_PARAM));
		memcpy (&m_bufSnd[sizeof(ITEM_HDR_REQM)], pIIn->pIReqM_SET_PARAM, pIIn->numItem *sizeof(ITEM_REQM_SET_PARAM));
		pITrlReqM = (ITEM_TRL_REQM *)&m_bufSnd[sizeof(ITEM_HDR_REQM) +(pIIn->numItem *sizeof(ITEM_REQM_SET_PARAM))];
		break;
	case IDX_ICCS_DT_REQ_SET_USM_TO_LGM:
		nLength = 3 +(pIIn->numItem *sizeof(ITEM_REQM_SET_USM2LGM));
		memcpy (&m_bufSnd[sizeof(ITEM_HDR_REQM)], pIIn->pIReqM_SET_USM2LGM, pIIn->numItem *sizeof(ITEM_REQM_SET_USM2LGM));
		pITrlReqM = (ITEM_TRL_REQM *)&m_bufSnd[sizeof(ITEM_HDR_REQM) +(pIIn->numItem *sizeof(ITEM_REQM_SET_USM2LGM))];
		break;
	case IDX_ICCS_DT_REQ_SENSING_OFF:
		nLength = 3;
		pITrlReqM = (ITEM_TRL_REQM *)&m_bufSnd[sizeof(ITEM_HDR_REQM)];
		break;
	case IDX_ICCS_DT_REQ_SENSING_ON:
		nLength = 3;
		pITrlReqM = (ITEM_TRL_REQM *)&m_bufSnd[sizeof(ITEM_HDR_REQM)];
		break;
	case IDX_ICCS_DT_REQ_SET_EBD_PARAM:
		nLength = 3 +(pIIn->numItem *sizeof(ITEM_REQM_SET_PARAM));
		memcpy (&m_bufSnd[sizeof(ITEM_HDR_REQM)], pIIn->pIReqM_SET_PARAM, pIIn->numItem *sizeof(ITEM_REQM_SET_PARAM));
		pITrlReqM = (ITEM_TRL_REQM *)&m_bufSnd[sizeof(ITEM_HDR_REQM) +(pIIn->numItem *sizeof(ITEM_REQM_SET_PARAM))];
		break;
	}
	
	pIHdrReqM->nDataType = pIIn->idxDT_Req;
	pIHdrReqM->devID_SCM = pIIn->nDevID_SCM;
	pIHdrReqM->numItem = pIIn->numItem;
	pIHdrReqM->bufLength[0] = (nLength >>8) &0xff;
	pIHdrReqM->bufLength[1] = (nLength >>0) &0xff;
	
	nCRC16 = CRC16_GenCode (&m_bufSnd[SZ_ICCS_PKT_ITEM_STX], 2 +nLength);
	pITrlReqM->bufCRC[0] = (nCRC16 >>8) &0xff;
	pITrlReqM->bufCRC[1] = (nCRC16 >>0) &0xff;
	
	memcpy (pITrlReqM->bufETX, GL_ICCS_BUF_ETX, SZ_ICCS_PKT_ITEM_ETX);

	::ResetEvent (m_hEvtWaitNetComm);
	m_idxReqCmd = pIIn->idxDT_Req;
	::WaitForSingleObject (m_hEvtWaitNetComm, INFINITE);

	pIOut->idxNC_Err = m_idxLastNC_Err;
//	pIOut->statSCM = ((ITEM_HDR_REQM *)&m_bufRcv[m_posRcvStart])->numItem;
	

	if (m_bCommOk == TRUE && m_posRcvStart >= 0 && ((ITEM_HDR_REQM *)&m_bufRcv[m_posRcvStart])->nDataType == IDX_ICCS_DT_RES_USM_LGM_STAT)
	{
		memcpy (pIOut->pIResM_USM_LGM_STAT, &m_bufRcv[m_posRcvStart +sizeof(ITEM_HDR_REQM)], sizeof(ITEM_RESM_USM_LGM_STAT));
		pIOut->statSCM = pIOut->pIResM_USM_LGM_STAT->statSCM;

		if( ((ITEM_HDR_REQM *)&m_bufRcv[m_posRcvStart])->numItem !=0)
			glInfoGlobal.unGVA.iGVA.m_CentralMonitorGreenCnt =(int) ((ITEM_HDR_REQM *)&m_bufRcv[m_posRcvStart])->numItem;
	}

	m_crtAccSock.Unlock ();

	return m_bCommOk;
}

void CManNetComm::SetNetCommSettings (INFO_NET_COMM_SETTINGS *pINCS)
{
	m_crtAccSock.Lock ();

	memcpy (&m_iNCS, pINCS, sizeof(INFO_NET_COMM_SETTINGS));

	m_crtAccSock.Unlock ();
}

BOOL CManNetComm::PktSnd ()
{
	m_pSockComm->Receive (m_bufRcv, SZ_BUF_NET_RES_PKT);

	ITEM_HDR_REQM *pIHdrReqM = (ITEM_HDR_REQM *)&m_bufSnd[0];
	BOOL bErrSnd;
	int posCurSnd, numSnd, szBufSnd;

	posCurSnd = 0;
	szBufSnd = SZ_ICCS_PKT_HDR_ALL +((pIHdrReqM->bufLength[0] <<8) +pIHdrReqM->bufLength[1]);
	bErrSnd = FALSE;
	while (1)
	{
		numSnd = m_pSockComm->Send (&m_bufSnd[posCurSnd], szBufSnd);

		if (numSnd == SOCKET_ERROR)
		{
			int nErrCode = m_pSockComm->GetLastError ();

			if (nErrCode != WSAEWOULDBLOCK)
			{
				LogPrintSocketError (0, nErrCode);

				bErrSnd = TRUE;
				break;
			}
		}
		else
		{
			posCurSnd += numSnd;
			if (posCurSnd >= szBufSnd)
			{
				break;
			}
		}
	}
	if (bErrSnd == TRUE)
	{
		return FALSE;
	}

	return TRUE;
}

BOOL CManNetComm::PktRcv (DWORD timeRcvTimeout, BOOL &bNAck, BOOL &bRcvTimeout)
{
	m_idxLastNC_Err = IDX_NC_ERR_NO_ERROR;

	bNAck = FALSE;
	bRcvTimeout = FALSE;

	memset (m_bufRcv, 0, SZ_BUF_NET_RES_PKT);

	ITEM_HDR_REQM *pIHdrReqM_Snd = (ITEM_HDR_REQM *)&m_bufSnd[0];
	ITEM_HDR_REQM *pIHdrReqM_Rcv;

	DWORD timeStart;
	int i, numRcv, posCurRcv, nLength;
	WORD nCRC16;

	posCurRcv = 0;
	timeStart = ::timeGetTime ();
	Sleep (100);
	while (1)
	{
		numRcv = m_pSockComm->Receive (&m_bufRcv[posCurRcv], SZ_BUF_NET_RES_PKT -posCurRcv);

		if (numRcv == 0)	// The connection has been closed.
		{
			AddReport ("[%2d] RcvNetResPkt() : numRcv == 0\n", m_nDevID_CCM);
			m_idxLastNC_Err = IDX_NC_ERR_SOCK_CLOSED;
			return FALSE;
		}
		else if (numRcv == SOCKET_ERROR)
		{
			int nErrCode = m_pSockComm->GetLastError ();

			if (nErrCode != WSAEWOULDBLOCK)
			{
				LogPrintSocketError (1, nErrCode);
				m_idxLastNC_Err = IDX_NC_ERR_SOCK_ERR;
				AddReport ("SOCKERR\n");
				return FALSE;
			}
		}
		else
		{
			posCurRcv += numRcv;

			for (i=0; i<posCurRcv -SZ_ICCS_PKT_ITEM_STX -SZ_ICCS_PKT_ITEM_LENGTH; i++)
			{
				if (m_bufRcv[i +0] == GL_ICCS_BUF_STX[0] && m_bufRcv[i +1] == GL_ICCS_BUF_STX[1] &&
					m_bufRcv[i +2] == GL_ICCS_BUF_STX[2] && m_bufRcv[i +3] == GL_ICCS_BUF_STX[3] &&
					posCurRcv >= SZ_ICCS_PKT_HDR_ALL +((m_bufRcv[i +4] <<8) +m_bufRcv[i +5]))	// Packet detection
				{
					pIHdrReqM_Rcv = (ITEM_HDR_REQM *)&m_bufRcv[i];
					nLength = (m_bufRcv[i +4] <<8) +m_bufRcv[i +5];

					// Check CRC16[S]
					nCRC16 = CRC16_GenCode (&m_bufRcv[i +SZ_ICCS_PKT_ITEM_STX], SZ_ICCS_PKT_ITEM_LENGTH +nLength);
					if (m_bufRcv[i +SZ_ICCS_PKT_ITEM_STX +SZ_ICCS_PKT_ITEM_LENGTH +nLength +0] != ((nCRC16 >>8) &0xff) ||
						m_bufRcv[i +SZ_ICCS_PKT_ITEM_STX +SZ_ICCS_PKT_ITEM_LENGTH +nLength +1] != ((nCRC16 >>0) &0xff))
					{
						continue;
					}
					// Check CRC16[E]

					// Check ETX[S]
					if (m_bufRcv[i +SZ_ICCS_PKT_ITEM_STX +SZ_ICCS_PKT_ITEM_LENGTH +nLength +SZ_ICCS_PKT_ITEM_CRC16 +0] != GL_ICCS_BUF_ETX[0] ||
						m_bufRcv[i +SZ_ICCS_PKT_ITEM_STX +SZ_ICCS_PKT_ITEM_LENGTH +nLength +SZ_ICCS_PKT_ITEM_CRC16 +1] != GL_ICCS_BUF_ETX[1] ||
						m_bufRcv[i +SZ_ICCS_PKT_ITEM_STX +SZ_ICCS_PKT_ITEM_LENGTH +nLength +SZ_ICCS_PKT_ITEM_CRC16 +2] != GL_ICCS_BUF_ETX[2] ||
						m_bufRcv[i +SZ_ICCS_PKT_ITEM_STX +SZ_ICCS_PKT_ITEM_LENGTH +nLength +SZ_ICCS_PKT_ITEM_CRC16 +3] != GL_ICCS_BUF_ETX[3])
					{
						continue;
					}
					// Check ETX[E]

					// Valid packet processing[S]
					if (pIHdrReqM_Snd->nDataType == IDX_ICCS_DT_REQ_USM_LGM_STAT)
					{
						if (pIHdrReqM_Rcv->nDataType != IDX_ICCS_DT_RES_USM_LGM_STAT)
						{
							bNAck = TRUE;
							AddReport ("bNAck 1\n");
						}
					}
					else
					{
						if (pIHdrReqM_Rcv->nDataType != IDX_ICCS_DT_ACK)
						{
							bNAck = TRUE;
							AddReport ("bNAck 2, %02x\n", pIHdrReqM_Rcv->nDataType);
						}
					}

					m_posRcvStart = i;
					i = -1;
					break;
					// Valid packet processing[E]
				}
			}
			if (i == -1)
			{
				break;
			}
		}

		if (::timeGetTime () > timeStart +timeRcvTimeout)
		{
			if (posCurRcv > 0)
			{
				m_idxLastNC_Err = IDX_NC_ERR_RCV_INVALID;
			}
			else
			{
				m_idxLastNC_Err = IDX_NC_ERR_RCV_TIMEOUT;
			}

			bRcvTimeout = TRUE;
			AddReport ("TIMEOUT: %d\n", posCurRcv);	////////////
			break;
		}

		Sleep (100);
	}

	return TRUE;
}

void CManNetComm::LogPrintSocketError (int idxFunc, int idxErrCode)
{
	char strTmp[MAX_PATH];

	if (idxFunc == 0)
	{
		strcpy_s (strTmp, MAX_PATH, "PktSnd()");
	}
	else
	{
		strcpy_s (strTmp, MAX_PATH, "PktRcv()");
	}

	switch (idxErrCode)
	{
	case WSAENETDOWN:
		AddReport ("[%2d] %s : ERROR: WSAENETDOWN\n", m_nDevID_CCM, strTmp);
		break;
	case WSAENOTCONN:
		AddReport ("[%2d] %s : ERROR: WSAENOTCONN\n", m_nDevID_CCM, strTmp);
		break;
	case WSAEINPROGRESS:
		AddReport ("[%2d] %s : ERROR: WSAEINPROGRESS\n", m_nDevID_CCM, strTmp);
		break;
	case WSAESHUTDOWN:
		AddReport ("[%2d] %s : ERROR: WSAESHUTDOWN\n", m_nDevID_CCM, strTmp);
		break;
/*	case WSAEWOULDBLOCK:
		AddReport ("[%2d] %s : WSAEWOULDBLOCK\n", m_nDevID_CCM, strTmp);
		break;*/
	case WSAEMSGSIZE:
		AddReport ("[%2d] %s : ERROR: WSAEMSGSIZE\n", m_nDevID_CCM, strTmp);
		break;
	case WSAEINVAL:
		AddReport ("[%2d] %s : ERROR: WSAEINVAL\n", m_nDevID_CCM, strTmp);
		break;
	case WSAECONNABORTED:
		AddReport ("[%2d] %s : ERROR: WSAECONNABORTED\n", m_nDevID_CCM, strTmp);
		break;
	case WSAECONNRESET:
		AddReport ("[%2d] %s : ERROR: WSAECONNRESET\n", m_nDevID_CCM, strTmp);
		break;
	}
}
