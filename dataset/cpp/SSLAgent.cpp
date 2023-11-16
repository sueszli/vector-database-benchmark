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
 
#include "SSLAgent.h"
#include "SSLHelper.h"

#ifdef _SSL_SUPPORT

BOOL CSSLAgent::CheckParams()
{
	if(!m_sslCtx.IsValid())
	{
		SetLastError(SE_SSL_ENV_NOT_READY, __FUNCTION__, ERROR_NOT_READY);
		return FALSE;
	}

	return __super::CheckParams();
}

void CSSLAgent::PrepareStart()
{
	__super::PrepareStart();

	m_sslPool.SetItemCapacity	(GetSocketBufferSize());
	m_sslPool.SetItemPoolSize	(GetFreeBufferObjPool());
	m_sslPool.SetItemPoolHold	(GetFreeBufferObjHold());
	m_sslPool.SetSessionLockTime(GetFreeSocketObjLockTime());
	m_sslPool.SetSessionPoolSize(GetFreeSocketObjPool());
	m_sslPool.SetSessionPoolHold(GetFreeSocketObjHold());

	m_sslPool.Prepare();
}

void CSSLAgent::Reset()
{
	m_sslPool.Clear();
	m_sslCtx.RemoveThreadLocalState();

	__super::Reset();
}

void CSSLAgent::OnWorkerThreadEnd(THR_ID dwThreadID)
{
	m_sslCtx.RemoveThreadLocalState();

	__super::OnWorkerThreadEnd(dwThreadID);
}

BOOL CSSLAgent::SendPackets(CONNID dwConnID, const WSABUF pBuffers[], int iCount)
{
	ASSERT(pBuffers && iCount > 0);

	TAgentSocketObj* pSocketObj = FindSocketObj(dwConnID);

	if(!TAgentSocketObj::IsValid(pSocketObj))
	{
		::SetLastError(ERROR_OBJECT_NOT_FOUND);
		return FALSE;
	}

	CSSLSession* pSession = nullptr;
	GetConnectionReserved2(pSocketObj, (PVOID*)&pSession);

	if(pSession != nullptr)
	{
		CLocalSafeCounter localcounter(*pSession);
		return ::ProcessSend(this, pSocketObj, pSession, pBuffers, iCount);
	}

	return DoSendPackets(pSocketObj, pBuffers, iCount);
}

EnHandleResult CSSLAgent::FireConnect(TAgentSocketObj* pSocketObj)
{
	EnHandleResult result = DoFireConnect(pSocketObj);

	if(result != HR_ERROR && m_bSSLAutoHandShake)
		DoSSLHandShake(pSocketObj);

	return result;
}

EnHandleResult CSSLAgent::FireReceive(TAgentSocketObj* pSocketObj, const BYTE* pData, int iLength)
{
	CSSLSession* pSession = nullptr;
	GetConnectionReserved2(pSocketObj, (PVOID*)&pSession);

	if(pSession != nullptr)
	{
		CLocalSafeCounter localcounter(*pSession);
		return ::ProcessReceive(this, pSocketObj, pSession, pData, iLength);
	}

	return DoFireReceive(pSocketObj, pData, iLength);
}

EnHandleResult CSSLAgent::FireClose(TAgentSocketObj* pSocketObj, EnSocketOperation enOperation, int iErrorCode)
{
	EnHandleResult result = DoFireClose(pSocketObj, enOperation, iErrorCode);

	CSSLSession* pSession = nullptr;
	GetConnectionReserved2(pSocketObj, (PVOID*)&pSession);

	if(pSession != nullptr)
		m_sslPool.PutFreeSession(pSession);

	return result;
}

BOOL CSSLAgent::StartSSLHandShake(CONNID dwConnID)
{
	if(IsSSLAutoHandShake())
	{
		::SetLastError(ERROR_INVALID_OPERATION);
		return FALSE;
	}

	TAgentSocketObj* pSocketObj = FindSocketObj(dwConnID);

	if(!TAgentSocketObj::IsValid(pSocketObj))
	{
		::SetLastError(ERROR_OBJECT_NOT_FOUND);
		return FALSE;
	}

	return StartSSLHandShake(pSocketObj);
}

BOOL CSSLAgent::StartSSLHandShake(TAgentSocketObj* pSocketObj)
{
	if(!pSocketObj->HasConnected())
	{
		::SetLastError(ERROR_INVALID_STATE);
		return FALSE;
	}

	CReentrantCriSecLock locallock(pSocketObj->csSend);

	if(!TAgentSocketObj::IsValid(pSocketObj))
	{
		::SetLastError(ERROR_OBJECT_NOT_FOUND);
		return FALSE;
	}

	if(!pSocketObj->HasConnected())
	{
		::SetLastError(ERROR_INVALID_STATE);
		return FALSE;
	}

	CSSLSession* pSession = nullptr;
	GetConnectionReserved2(pSocketObj, (PVOID*)&pSession);

	if(pSession != nullptr)
	{
		::SetLastError(ERROR_ALREADY_INITIALIZED);
		return FALSE;
	}

	DoSSLHandShake(pSocketObj);

	return TRUE;
}

void CSSLAgent::DoSSLHandShake(TAgentSocketObj* pSocketObj)
{
	CSSLSession* pSession = m_sslPool.PickFreeSession(pSocketObj->host);

	CLocalSafeCounter localcounter(*pSession);

	ENSURE(SetConnectionReserved2(pSocketObj, pSession));
	ENSURE(::ProcessHandShake(this, pSocketObj, pSession) == HR_OK);
}

BOOL CSSLAgent::GetSSLSessionInfo(CONNID dwConnID, EnSSLSessionInfo enInfo, LPVOID* lppInfo)
{
	ASSERT(lppInfo != nullptr);

	*lppInfo					= nullptr;
	TAgentSocketObj* pSocketObj	= FindSocketObj(dwConnID);

	if(!TAgentSocketObj::IsValid(pSocketObj))
	{
		::SetLastError(ERROR_OBJECT_NOT_FOUND);
		return FALSE;
	}

	CSSLSession* pSession = nullptr;
	GetConnectionReserved2(pSocketObj, (PVOID*)&pSession);

	if(pSession == nullptr || !pSession->IsValid())
	{
		::SetLastError(ERROR_INVALID_STATE);
		return FALSE;
	}

	return pSession->GetSessionInfo(enInfo, lppInfo);
}

#endif