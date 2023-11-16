#include "Iop_Thsema.h"
#include "IopBios.h"
#include "../Log.h"

#define LOG_NAME ("iop_thsema")

using namespace Iop;

#define FUNCTION_CREATESEMAPHORE "CreateSemaphore"
#define FUNCTION_DELETESEMAPHORE "DeleteSemaphore"
#define FUNCTION_SIGNALSEMAPHORE "SignalSemaphore"
#define FUNCTION_ISIGNALSEMAPHORE "iSignalSemaphore"
#define FUNCTION_WAITSEMAPHORE "WaitSemaphore"
#define FUNCTION_POLLSEMAPHORE "PollSemaphore"
#define FUNCTION_REFERSEMASTATUS "ReferSemaStatus"
#define FUNCTION_IREFERSEMASTATUS "iReferSemaStatus"

CThsema::CThsema(CIopBios& bios, uint8* ram)
    : m_bios(bios)
    , m_ram(ram)
{
}

std::string CThsema::GetId() const
{
	return "thsemap";
}

std::string CThsema::GetFunctionName(unsigned int functionId) const
{
	switch(functionId)
	{
	case 4:
		return FUNCTION_CREATESEMAPHORE;
		break;
	case 5:
		return FUNCTION_DELETESEMAPHORE;
		break;
	case 6:
		return FUNCTION_SIGNALSEMAPHORE;
		break;
	case 7:
		return FUNCTION_ISIGNALSEMAPHORE;
		break;
	case 8:
		return FUNCTION_WAITSEMAPHORE;
		break;
	case 9:
		return FUNCTION_POLLSEMAPHORE;
		break;
	case 11:
		return FUNCTION_REFERSEMASTATUS;
		break;
	case 12:
		return FUNCTION_IREFERSEMASTATUS;
		break;
	default:
		return "unknown";
		break;
	}
}

void CThsema::Invoke(CMIPS& context, unsigned int functionId)
{
	switch(functionId)
	{
	case 4:
		context.m_State.nGPR[CMIPS::V0].nD0 = static_cast<int32>(CreateSemaphore(
		    reinterpret_cast<SEMAPHORE*>(&m_ram[context.m_State.nGPR[CMIPS::A0].nV0])));
		break;
	case 5:
		context.m_State.nGPR[CMIPS::V0].nD0 = static_cast<int32>(DeleteSemaphore(
		    context.m_State.nGPR[CMIPS::A0].nV0));
		break;
	case 6:
		context.m_State.nGPR[CMIPS::V0].nD0 = static_cast<int32>(SignalSemaphore(
		    context.m_State.nGPR[CMIPS::A0].nV0));
		break;
	case 7:
		context.m_State.nGPR[CMIPS::V0].nD0 = static_cast<int32>(iSignalSemaphore(
		    context.m_State.nGPR[CMIPS::A0].nV0));
		break;
	case 8:
		context.m_State.nGPR[CMIPS::V0].nD0 = static_cast<int32>(WaitSemaphore(
		    context.m_State.nGPR[CMIPS::A0].nV0));
		break;
	case 9:
		context.m_State.nGPR[CMIPS::V0].nD0 = static_cast<int32>(PollSemaphore(
		    context.m_State.nGPR[CMIPS::A0].nV0));
		break;
	case 11:
	case 12:
		context.m_State.nGPR[CMIPS::V0].nD0 = static_cast<int32>(ReferSemaphoreStatus(
		    context.m_State.nGPR[CMIPS::A0].nV0,
		    context.m_State.nGPR[CMIPS::A1].nV0));
		break;
	default:
		CLog::GetInstance().Warn(LOG_NAME, "Unknown function (%d) called at (%08X).\r\n", functionId, context.m_State.nPC);
		break;
	}
}

uint32 CThsema::CreateSemaphore(const SEMAPHORE* semaphore)
{
	return m_bios.CreateSemaphore(semaphore->initialCount, semaphore->maxCount, semaphore->options, semaphore->attributes);
}

uint32 CThsema::DeleteSemaphore(uint32 semaphoreId)
{
	return m_bios.DeleteSemaphore(semaphoreId);
}

uint32 CThsema::SignalSemaphore(uint32 semaphoreId)
{
	return m_bios.SignalSemaphore(semaphoreId, false);
}

uint32 CThsema::iSignalSemaphore(uint32 semaphoreId)
{
	return m_bios.SignalSemaphore(semaphoreId, true);
}

uint32 CThsema::WaitSemaphore(uint32 semaphoreId)
{
	return m_bios.WaitSemaphore(semaphoreId);
}

uint32 CThsema::PollSemaphore(uint32 semaphoreId)
{
	return m_bios.PollSemaphore(semaphoreId);
}

uint32 CThsema::ReferSemaphoreStatus(uint32 semaphoreId, uint32 statusPtr)
{
	return m_bios.ReferSemaphoreStatus(semaphoreId, statusPtr);
}
