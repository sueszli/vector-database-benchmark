#include "Iop_Modload.h"
#include "IopBios.h"
#include "../Log.h"

using namespace Iop;

#define LOG_NAME ("iop_modload")

#define FUNCTION_LOADSTARTMODULE "LoadStartModule"
#define FUNCTION_STARTMODULE "StartModule"
#define FUNCTION_LOADMODULEBUFFERADDRESS "LoadModuleBufferAddress"
#define FUNCTION_LOADMODULEBUFFER "LoadModuleBuffer"
#define FUNCTION_GETMODULEIDLIST "GetModuleIdList"
#define FUNCTION_REFERMODULESTATUS "ReferModuleStatus"
#define FUNCTION_LOADMODULEWITHOPTION "LoadModuleWithOption"
#define FUNCTION_STOPMODULE "StopModule"
#define FUNCTION_UNLOADMODULE "UnloadModule"
#define FUNCTION_SEARCHMODULEBYNAME "SearchModuleByName"
#define FUNCTION_ALLOCLOADMEMORY "AllocLoadMemory"

struct LMWO_OPTION
{
	uint8 position;
	uint8 access;
	uint8 reserved1[2];
	uint32 distAddrPtr;
	uint32 distOffset;
	uint32 funcTablePtr;
	uint32 funcOpt;
	uint32 reserved2[3];
};
static_assert(sizeof(LMWO_OPTION) == 0x20);

CModload::CModload(CIopBios& bios, uint8* ram)
    : m_bios(bios)
    , m_ram(ram)
{
}

std::string CModload::GetId() const
{
	return "modload";
}

std::string CModload::GetFunctionName(unsigned int functionId) const
{
	switch(functionId)
	{
	case 7:
		return FUNCTION_LOADSTARTMODULE;
		break;
	case 8:
		return FUNCTION_STARTMODULE;
		break;
	case 9:
		return FUNCTION_LOADMODULEBUFFERADDRESS;
		break;
	case 10:
		return FUNCTION_LOADMODULEBUFFER;
		break;
	case 16:
		return FUNCTION_GETMODULEIDLIST;
		break;
	case 17:
		return FUNCTION_REFERMODULESTATUS;
		break;
	case 19:
		return FUNCTION_LOADMODULEWITHOPTION;
		break;
	case 20:
		return FUNCTION_STOPMODULE;
		break;
	case 21:
		return FUNCTION_UNLOADMODULE;
		break;
	case 22:
		return FUNCTION_SEARCHMODULEBYNAME;
		break;
	case 28:
		return FUNCTION_ALLOCLOADMEMORY;
		break;
	default:
		return "unknown";
		break;
	}
}

void CModload::Invoke(CMIPS& context, unsigned int functionId)
{
	switch(functionId)
	{
	case 7:
		context.m_State.nGPR[CMIPS::V0].nD0 = static_cast<int32>(LoadStartModule(
		    context.m_State.nGPR[CMIPS::A0].nV0,
		    context.m_State.nGPR[CMIPS::A1].nV0,
		    context.m_State.nGPR[CMIPS::A2].nV0,
		    context.m_State.nGPR[CMIPS::A3].nV0));
		break;
	case 8:
		context.m_State.nGPR[CMIPS::V0].nD0 = static_cast<int32>(StartModule(
		    context.m_State.nGPR[CMIPS::A0].nV0,
		    context.m_State.nGPR[CMIPS::A1].nV0,
		    context.m_State.nGPR[CMIPS::A2].nV0,
		    context.m_State.nGPR[CMIPS::A3].nV0,
		    context.m_pMemoryMap->GetWord(context.m_State.nGPR[CMIPS::SP].nV0 + 0x10)));
		break;
	case 9:
		context.m_State.nGPR[CMIPS::V0].nD0 = static_cast<int32>(LoadModuleBufferAddress(
		    context.m_State.nGPR[CMIPS::A0].nV0,
		    context.m_State.nGPR[CMIPS::A1].nV0,
		    context.m_State.nGPR[CMIPS::A2].nV0));
		break;
	case 10:
		context.m_State.nGPR[CMIPS::V0].nD0 = static_cast<int32>(LoadModuleBuffer(
		    context.m_State.nGPR[CMIPS::A0].nV0));
		break;
	case 16:
		context.m_State.nGPR[CMIPS::V0].nD0 = static_cast<int32>(GetModuleIdList(
		    context.m_State.nGPR[CMIPS::A0].nV0,
		    context.m_State.nGPR[CMIPS::A1].nV0,
		    context.m_State.nGPR[CMIPS::A2].nV0));
		break;
	case 17:
		context.m_State.nGPR[CMIPS::V0].nD0 = ReferModuleStatus(
		    context.m_State.nGPR[CMIPS::A0].nV0,
		    context.m_State.nGPR[CMIPS::A1].nV0);
		break;
	case 19:
		context.m_State.nGPR[CMIPS::V0].nD0 = LoadModuleWithOption(
		    context.m_State.nGPR[CMIPS::A0].nV0,
		    context.m_State.nGPR[CMIPS::A1].nV0);
		break;
	case 20:
		context.m_State.nGPR[CMIPS::V0].nD0 = StopModule(
		    context.m_State.nGPR[CMIPS::A0].nV0,
		    context.m_State.nGPR[CMIPS::A1].nV0,
		    context.m_State.nGPR[CMIPS::A2].nV0,
		    context.m_State.nGPR[CMIPS::A3].nV0);
		break;
	case 21:
		context.m_State.nGPR[CMIPS::V0].nD0 = UnloadModule(
		    context.m_State.nGPR[CMIPS::A0].nV0);
		break;
	case 22:
		context.m_State.nGPR[CMIPS::V0].nD0 = SearchModuleByName(
		    context.m_State.nGPR[CMIPS::A0].nV0);
		break;
	case 28:
		context.m_State.nGPR[CMIPS::V0].nD0 = AllocLoadMemory(
		    context.m_State.nGPR[CMIPS::A0].nV0,
		    context.m_State.nGPR[CMIPS::A1].nV0,
		    context.m_State.nGPR[CMIPS::A2].nV0);
		break;
	default:
		CLog::GetInstance().Warn(LOG_NAME, "(%08X): Unknown function (%d) called.\r\n",
		                         context.m_State.nPC, functionId);
		break;
	}
}

uint32 CModload::LoadStartModule(uint32 pathPtr, uint32 argsLength, uint32 argsPtr, uint32 resultPtr)
{
	const char* path = reinterpret_cast<const char*>(m_ram + pathPtr);
	const char* args = reinterpret_cast<const char*>(m_ram + argsPtr);
	try
	{
		auto moduleId = m_bios.LoadModuleFromPath(path);
		if(moduleId >= 0)
		{
			moduleId = m_bios.StartModule(CIopBios::MODULESTARTREQUEST_SOURCE::LOCAL, moduleId, path, args, argsLength);
		}
		return moduleId;
	}
	catch(const std::exception& except)
	{
		CLog::GetInstance().Warn(LOG_NAME, "Error occured while trying to load module '%s' : %s\r\n",
		                         path, except.what());
	}
	return -1;
}

uint32 CModload::StartModule(uint32 moduleId, uint32 pathPtr, uint32 argsLength, uint32 argsPtr, uint32 resultPtr)
{
	const char* path = reinterpret_cast<const char*>(m_ram + pathPtr);
	const char* args = reinterpret_cast<const char*>(m_ram + argsPtr);
	CLog::GetInstance().Print(LOG_NAME, FUNCTION_STARTMODULE "(moduleId = %d, path = '%s', argsLength = %d, argsPtr = 0x%08X, resultPtr = 0x%08X);\r\n",
	                          moduleId, path, argsLength, argsPtr, resultPtr);
	auto result = m_bios.StartModule(CIopBios::MODULESTARTREQUEST_SOURCE::LOCAL, moduleId, path, args, argsLength);
	return result;
}

uint32 CModload::LoadModuleBufferAddress(uint32 modBufPtr, uint32 dstAddr, uint32 offset)
{
	CLog::GetInstance().Print(LOG_NAME, FUNCTION_LOADMODULEBUFFERADDRESS "(modBufPtr = 0x%08X, dstAddr = 0x%08X, offset = %d);\r\n",
	                          modBufPtr, dstAddr, offset);
	assert((modBufPtr & 0x03) == 0);
	if(dstAddr == 0)
	{
		return m_bios.LoadModuleFromAddress(modBufPtr);
	}
	else
	{
		return m_bios.LoadModuleFromAddress(modBufPtr, dstAddr + offset, offset == 0);
	}
}

uint32 CModload::LoadModuleBuffer(uint32 modBufPtr)
{
	CLog::GetInstance().Print(LOG_NAME, FUNCTION_LOADMODULEBUFFER "(modBufPtr = 0x%08X);\r\n",
	                          modBufPtr);
	assert((modBufPtr & 0x03) == 0);
	auto result = m_bios.LoadModuleFromAddress(modBufPtr);
	return result;
}

uint32 CModload::GetModuleIdList(uint32 readBufPtr, uint32 readBufSize, uint32 moduleCountPtr)
{
	CLog::GetInstance().Print(LOG_NAME, FUNCTION_GETMODULEIDLIST "(readBufPtr = 0x%08X, readBufSize = 0x%08X, moduleCountPtr = 0x%08X);\r\n",
	                          readBufPtr, readBufSize, moduleCountPtr);
	auto moduleCount = (moduleCountPtr != 0) ? reinterpret_cast<uint32*>(m_ram + moduleCountPtr) : nullptr;
	if(moduleCount)
	{
		(*moduleCount) = 0;
	}
	CLog::GetInstance().Warn(LOG_NAME, "Using " FUNCTION_GETMODULEIDLIST ", but it is not implemented.\r\n");
	return 0;
}

int32 CModload::ReferModuleStatus(uint32 moduleId, uint32 moduleStatusPtr)
{
	CLog::GetInstance().Print(LOG_NAME, FUNCTION_REFERMODULESTATUS "(moduleId = %d, moduleStatusPtr = 0x%08X);\r\n",
	                          moduleId, moduleStatusPtr);
	return m_bios.ReferModuleStatus(moduleId, moduleStatusPtr);
}

int32 CModload::LoadModuleWithOption(uint32 pathPtr, uint32 optionPtr)
{
	CLog::GetInstance().Print(LOG_NAME, FUNCTION_LOADMODULEWITHOPTION "(pathPtr = 0x%08X, optionPtr = 0x%08X);\r\n",
	                          pathPtr, optionPtr);
	const char* path = reinterpret_cast<const char*>(m_ram + pathPtr);
	FRAMEWORK_MAYBE_UNUSED auto option = reinterpret_cast<const LMWO_OPTION*>(m_ram + optionPtr);
	//We don't support custom loader functions
	assert(option->funcTablePtr == 0);
	return m_bios.LoadModuleFromPath(path);
}

int32 CModload::StopModule(uint32 moduleId, uint32 argsLength, uint32 argsPtr, uint32 resultPtr)
{
	FRAMEWORK_MAYBE_UNUSED const char* args = reinterpret_cast<const char*>(m_ram + argsPtr);
	CLog::GetInstance().Print(LOG_NAME, FUNCTION_STOPMODULE "(moduleId = %d, argsLength = %d, argsPtr = 0x%08X, resultPtr = 0x%08X);\r\n",
	                          moduleId, argsLength, argsPtr, resultPtr);
	return m_bios.StopModule(CIopBios::MODULESTARTREQUEST_SOURCE::LOCAL, moduleId);
}

int32 CModload::UnloadModule(uint32 moduleId)
{
	CLog::GetInstance().Print(LOG_NAME, FUNCTION_UNLOADMODULE "(moduleId = %d);\r\n",
	                          moduleId);
	return m_bios.UnloadModule(moduleId);
}

int32 CModload::SearchModuleByName(uint32 moduleNamePtr)
{
	CLog::GetInstance().Print(LOG_NAME, FUNCTION_SEARCHMODULEBYNAME "(moduleNamePtr = %s);\r\n",
	                          PrintStringParameter(m_ram, moduleNamePtr).c_str());
	return m_bios.SearchModuleByName(reinterpret_cast<const char*>(m_ram + moduleNamePtr));
}

int32 CModload::AllocLoadMemory(uint32 unknown1, uint32 size, uint32 unknown2)
{
	CLog::GetInstance().Print(LOG_NAME, FUNCTION_ALLOCLOADMEMORY "(unknown1 = %d, size = 0x%08X, unknown2 = %d);\r\n",
	                          unknown1, size, unknown2);
	auto sysmem = m_bios.GetSysmem();
	uint32 result = sysmem->AllocateMemory(size, 0, 0);
	return result;
}
