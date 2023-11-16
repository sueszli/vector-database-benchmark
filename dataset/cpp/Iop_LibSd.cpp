#include "Iop_LibSd.h"
#include "../Log.h"
#include "string_format.h"

//Not an actual implementation of the LIBSD module
//It is only used for debugging purposes (ie.: function names)

using namespace Iop;

#define FUNCTION_INIT "Init"
#define FUNCTION_SETPARAM "SetParam"
#define FUNCTION_GETPARAM "GetParam"
#define FUNCTION_SETSWITCH "SetSwitch"
#define FUNCTION_GETSWITCH "GetSwitch"
#define FUNCTION_SETADDR "SetAddr"
#define FUNCTION_GETADDR "GetAddr"
#define FUNCTION_SETCOREATTR "SetCoreAttr"
#define FUNCTION_PROCBATCH "ProcBatch"
#define FUNCTION_VOICETRANS "VoiceTrans"
#define FUNCTION_BLOCKTRANS "BlockTrans"
#define FUNCTION_VOICETRANSSTATUS "VoiceTransStatus"
#define FUNCTION_BLOCKTRANSSTATUS "BlockTransStatus"
#define FUNCTION_SETTRANSCALLBACK "SetTransCallback"
#define FUNCTION_SETEFFECTATTR "SetEffectAttr"
#define FUNCTION_SETTRANSINTRHANDLER "SetTransIntrHandler"
#define FUNCTION_SETSPU2INTRHANDLER "SetSpu2IntrHandler"

#define LOG_NAME "iop_libsd"

std::string CLibSd::GetId() const
{
	return "libsd";
}

std::string CLibSd::GetFunctionName(unsigned int functionId) const
{
	switch(functionId)
	{
	case 4:
		return FUNCTION_INIT;
		break;
	case 5:
		return FUNCTION_SETPARAM;
		break;
	case 6:
		return FUNCTION_GETPARAM;
		break;
	case 7:
		return FUNCTION_SETSWITCH;
		break;
	case 8:
		return FUNCTION_GETSWITCH;
		break;
	case 9:
		return FUNCTION_SETADDR;
		break;
	case 10:
		return FUNCTION_GETADDR;
		break;
	case 11:
		return FUNCTION_SETCOREATTR;
		break;
	case 15:
		return FUNCTION_PROCBATCH;
		break;
	case 17:
		return FUNCTION_VOICETRANS;
		break;
	case 18:
		return FUNCTION_BLOCKTRANS;
		break;
	case 19:
		return FUNCTION_VOICETRANSSTATUS;
		break;
	case 20:
		return FUNCTION_BLOCKTRANSSTATUS;
		break;
	case 21:
		return FUNCTION_SETTRANSCALLBACK;
		break;
	case 23:
		return FUNCTION_SETEFFECTATTR;
		break;
	case 26:
		return FUNCTION_SETTRANSINTRHANDLER;
		break;
	case 27:
		return FUNCTION_SETSPU2INTRHANDLER;
		break;
	default:
		return "unknown";
		break;
	}
}

void CLibSd::Invoke(CMIPS&, unsigned int)
{
}

static std::string DecodeAddress(uint16 addressId)
{
	std::string result;
	bool voiceAddress = false;
	switch(addressId >> 8)
	{
	case 0x1F:
		result = "IRQA";
		break;
	case 0x20:
		result = "SSA";
		voiceAddress = true;
		break;
	case 0x21:
		result = "LSAX";
		voiceAddress = true;
		break;
	case 0x22:
		result = "NAX";
		voiceAddress = true;
		break;
	default:
		result = string_format("unknown (0x%02X)", addressId >> 8);
		break;
	}
	if(voiceAddress)
	{
		result += string_format(", CORE%d, VOICE%d", addressId & 1, (addressId & 0x3E) >> 1);
	}
	else
	{
		result += string_format(", CORE%d", addressId & 1);
	}
	return result;
}

static std::string DecodeParam(uint16 paramId)
{
	std::string result;
	uint16 paramIdx = (paramId >> 8);
	bool voiceParam = (paramIdx < 8);
	switch(paramIdx)
	{
	case 0x00:
		result = "VOLL";
		break;
	case 0x01:
		result = "VOLR";
		break;
	case 0x02:
		result = "PITCH";
		break;
	case 0x03:
		result = "ADSR1";
		break;
	case 0x04:
		result = "ADSR2";
		break;
	case 0x05:
		result = "ENVX";
		break;
	case 0x06:
		result = "VOLXL";
		break;
	case 0x07:
		result = "VOLXR";
		break;
	case 0x08:
		result = "MMIX";
		break;
	case 0x09:
		result = "MVOLL";
		break;
	case 0x0A:
		result = "MVOLR";
		break;
	case 0x0B:
		result = "EVOLL";
		break;
	case 0x0C:
		result = "EVOLR";
		break;
	case 0x0D:
		result = "AVOLL";
		break;
	case 0x0E:
		result = "AVOLR";
		break;
	default:
		result = string_format("unknown (0x%02X)", paramIdx);
		break;
	}
	if(voiceParam)
	{
		result += string_format(", CORE%d, VOICE%d", paramId & 1, (paramId & 0x3E) >> 1);
	}
	else
	{
		result += string_format(", CORE%d", paramId & 1);
	}
	return result;
}

static std::string DecodeSwitch(uint16 switchId)
{
	std::string result;
	switch(switchId >> 8)
	{
	case 0x13:
		result = "PMON";
		break;
	case 0x14:
		result = "NON";
		break;
	case 0x15:
		result = "KON";
		break;
	case 0x16:
		result = "KOFF";
		break;
	case 0x17:
		result = "ENDX";
		break;
	case 0x18:
		result = "VMIXL";
		break;
	case 0x19:
		result = "VMIXEL";
		break;
	case 0x1A:
		result = "VMIXR";
		break;
	case 0x1B:
		result = "VMIXER";
		break;
	default:
		result = string_format("unknown (0x%02X)", switchId >> 8);
		break;
	}
	result += string_format(", CORE%d", switchId & 1);
	return result;
}

void CLibSd::TraceCall(CMIPS& context, unsigned int functionId)
{
	switch(functionId)
	{
	case 4:
		CLog::GetInstance().Print(LOG_NAME, FUNCTION_INIT "(flag = %d);\r\n",
		                          context.m_State.nGPR[CMIPS::A0].nV0);
		break;
	case 5:
		CLog::GetInstance().Print(LOG_NAME, FUNCTION_SETPARAM "(entry = 0x%04X, value = 0x%04X); //(%s)\r\n",
		                          context.m_State.nGPR[CMIPS::A0].nV0, context.m_State.nGPR[CMIPS::A1].nV0,
		                          DecodeParam(static_cast<uint16>(context.m_State.nGPR[CMIPS::A0].nV0)).c_str());
		break;
	case 6:
		CLog::GetInstance().Print(LOG_NAME, FUNCTION_GETPARAM "(entry = 0x%04X); //(%s)\r\n",
		                          context.m_State.nGPR[CMIPS::A0].nV0,
		                          DecodeParam(static_cast<uint16>(context.m_State.nGPR[CMIPS::A0].nV0)).c_str());
		break;
	case 7:
		CLog::GetInstance().Print(LOG_NAME, FUNCTION_SETSWITCH "(entry = 0x%04X, value = 0x%08X); //(%s)\r\n",
		                          context.m_State.nGPR[CMIPS::A0].nV0, context.m_State.nGPR[CMIPS::A1].nV0,
		                          DecodeSwitch(static_cast<uint16>(context.m_State.nGPR[CMIPS::A0].nV0)).c_str());
		break;
	case 8:
		CLog::GetInstance().Print(LOG_NAME, FUNCTION_GETSWITCH "(entry = 0x%04X); //(%s)\r\n",
		                          context.m_State.nGPR[CMIPS::A0].nV0,
		                          DecodeSwitch(static_cast<uint16>(context.m_State.nGPR[CMIPS::A0].nV0)).c_str());
		break;
	case 9:
		CLog::GetInstance().Print(LOG_NAME, FUNCTION_SETADDR "(entry = 0x%04X, value = 0x%08X); //(%s)\r\n",
		                          context.m_State.nGPR[CMIPS::A0].nV0, context.m_State.nGPR[CMIPS::A1].nV0,
		                          DecodeAddress(static_cast<uint16>(context.m_State.nGPR[CMIPS::A0].nV0)).c_str());
		break;
	case 10:
		CLog::GetInstance().Print(LOG_NAME, FUNCTION_GETADDR "(entry = 0x%04X); //(%s)\r\n",
		                          context.m_State.nGPR[CMIPS::A0].nV0,
		                          DecodeAddress(static_cast<uint16>(context.m_State.nGPR[CMIPS::A0].nV0)).c_str());
		break;
	case 11:
		CLog::GetInstance().Print(LOG_NAME, FUNCTION_SETCOREATTR "(entry = 0x%04X, value = 0x%04X);\r\n",
		                          context.m_State.nGPR[CMIPS::A0].nV0, context.m_State.nGPR[CMIPS::A1].nV0);
		break;
	case 17:
		CLog::GetInstance().Print(LOG_NAME, FUNCTION_VOICETRANS "(channel = 0x%04X, mode = 0x%04X, maddr = 0x%08X, saddr = 0x%08X, size = 0x%08X);\r\n",
		                          context.m_State.nGPR[CMIPS::A0].nV0, context.m_State.nGPR[CMIPS::A1].nV0,
		                          context.m_State.nGPR[CMIPS::A2].nV0, context.m_State.nGPR[CMIPS::A3].nV0,
		                          context.m_pMemoryMap->GetWord(context.m_State.nGPR[CMIPS::SP].nV0 + 0x10));
		break;
	case 18:
		CLog::GetInstance().Print(LOG_NAME, FUNCTION_BLOCKTRANS "(channel = 0x%04X, mode = 0x%04X, maddr = 0x%08X, size = 0x%08X);\r\n",
		                          context.m_State.nGPR[CMIPS::A0].nV0, context.m_State.nGPR[CMIPS::A1].nV0,
		                          context.m_State.nGPR[CMIPS::A2].nV0, context.m_State.nGPR[CMIPS::A3].nV0);
		break;
	case 19:
		CLog::GetInstance().Print(LOG_NAME, FUNCTION_VOICETRANSSTATUS "(channel = 0x%04X, flag = 0x%04X);\r\n",
		                          context.m_State.nGPR[CMIPS::A0].nV0, context.m_State.nGPR[CMIPS::A1].nV0);
		break;
	case 20:
		CLog::GetInstance().Print(LOG_NAME, FUNCTION_BLOCKTRANSSTATUS "(channel = 0x%04X, flag = 0x%04X);\r\n",
		                          context.m_State.nGPR[CMIPS::A0].nV0, context.m_State.nGPR[CMIPS::A1].nV0);
		break;
	case 21:
		CLog::GetInstance().Print(LOG_NAME, FUNCTION_SETTRANSCALLBACK "(channel = 0x%04X, function = 0x%08X);\r\n",
		                          context.m_State.nGPR[CMIPS::A0].nV0, context.m_State.nGPR[CMIPS::A1].nV0);
		break;
	case 23:
		CLog::GetInstance().Print(LOG_NAME, FUNCTION_SETEFFECTATTR "(core = 0x%04X, attr = 0x%08X);\r\n",
		                          context.m_State.nGPR[CMIPS::A0].nV0, context.m_State.nGPR[CMIPS::A1].nV0);
		break;
	case 26:
		CLog::GetInstance().Print(LOG_NAME, FUNCTION_SETTRANSINTRHANDLER "(channel = 0x%04X, function = 0x%08X, arg = 0x%08X);\r\n",
		                          context.m_State.nGPR[CMIPS::A0].nV0, context.m_State.nGPR[CMIPS::A1].nV0,
		                          context.m_State.nGPR[CMIPS::A2].nV0);
		break;
	case 27:
		CLog::GetInstance().Print(LOG_NAME, FUNCTION_SETSPU2INTRHANDLER "(function = 0x%08X, arg = 0x%08X);\r\n",
		                          context.m_State.nGPR[CMIPS::A0].nV0, context.m_State.nGPR[CMIPS::A1].nV0);
		break;
	default:
		CLog::GetInstance().Print(LOG_NAME, "unknownlibsd(%d);\r\n", functionId);
		break;
	}
}
