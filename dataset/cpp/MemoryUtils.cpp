#include <cstring>
#include "MemoryUtils.h"
#include "Integer64.h"
#include "Log.h"

#define LOG_NAME "MemoryMap"

uint32 MemoryUtils_GetByteProxy(CMIPS* context, uint32 vAddress)
{
	uint32 address = context->m_pAddrTranslator(context, vAddress);
	return static_cast<uint32>(context->m_pMemoryMap->GetByte(address));
}

uint32 MemoryUtils_GetHalfProxy(CMIPS* context, uint32 vAddress)
{
	uint32 address = context->m_pAddrTranslator(context, vAddress);
	return static_cast<uint32>(context->m_pMemoryMap->GetHalf(address));
}

uint32 MemoryUtils_GetWordProxy(CMIPS* context, uint32 vAddress)
{
	uint32 address = context->m_pAddrTranslator(context, vAddress);
	return context->m_pMemoryMap->GetWord(address);
}

uint64 MemoryUtils_GetDoubleProxy(CMIPS* context, uint32 vAddress)
{
	uint32 address = context->m_pAddrTranslator(context, vAddress);
	assert((address & 0x07) == 0);
	auto e = context->m_pMemoryMap->GetReadMap(address);
	INTEGER64 result;
#ifdef _DEBUG
	result.q = 0xCCCCCCCCCCCCCCCCull;
#endif
	if(e)
	{
		switch(e->nType)
		{
		case CMemoryMap::MEMORYMAP_TYPE_MEMORY:
			result.q = *reinterpret_cast<uint64*>(reinterpret_cast<uint8*>(e->pPointer) + (address - e->nStart));
			break;
		case CMemoryMap::MEMORYMAP_TYPE_FUNCTION:
			for(unsigned int i = 0; i < 2; i++)
			{
				result.d[i] = e->handler(address + (i * 4), 0);
			}
			break;
		default:
			assert(0);
			break;
		}
	}
	return result.q;
}

uint128 MemoryUtils_GetQuadProxy(CMIPS* context, uint32 vAddress)
{
	uint32 address = context->m_pAddrTranslator(context, vAddress);
	address &= ~0x0F;
	auto e = context->m_pMemoryMap->GetReadMap(address);
	uint128 result;
#ifdef _DEBUG
	memset(&result, 0xCC, sizeof(result));
#endif
	if(e)
	{
		switch(e->nType)
		{
		case CMemoryMap::MEMORYMAP_TYPE_MEMORY:
			result = *reinterpret_cast<uint128*>(reinterpret_cast<uint8*>(e->pPointer) + (address - e->nStart));
			break;
		case CMemoryMap::MEMORYMAP_TYPE_FUNCTION:
			for(unsigned int i = 0; i < 4; i++)
			{
				result.nV[i] = e->handler(address + (i * 4), 0);
			}
			break;
		default:
			assert(0);
			break;
		}
	}
	return result;
}

void MemoryUtils_SetByteProxy(CMIPS* context, uint32 value, uint32 vAddress)
{
	uint32 address = context->m_pAddrTranslator(context, vAddress);
	context->m_pMemoryMap->SetByte(address, static_cast<uint8>(value));
}

void MemoryUtils_SetHalfProxy(CMIPS* context, uint32 value, uint32 vAddress)
{
	uint32 address = context->m_pAddrTranslator(context, vAddress);
	context->m_pMemoryMap->SetHalf(address, static_cast<uint16>(value));
}

void MemoryUtils_SetWordProxy(CMIPS* context, uint32 value, uint32 vAddress)
{
	uint32 address = context->m_pAddrTranslator(context, vAddress);
	context->m_pMemoryMap->SetWord(address, value);
}

void MemoryUtils_SetDoubleProxy(CMIPS* context, uint64 value64, uint32 vAddress)
{
	uint32 address = context->m_pAddrTranslator(context, vAddress);
	assert((address & 0x07) == 0);
	INTEGER64 value;
	value.q = value64;
	auto e = context->m_pMemoryMap->GetWriteMap(address);
	if(!e)
	{
		CLog::GetInstance().Print(LOG_NAME, "Wrote to unmapped memory (0x%08X, [0x%08X, 0x%08X]).\r\n",
		                          address, value.d0, value.d1);
		return;
	}
	switch(e->nType)
	{
	case CMemoryMap::MEMORYMAP_TYPE_MEMORY:
		*reinterpret_cast<uint64*>(reinterpret_cast<uint8*>(e->pPointer) + (address - e->nStart)) = value.q;
		break;
	case CMemoryMap::MEMORYMAP_TYPE_FUNCTION:
		for(unsigned int i = 0; i < 2; i++)
		{
			e->handler(address + (i * 4), value.d[i]);
		}
		break;
	default:
		assert(0);
		break;
	}
}

void MemoryUtils_SetQuadProxy(CMIPS* context, const uint128& value, uint32 vAddress)
{
	uint32 address = context->m_pAddrTranslator(context, vAddress);
	address &= ~0x0F;
	auto e = context->m_pMemoryMap->GetWriteMap(address);
	if(!e)
	{
		CLog::GetInstance().Print(LOG_NAME, "Wrote to unmapped memory (0x%08X, [0x%08X, 0x%08X, 0x%08X, 0x%08X]).\r\n",
		                          address, value.nV0, value.nV1, value.nV2, value.nV3);
		return;
	}
	switch(e->nType)
	{
	case CMemoryMap::MEMORYMAP_TYPE_MEMORY:
		*reinterpret_cast<uint128*>(reinterpret_cast<uint8*>(e->pPointer) + (address - e->nStart)) = value;
		break;
	case CMemoryMap::MEMORYMAP_TYPE_FUNCTION:
		for(unsigned int i = 0; i < 4; i++)
		{
			e->handler(address + (i * 4), value.nV[i]);
		}
		break;
	default:
		assert(0);
		break;
	}
}
