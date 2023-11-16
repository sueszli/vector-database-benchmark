#include <cassert>
#include "MemoryMap.h"
#include "Log.h"

#define LOG_NAME "MemoryMap"

void CMemoryMap::InsertReadMap(uint32 start, uint32 end, void* pointer, unsigned char key)
{
	assert(GetReadMap(start) == nullptr);
	InsertMap(m_readMap, start, end, pointer, key);
}

void CMemoryMap::InsertReadMap(uint32 start, uint32 end, const MemoryMapHandlerType& handler, unsigned char key)
{
	assert(GetReadMap(start) == nullptr);
	InsertMap(m_readMap, start, end, handler, key);
}

void CMemoryMap::InsertWriteMap(uint32 start, uint32 end, void* pointer, unsigned char key)
{
	assert(GetWriteMap(start) == nullptr);
	InsertMap(m_writeMap, start, end, pointer, key);
}

void CMemoryMap::InsertWriteMap(uint32 start, uint32 end, const MemoryMapHandlerType& handler, unsigned char key)
{
	assert(GetWriteMap(start) == nullptr);
	InsertMap(m_writeMap, start, end, handler, key);
}

void CMemoryMap::InsertInstructionMap(uint32 start, uint32 end, void* pointer, unsigned char key)
{
	assert(GetMap(m_instructionMap, start) == nullptr);
	InsertMap(m_instructionMap, start, end, pointer, key);
}

const CMemoryMap::MemoryMapListType& CMemoryMap::GetInstructionMaps()
{
	return m_instructionMap;
}

const CMemoryMap::MEMORYMAPELEMENT* CMemoryMap::GetReadMap(uint32 address) const
{
	return GetMap(m_readMap, address);
}

const CMemoryMap::MEMORYMAPELEMENT* CMemoryMap::GetWriteMap(uint32 address) const
{
	return GetMap(m_writeMap, address);
}

void CMemoryMap::InsertMap(MemoryMapListType& memoryMap, uint32 start, uint32 end, void* pointer, unsigned char key)
{
	MEMORYMAPELEMENT element;
	element.nStart = start;
	element.nEnd = end;
	element.pPointer = pointer;
	element.nType = MEMORYMAP_TYPE_MEMORY;
	memoryMap.push_back(element);
}

void CMemoryMap::InsertMap(MemoryMapListType& memoryMap, uint32 start, uint32 end, const MemoryMapHandlerType& handler, unsigned char key)
{
	MEMORYMAPELEMENT element;
	element.nStart = start;
	element.nEnd = end;
	element.handler = handler;
	element.pPointer = nullptr;
	element.nType = MEMORYMAP_TYPE_FUNCTION;
	memoryMap.push_back(element);
}

const CMemoryMap::MEMORYMAPELEMENT* CMemoryMap::GetMap(const MemoryMapListType& memoryMap, uint32 nAddress)
{
	for(const auto& mapElement : memoryMap)
	{
		if(nAddress <= mapElement.nEnd)
		{
			if(!(nAddress >= mapElement.nStart)) return nullptr;
			return &mapElement;
		}
	}
	return nullptr;
}

uint8 CMemoryMap::GetByte(uint32 nAddress)
{
	const auto e = GetMap(m_readMap, nAddress);
	if(!e)
	{
		CLog::GetInstance().Print(LOG_NAME, "Read byte from unmapped memory (0x%08X).\r\n", nAddress);
		return 0xCC;
	}
	switch(e->nType)
	{
	case MEMORYMAP_TYPE_MEMORY:
		return *(uint8*)&((uint8*)e->pPointer)[nAddress - e->nStart];
		break;
	case MEMORYMAP_TYPE_FUNCTION:
		return static_cast<uint8>(e->handler(nAddress, 0));
		break;
	default:
		assert(0);
		return 0xCC;
		break;
	}
}

void CMemoryMap::SetByte(uint32 nAddress, uint8 nValue)
{
	const auto e = GetMap(m_writeMap, nAddress);
	if(!e)
	{
		CLog::GetInstance().Print(LOG_NAME, "Wrote byte to unmapped memory (0x%08X, 0x%02X).\r\n", nAddress, nValue);
		return;
	}
	switch(e->nType)
	{
	case MEMORYMAP_TYPE_MEMORY:
		*(uint8*)&((uint8*)e->pPointer)[nAddress - e->nStart] = nValue;
		break;
	case MEMORYMAP_TYPE_FUNCTION:
		e->handler(nAddress, nValue);
		break;
	default:
		assert(0);
		break;
	}
}

//////////////////////////////////////////////////////////////////
//LSB First Memory Map Implementation
//////////////////////////////////////////////////////////////////

uint16 CMemoryMap_LSBF::GetHalf(uint32 nAddress)
{
	assert((nAddress & 0x01) == 0);
	const auto e = GetMap(m_readMap, nAddress);
	if(!e)
	{
		CLog::GetInstance().Print(LOG_NAME, "Read half from unmapped memory (0x%08X).\r\n", nAddress);
		return 0xCCCC;
	}
	switch(e->nType)
	{
	case MEMORYMAP_TYPE_MEMORY:
		return *(uint16*)&((uint8*)e->pPointer)[nAddress - e->nStart];
		break;
	default:
		return static_cast<uint16>(e->handler(nAddress, 0));
		break;
	}
}

uint32 CMemoryMap_LSBF::GetWord(uint32 nAddress)
{
	assert((nAddress & 0x03) == 0);
	const auto e = GetMap(m_readMap, nAddress);
	if(!e)
	{
		CLog::GetInstance().Print(LOG_NAME, "Read word from unmapped memory (0x%08X).\r\n", nAddress);
		return 0xCCCCCCCC;
	}
	switch(e->nType)
	{
	case MEMORYMAP_TYPE_MEMORY:
		return *(uint32*)&((uint8*)e->pPointer)[nAddress - e->nStart];
		break;
	case MEMORYMAP_TYPE_FUNCTION:
		return e->handler(nAddress, 0);
		break;
	default:
		assert(0);
		return 0xCCCCCCCC;
		break;
	}
}

uint32 CMemoryMap_LSBF::GetInstruction(uint32 address)
{
	assert((address & 0x03) == 0);
	const auto e = GetMap(m_instructionMap, address);
	if(!e) return 0xCCCCCCCC;
	switch(e->nType)
	{
	case MEMORYMAP_TYPE_MEMORY:
		return *reinterpret_cast<uint32*>(&reinterpret_cast<uint8*>(e->pPointer)[address - e->nStart]);
		break;
	default:
		assert(0);
		return 0xCCCCCCCC;
		break;
	}
}

void CMemoryMap_LSBF::SetHalf(uint32 nAddress, uint16 nValue)
{
	assert((nAddress & 0x01) == 0);
	const auto e = GetMap(m_writeMap, nAddress);
	if(!e)
	{
		CLog::GetInstance().Print(LOG_NAME, "Wrote half to unmapped memory (0x%08X, 0x%04X).\r\n", nAddress, nValue);
		return;
	}
	switch(e->nType)
	{
	case MEMORYMAP_TYPE_MEMORY:
		*reinterpret_cast<uint16*>(&reinterpret_cast<uint8*>(e->pPointer)[nAddress - e->nStart]) = nValue;
		break;
	case MEMORYMAP_TYPE_FUNCTION:
		e->handler(nAddress, nValue);
		break;
	default:
		assert(0);
		break;
	}
}

void CMemoryMap_LSBF::SetWord(uint32 nAddress, uint32 nValue)
{
	assert((nAddress & 0x03) == 0);
	const auto e = GetMap(m_writeMap, nAddress);
	if(!e)
	{
		CLog::GetInstance().Print(LOG_NAME, "Wrote word to unmapped memory (0x%08X, 0x%08X).\r\n", nAddress, nValue);
		return;
	}
	switch(e->nType)
	{
	case MEMORYMAP_TYPE_MEMORY:
		*(uint32*)&((uint8*)e->pPointer)[nAddress - e->nStart] = nValue;
		break;
	case MEMORYMAP_TYPE_FUNCTION:
		e->handler(nAddress, nValue);
		break;
	default:
		assert(0);
		break;
	}
}
