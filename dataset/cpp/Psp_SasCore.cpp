#include "Psp_SasCore.h"
#include "Log.h"

#define LOGNAME ("Psp_SasCore")

using namespace Psp;

#define SPURAM_ALLOC_BASEADDRESS (0x40000)

// clang-format off
CSasCore::REVERBINFO CSasCore::g_ReverbStudioC =
{
	0x6FE0,
	{
		0x00E3,
		0x00A9,
		0x6F60,
		0x4FA8,
		0xBCE0,
		0x4510,
		0xBEF0,
		0xA680,
		0x5680,
		0x52C0,
		0x0DFB,
		0x0B58,
		0x0D09,
		0x0A3C,
		0x0BD9,
		0x0973,
		0x0B59,
		0x08DA,
		0x08D9,
		0x05E9,
		0x07EC,
		0x04B0,
		0x06EF,
		0x03D2,
		0x05EA,
		0x031D,
		0x031C,
		0x0238,
		0x0154,
		0x00AA,
		0x8000,
		0x8000,
	}
};

CSasCore::REVERBINFO CSasCore::g_ReverbHall =
{
	0xADE0,
	{
		0x01A5,
		0x0139,
		0x6000,
		0x5000,
		0x4C00,
		0xB800,
		0xBC00,
		0xC000,
		0x6000,
		0x5C00,
		0x15BA,
		0x11BB,
		0x14C2,
		0x10BD,
		0x11BC,
		0x0DC1,
		0x11C0,
		0x0DC3,
		0x0DC0,
		0x09C1,
		0x0BC4,
		0x07C1,
		0x0A00,
		0x06CD,
		0x09C2,
		0x05C1,
		0x05C0,
		0x041A,
		0x0274,
		0x013A,
		0x8000,
		0x8000,
	}
};

CSasCore::REVERBINFO CSasCore::g_ReverbSpace =
{
	0xF6C0,
	{
		0x033D,
		0x0231,
		0x7E00,
		0x5000,
		0xB400,
		0xB000,
		0x4C00,
		0xB000,
		0x6000,
		0x5400,
		0x1ED6,
		0x1A31,
		0x1D14,
		0x183B,
		0x1BC2,
		0x16B2,
		0x1A32,
		0x15EF,
		0x15EE,
		0x1055,
		0x1334,
		0x0F2D,
		0x11F6,
		0x0C5D,
		0x1056,
		0x0AE1,
		0x0AE0,
		0x07A2,
		0x0464,
		0x0232,
		0x8000,
		0x8000,
	}
};
// clang-format on

CSasCore::CSasCore(uint8* ram)
    : m_ram(ram)
{
}

std::string CSasCore::GetName() const
{
	return "sceSasCore";
}

void CSasCore::SetSpuInfo(Iop::CSpuSampleCache* sampleCache, Iop::CSpuBase* spu0, Iop::CSpuBase* spu1, uint8* spuRam, uint32 spuRamSize)
{
	assert(!m_spuSampleCache && !m_spu[0] && !m_spu[1]);

	m_spuSampleCache = sampleCache;
	m_spu[0] = spu0;
	m_spu[1] = spu1;

	m_spuRam = spuRam;
	m_spuRamSize = spuRamSize;
	m_spuRam[1] = 0x07;

	SPUMEMBLOCK endBlock;
	endBlock.address = m_spuRamSize;
	endBlock.size = -1;
	m_blocks.push_back(endBlock);
}

std::pair<Iop::CSpuBase*, uint32> CSasCore::TranslateSpuChannel(uint32 channelId) const
{
	assert(channelId < CHANNEL_COUNT);
	if(channelId >= CHANNEL_COUNT)
	{
		return std::make_pair(nullptr, 0);
	}

	if(channelId < 24)
	{
		return std::make_pair(m_spu[0], channelId);
	}
	else
	{
		return std::make_pair(m_spu[1], channelId - 24);
	}
}

Iop::CSpuBase::CHANNEL* CSasCore::GetSpuChannel(uint32 channelId) const
{
	auto chanInfo = TranslateSpuChannel(channelId);
	if(!chanInfo.first)
	{
		return nullptr;
	}
	return &chanInfo.first->GetChannel(chanInfo.second);
}

uint32 CSasCore::AllocMemory(uint32 size)
{
	assert(m_spuRamSize != 0);

	const uint32 startAddress = SPURAM_ALLOC_BASEADDRESS;
	uint32 currentAddress = startAddress;
	auto blockIterator(m_blocks.begin());
	while(blockIterator != m_blocks.end())
	{
		const auto& block(*blockIterator);
		uint32 space = block.address - currentAddress;
		if(space >= size)
		{
			SPUMEMBLOCK newBlock;
			newBlock.address = currentAddress;
			newBlock.size = size;
			m_blocks.insert(blockIterator, newBlock);
			return currentAddress;
		}
		currentAddress = block.address + block.size;
		blockIterator++;
	}
	assert(0);
	return 0;
}

void CSasCore::FreeMemory(uint32 address)
{
	for(auto blockIterator(m_blocks.begin());
	    blockIterator != m_blocks.end(); blockIterator++)
	{
		const auto& block(*blockIterator);
		if(block.address == address)
		{
			m_blocks.erase(blockIterator);
			return;
		}
	}
	assert(0);
}

#ifdef _DEBUG

void CSasCore::VerifyAllocationMap()
{
	for(auto blockIterator(m_blocks.begin());
	    blockIterator != m_blocks.end(); blockIterator++)
	{
		auto nextBlockIterator = blockIterator;
		nextBlockIterator++;
		if(nextBlockIterator == m_blocks.end()) break;
		auto& currBlock(*blockIterator);
		auto& nextBlock(*nextBlockIterator);
		assert(currBlock.address + currBlock.size <= nextBlock.address);
	}
}

#endif

void CSasCore::SetupReverb(const REVERBINFO& reverbInfo)
{
	for(unsigned int spuIdx = 0; spuIdx < 2; spuIdx++)
	{
		uint32 endAddress = 0x20000 + (spuIdx * 0x20000);
		uint32 startAddress = endAddress - reverbInfo.workAreaSize;
		m_spu[spuIdx]->SetReverbWorkAddressStart(startAddress);
		m_spu[spuIdx]->SetReverbWorkAddressEnd(endAddress - 1);
		for(unsigned int i = 0; i < Iop::CSpuBase::REVERB_PARAM_COUNT; i++)
		{
			uint32 param = reverbInfo.params[i];
			if(Iop::CSpuBase::g_reverbParamIsAddress[i])
			{
				param *= 8;
			}
			m_spu[spuIdx]->SetReverbParam(i, param);
		}
	}
}

uint32 CSasCore::Init(uint32 contextAddr, uint32 grain, uint32 unknown2, uint32 unknown3, uint32 frequency)
{
#ifdef _DEBUG
	CLog::GetInstance().Print(LOGNAME, "Init(contextAddr = 0x%0.8X, grain = %d, unk = 0x%0.8X, unk = 0x%0.8X, frequency = %d);\r\n",
	                          contextAddr, grain, unknown2, unknown3, frequency);
#endif
	m_grain = grain;
	return 0;
}

uint32 CSasCore::Core(uint32 contextAddr, uint32 bufferAddr)
{
#ifdef _DEBUG
	CLog::GetInstance().Print(LOGNAME, "Core(contextAddr = 0x%0.8X, bufferAddr = 0x%0.8X);\r\n", contextAddr, bufferAddr);
#endif

	assert(bufferAddr != 0);
	if(bufferAddr == 0)
	{
		return -1;
	}

	unsigned int sampleCount = m_grain * 2;
	int16* samplesSpu0 = reinterpret_cast<int16*>(m_ram + bufferAddr);
	int16* samplesSpu1 = reinterpret_cast<int16*>(alloca(sampleCount * sizeof(int16)));

	m_spu[0]->Render(samplesSpu0, sampleCount);
	m_spu[1]->Render(samplesSpu1, sampleCount);

	for(unsigned int i = 0; i < sampleCount; i++)
	{
		int32 resultSample = static_cast<int32>(samplesSpu0[i]) + static_cast<int32>(samplesSpu1[i]);
		resultSample = std::max<int32>(resultSample, SHRT_MIN);
		resultSample = std::min<int32>(resultSample, SHRT_MAX);
		samplesSpu0[i] = static_cast<int16>(resultSample);
	}

	return 0;
}

uint32 CSasCore::SetVoice(uint32 contextAddr, uint32 voice, uint32 dataPtr, uint32 dataSize, uint32 loop)
{
#ifdef _DEBUG
	CLog::GetInstance().Print(LOGNAME, "SetVoice(contextAddr = 0x%0.8X, voice = %d, dataPtr = 0x%0.8X, dataSize = 0x%0.8X, loop = %d);\r\n",
	                          contextAddr, voice, dataPtr, dataSize, loop);
#endif
	assert(dataPtr != NULL);
	if(dataPtr == NULL)
	{
		return -1;
	}

	auto channel = GetSpuChannel(voice);
	if(!channel) return -1;
	uint8* samples = m_ram + dataPtr;

	uint32 currentAddress = channel->address;
	uint32 allocationSize = ((dataSize + 0xF) / 0x10) * 0x10;

	if(currentAddress != 0)
	{
		FreeMemory(currentAddress);
	}

	currentAddress = AllocMemory(allocationSize);
	assert((currentAddress + allocationSize) <= m_spuRamSize);
	assert(currentAddress != 0);
	if(currentAddress != 0)
	{
		memcpy(m_spuRam + currentAddress, samples, dataSize);
		m_spuSampleCache->ClearRange(currentAddress, dataSize);
	}

	channel->address = currentAddress;
	channel->repeat = currentAddress;

	return 0;
}

uint32 CSasCore::SetPitch(uint32 contextAddr, uint32 voice, uint32 pitch)
{
#ifdef _DEBUG
	CLog::GetInstance().Print(LOGNAME, "SetPitch(contextAddr = 0x%0.8X, voice = %d, pitch = 0x%0.4X);\r\n",
	                          contextAddr, voice, pitch);
#endif
	auto chanInfo = TranslateSpuChannel(voice);
	if(!chanInfo.first) return -1;
	auto* channel = &chanInfo.first->GetChannel(chanInfo.second);
	channel->pitch = static_cast<uint16>(pitch);
	chanInfo.first->OnChannelPitchChanged(chanInfo.second);
	return 0;
}

uint32 CSasCore::SetVolume(uint32 contextAddr, uint32 voice, uint32 left, uint32 right, uint32 effectLeft, uint32 effectRight)
{
#ifdef _DEBUG
	CLog::GetInstance().Print(LOGNAME, "SetVolume(contextAddr = 0x%0.8X, voice = %d, left = 0x%0.4X, right = 0x%0.4X, effectLeft = 0x%0.4X, effectRight = 0x%0.4X);\r\n",
	                          contextAddr, voice, left, right, effectLeft, effectRight);
#endif
	auto channel = GetSpuChannel(voice);
	if(!channel) return -1;
	channel->volumeLeft <<= static_cast<uint16>(left * 4);
	channel->volumeRight <<= static_cast<uint16>(right * 4);
	return 0;
}

uint32 CSasCore::SetSimpleADSR(uint32 contextAddr, uint32 voice, uint32 adsr1, uint32 adsr2)
{
#ifdef _DEBUG
	CLog::GetInstance().Print(LOGNAME, "SetSimpleADSR(contextAddr = 0x%0.8X, voice = %d, adsr1 = 0x%0.4X, adsr2 = 0x%0.4X);\r\n",
	                          contextAddr, voice, adsr1, adsr2);
#endif
	auto channel = GetSpuChannel(voice);
	if(!channel) return -1;
	channel->adsrLevel <<= static_cast<uint16>(adsr1);
	channel->adsrRate <<= static_cast<uint16>(adsr2);
	if((channel->adsrRate.sustainDirection == 1) && (channel->adsrRate.sustainMode == 1))
	{
		//Exp-Dec sustain mode seems to be a bit different on SaS, adjust rate to make sure things sound good
		int32 rate = channel->adsrRate.sustainRate ^ 0x7F;
		rate = std::max(0, rate - 0x10);
		channel->adsrRate.sustainRate = rate ^ 0x7F;
	}
	return 0;
}

uint32 CSasCore::SetKeyOn(uint32 contextAddr, uint32 voice)
{
#ifdef _DEBUG
	CLog::GetInstance().Print(LOGNAME, "SetKeyOn(contextAddr = 0x%0.8X, voice = %d);\r\n", contextAddr, voice);
#endif

	auto chanInfo = TranslateSpuChannel(voice);
	if(!chanInfo.first)
	{
		return -1;
	}
	chanInfo.first->SendKeyOn(1 << chanInfo.second);

	return 0;
}

uint32 CSasCore::SetKeyOff(uint32 contextAddr, uint32 voice)
{
#ifdef _DEBUG
	CLog::GetInstance().Print(LOGNAME, "SetKeyOff(contextAddr = 0x%0.8X, voice = %d);\r\n", contextAddr, voice);
#endif

	auto chanInfo = TranslateSpuChannel(voice);
	if(!chanInfo.first)
	{
		return -1;
	}
	chanInfo.first->SendKeyOff(1 << chanInfo.second);

	return 0;
}

uint32 CSasCore::GetAllEnvelope(uint32 contextAddr, uint32 envelopeAddr)
{
#ifdef _DEBUG
	CLog::GetInstance().Print(LOGNAME, "GetAllEnvelope(contextAddr = 0x%0.8X, envelopeAddr = 0x%0.8X);\r\n", contextAddr, envelopeAddr);
#endif
	assert(envelopeAddr != 0);
	if(envelopeAddr == 0)
	{
		return -1;
	}
	uint32* envelope = reinterpret_cast<uint32*>(m_ram + envelopeAddr);
	for(unsigned int i = 0; i < CHANNEL_COUNT; i++)
	{
		auto channel = GetSpuChannel(i);
		envelope[i] = channel->adsrVolume;
	}
	return 0;
}

uint32 CSasCore::GetPauseFlag(uint32 contextAddr)
{
#ifdef _DEBUG
	CLog::GetInstance().Print(LOGNAME, "GetPauseFlag(contextAddr = 0x%0.8X);\r\n", contextAddr);
#endif
	return 0;
}

uint32 CSasCore::GetEndFlag(uint32 contextAddr)
{
#ifdef _DEBUG
	CLog::GetInstance().Print(LOGNAME, "GetEndFlag(contextAddr = 0x%0.8X);\r\n", contextAddr);
#endif

	uint32 result = 0;
	for(unsigned int i = 0; i < CHANNEL_COUNT; i++)
	{
		const auto* channel = GetSpuChannel(i);
		if(channel->status == Iop::CSpuBase::STOPPED)
		{
			result |= (1 << i);
		}
	}

	return result;
}

uint32 CSasCore::SetEffectType(uint32 contextAddr, uint32 effectType)
{
#ifdef _DEBUG
	CLog::GetInstance().Print(LOGNAME, "SetEffectType(contextAddr = 0x%0.8X, effectType = 0x%0.8X);\r\n",
	                          contextAddr, effectType);
#endif
	switch(effectType)
	{
	case REVERB_STUDIOC:
		SetupReverb(g_ReverbStudioC);
		break;
	case REVERB_HALL:
		SetupReverb(g_ReverbHall);
		break;
	case REVERB_SPACE:
		SetupReverb(g_ReverbSpace);
		break;
	default:
		assert(false);
		break;
	}
	return 0;
}

uint32 CSasCore::SetEffectParam(uint32 contextAddr, uint32 dt, uint32 fb)
{
#ifdef _DEBUG
	CLog::GetInstance().Print(LOGNAME, "SetEffectParam(contextAddr = 0x%0.8X, dt = 0x%0.2X, fb = 0x%0.2X);\r\n",
	                          contextAddr, dt, fb);
#endif
	return 0;
}

uint32 CSasCore::SetEffectVolume(uint32 contextAddr, uint32 volumeLeft, uint32 volumeRight)
{
#ifdef _DEBUG
	CLog::GetInstance().Print(LOGNAME, "SetEffectVolume(contextAddr = 0x%0.8X, left = 0x%0.2X, right = 0x%0.2X);\r\n",
	                          contextAddr, volumeLeft, volumeRight);
#endif
	return 0;
}

uint32 CSasCore::SetEffect(uint32 contextAddr, uint32 drySwitch, uint32 wetSwitch)
{
#ifdef _DEBUG
	CLog::GetInstance().Print(LOGNAME, "SetEffect(contextAddr = 0x%0.8X, dry = %d, wet = %d);\r\n",
	                          contextAddr, drySwitch, wetSwitch);
#endif
	if(drySwitch)
	{
		for(unsigned int spuIdx = 0; spuIdx < 2; spuIdx++)
		{
			m_spu[spuIdx]->SetControl(Iop::CSpuBase::CONTROL_REVERB);
			m_spu[spuIdx]->SetChannelReverbLo(0xFFFF);
			m_spu[spuIdx]->SetChannelReverbHi(0xFFFF);
		}
	}
	return 0;
}

void CSasCore::Invoke(uint32 methodId, CMIPS& context)
{
	switch(methodId)
	{
	case 0x42778A9F:
		context.m_State.nGPR[CMIPS::V0].nV0 = Init(
		    context.m_State.nGPR[CMIPS::A0].nV0,
		    context.m_State.nGPR[CMIPS::A1].nV0,
		    context.m_State.nGPR[CMIPS::A2].nV0,
		    context.m_State.nGPR[CMIPS::A3].nV0,
		    context.m_State.nGPR[CMIPS::T0].nV0);
		break;
	case 0xA3589D81:
		context.m_State.nGPR[CMIPS::V0].nV0 = Core(
		    context.m_State.nGPR[CMIPS::A0].nV0,
		    context.m_State.nGPR[CMIPS::A1].nV0);
		break;
	case 0x99944089:
		context.m_State.nGPR[CMIPS::V0].nV0 = SetVoice(
		    context.m_State.nGPR[CMIPS::A0].nV0,
		    context.m_State.nGPR[CMIPS::A1].nV0,
		    context.m_State.nGPR[CMIPS::A2].nV0,
		    context.m_State.nGPR[CMIPS::A3].nV0,
		    context.m_State.nGPR[CMIPS::T0].nV0);
		break;
	case 0xAD84D37F:
		context.m_State.nGPR[CMIPS::V0].nV0 = SetPitch(
		    context.m_State.nGPR[CMIPS::A0].nV0,
		    context.m_State.nGPR[CMIPS::A1].nV0,
		    context.m_State.nGPR[CMIPS::A2].nV0);
		break;
	case 0x440CA7D8:
		context.m_State.nGPR[CMIPS::V0].nV0 = SetVolume(
		    context.m_State.nGPR[CMIPS::A0].nV0,
		    context.m_State.nGPR[CMIPS::A1].nV0,
		    context.m_State.nGPR[CMIPS::A2].nV0,
		    context.m_State.nGPR[CMIPS::A3].nV0,
		    context.m_State.nGPR[CMIPS::T0].nV0,
		    context.m_State.nGPR[CMIPS::T1].nV0);
		break;
	case 0xCBCD4F79:
		context.m_State.nGPR[CMIPS::V0].nV0 = SetSimpleADSR(
		    context.m_State.nGPR[CMIPS::A0].nV0,
		    context.m_State.nGPR[CMIPS::A1].nV0,
		    context.m_State.nGPR[CMIPS::A2].nV0,
		    context.m_State.nGPR[CMIPS::A3].nV0);
		break;
	case 0x76F01ACA:
		context.m_State.nGPR[CMIPS::V0].nV0 = SetKeyOn(
		    context.m_State.nGPR[CMIPS::A0].nV0,
		    context.m_State.nGPR[CMIPS::A1].nV0);
		break;
	case 0xA0CF2FA4:
		context.m_State.nGPR[CMIPS::V0].nV0 = SetKeyOff(
		    context.m_State.nGPR[CMIPS::A0].nV0,
		    context.m_State.nGPR[CMIPS::A1].nV0);
		break;
	case 0x07F58C24:
		context.m_State.nGPR[CMIPS::V0].nV0 = GetAllEnvelope(
		    context.m_State.nGPR[CMIPS::A0].nV0,
		    context.m_State.nGPR[CMIPS::A1].nV0);
		break;
	case 0x2C8E6AB3:
		context.m_State.nGPR[CMIPS::V0].nV0 = GetPauseFlag(
		    context.m_State.nGPR[CMIPS::A0].nV0);
		break;
	case 0x68A46B95:
		context.m_State.nGPR[CMIPS::V0].nV0 = GetEndFlag(
		    context.m_State.nGPR[CMIPS::A0].nV0);
		break;
	case 0x33D4AB37:
		context.m_State.nGPR[CMIPS::V0].nV0 = SetEffectType(
		    context.m_State.nGPR[CMIPS::A0].nV0,
		    context.m_State.nGPR[CMIPS::A1].nV0);
		break;
	case 0x267A6DD2:
		context.m_State.nGPR[CMIPS::V0].nV0 = SetEffectParam(
		    context.m_State.nGPR[CMIPS::A0].nV0,
		    context.m_State.nGPR[CMIPS::A1].nV0,
		    context.m_State.nGPR[CMIPS::A2].nV0);
		break;
	case 0xD5A229C9:
		context.m_State.nGPR[CMIPS::V0].nV0 = SetEffectVolume(
		    context.m_State.nGPR[CMIPS::A0].nV0,
		    context.m_State.nGPR[CMIPS::A1].nV0,
		    context.m_State.nGPR[CMIPS::A2].nV0);
		break;
	case 0xF983B186:
		context.m_State.nGPR[CMIPS::V0].nV0 = SetEffect(
		    context.m_State.nGPR[CMIPS::A0].nV0,
		    context.m_State.nGPR[CMIPS::A1].nV0,
		    context.m_State.nGPR[CMIPS::A2].nV0);
		break;
	default:
		CLog::GetInstance().Print(LOGNAME, "Unknown function called 0x%0.8X\r\n", methodId);
		break;
	}
}
