#include <cstring>
#include "FrameDump.h"
#include "states/MemoryStateFile.h"
#include "states/RegisterStateFile.h"

#define STATE_INITIAL_GSRAM "init/gsram"
#define STATE_INITIAL_GSREGS "init/gsregs"
#define STATE_INITIAL_GSPRIVREGS "init/gsprivregs"
#define STATE_PACKET_METADATA_PREFIX "packet_metadata_"
#define STATE_PACKET_REGISTERWRITES_PREFIX "packet_registerwrites_"
#define STATE_PACKET_IMAGEDATA_PREFIX "packet_imagedata_"

#define STATE_PRIVREG_SMODE2 "SMODE2"

CFrameDump::CFrameDump()
{
	m_initialGsRam = new uint8[CGSHandler::RAMSIZE];
	Reset();
}

CFrameDump::~CFrameDump()
{
	delete[] m_initialGsRam;
}

void CFrameDump::Reset()
{
	m_packets.clear();
	memset(m_initialGsRam, 0, CGSHandler::RAMSIZE);
	memset(&m_initialGsRegisters, 0, sizeof(m_initialGsRegisters));
	m_initialSMODE2 = 0;
}

uint8* CFrameDump::GetInitialGsRam()
{
	return m_initialGsRam;
}

uint64* CFrameDump::GetInitialGsRegisters()
{
	return m_initialGsRegisters;
}

uint64 CFrameDump::GetInitialSMODE2() const
{
	return m_initialSMODE2;
}

void CFrameDump::SetInitialSMODE2(uint64 value)
{
	m_initialSMODE2 = value;
}

const CFrameDump::PacketArray& CFrameDump::GetPackets() const
{
	return m_packets;
}

void CFrameDump::AddRegisterPacket(const CGSHandler::RegisterWrite* registerWrites, uint32 count, const CGsPacketMetadata* metadata)
{
	CGsPacket packet;
	packet.registerWrites = CGsPacket::RegisterWriteArray(registerWrites, registerWrites + count);
	if(metadata)
	{
		packet.metadata = *metadata;
	}
	m_packets.push_back(packet);
}

void CFrameDump::AddImagePacket(const uint8* imageData, uint32 size)
{
	CGsPacket packet;
	packet.imageData = CGsPacket::ImageDataArray(imageData, imageData + size);
	m_packets.push_back(packet);
}

void CFrameDump::Read(Framework::CStream& input)
{
	Reset();

	Framework::CZipArchiveReader archive(input);

	archive.BeginReadFile(STATE_INITIAL_GSRAM)->Read(m_initialGsRam, CGSHandler::RAMSIZE);
	archive.BeginReadFile(STATE_INITIAL_GSREGS)->Read(m_initialGsRegisters, sizeof(uint64) * CGSHandler::REGISTER_MAX);

	{
		CRegisterStateFile registerFile(*archive.BeginReadFile(STATE_INITIAL_GSPRIVREGS));
		m_initialSMODE2 = registerFile.GetRegister64(STATE_PRIVREG_SMODE2);
	}

	std::map<unsigned int, std::string> packetFiles;
	for(const auto& fileHeader : archive.GetFileHeaders())
	{
		if(fileHeader.first.find(STATE_PACKET_METADATA_PREFIX) == 0)
		{
			unsigned int packetIdx = 0;
			FRAMEWORK_MAYBE_UNUSED int scanCount = sscanf(fileHeader.first.c_str(), STATE_PACKET_METADATA_PREFIX "%d", &packetIdx);
			assert(scanCount == 1);
			packetFiles[packetIdx] = fileHeader.first;
		}
	}

	for(const auto& packetFilePair : packetFiles)
	{
		const auto& packetMetadataFileName = packetFilePair.second;
		auto packetRegisterWritesFileName = STATE_PACKET_REGISTERWRITES_PREFIX + std::to_string(packetFilePair.first);
		auto packetImageDataFileName = STATE_PACKET_IMAGEDATA_PREFIX + std::to_string(packetFilePair.first);

		CGsPacket packet;

		//Read metadata (mandatory)
		archive.BeginReadFile(packetMetadataFileName.c_str())->Read(&packet.metadata, sizeof(CGsPacketMetadata));

		//Read register writes
		if(const auto& packetRegisterWritesFileHeader = archive.GetFileHeader(packetRegisterWritesFileName.c_str()))
		{
			unsigned int writeCount = packetRegisterWritesFileHeader->uncompressedSize / sizeof(CGSHandler::RegisterWrite);
			assert(packetRegisterWritesFileHeader->uncompressedSize % sizeof(CGSHandler::RegisterWrite) == 0);
			packet.registerWrites.resize(writeCount);
			archive.BeginReadFile(packetRegisterWritesFileName.c_str())->Read(packet.registerWrites.data(), packet.registerWrites.size() * sizeof(CGSHandler::RegisterWrite));
		}

		//Read image data
		if(const auto& packetImageDataFileHeader = archive.GetFileHeader(packetImageDataFileName.c_str()))
		{
			unsigned int imageDataSize = packetImageDataFileHeader->uncompressedSize;
			packet.imageData.resize(imageDataSize);
			archive.BeginReadFile(packetImageDataFileName.c_str())->Read(packet.imageData.data(), packet.imageData.size());
		}

		m_packets.push_back(packet);
	}
}

void CFrameDump::Write(Framework::CStream& output) const
{
	Framework::CZipArchiveWriter archive;

	archive.InsertFile(std::make_unique<CMemoryStateFile>(STATE_INITIAL_GSRAM, m_initialGsRam, CGSHandler::RAMSIZE));
	archive.InsertFile(std::make_unique<CMemoryStateFile>(STATE_INITIAL_GSREGS, m_initialGsRegisters, sizeof(uint64) * CGSHandler::REGISTER_MAX));

	{
		auto privRegsStateFile = std::make_unique<CRegisterStateFile>(STATE_INITIAL_GSPRIVREGS);
		privRegsStateFile->SetRegister64(STATE_PRIVREG_SMODE2, m_initialSMODE2);
		archive.InsertFile(std::move(privRegsStateFile));
	}

	unsigned int currentPacket = 0;
	for(const auto& packet : m_packets)
	{
		auto packetMetadataFileName = STATE_PACKET_METADATA_PREFIX + std::to_string(currentPacket);
		archive.InsertFile(std::make_unique<CMemoryStateFile>(packetMetadataFileName.c_str(), &packet.metadata, sizeof(CGsPacketMetadata)));
		if(!packet.registerWrites.empty())
		{
			auto packetRegisterWritesFileName = STATE_PACKET_REGISTERWRITES_PREFIX + std::to_string(currentPacket);
			archive.InsertFile(std::make_unique<CMemoryStateFile>(packetRegisterWritesFileName.c_str(), packet.registerWrites.data(), packet.registerWrites.size() * sizeof(CGSHandler::RegisterWrite)));
		}
		if(!packet.imageData.empty())
		{
			auto packetImageDataName = STATE_PACKET_IMAGEDATA_PREFIX + std::to_string(currentPacket);
			archive.InsertFile(std::make_unique<CMemoryStateFile>(packetImageDataName.c_str(), packet.imageData.data(), packet.imageData.size()));
		}
		currentPacket++;
	}

	archive.Write(output);
}

void CFrameDump::IdentifyDrawingKicks()
{
	m_drawingKicks.clear();

	DRAWINGKICK_INFO drawingKickInfo;

	static const unsigned int g_initVertexCounts[8] = {1, 2, 2, 3, 3, 3, 2, 0};
	static const unsigned int g_nextVertexCounts[8] = {1, 2, 1, 3, 1, 1, 2, 0};

	CGSHandler::PRIM currentPrim;
	currentPrim <<= GetInitialGsRegisters()[GS_REG_PRIM];

	CGSHandler::XYOFFSET currentOfs[2];
	currentOfs[0] <<= GetInitialGsRegisters()[GS_REG_XYOFFSET_1];
	currentOfs[1] <<= GetInitialGsRegisters()[GS_REG_XYOFFSET_2];

	unsigned int vertexCount = g_initVertexCounts[currentPrim.nType];

	uint32 cmdIndex = 0;
	for(const auto& packet : GetPackets())
	{
		for(const auto& registerWrite : packet.registerWrites)
		{
			if(registerWrite.first == GS_REG_PRIM)
			{
				currentPrim <<= registerWrite.second;
				vertexCount = g_initVertexCounts[currentPrim.nType];
			}
			else if(
			    (registerWrite.first == GS_REG_XYOFFSET_1) ||
			    (registerWrite.first == GS_REG_XYOFFSET_2))
			{
				currentOfs[registerWrite.first - GS_REG_XYOFFSET_1] <<= registerWrite.second;
			}
			else if(
			    (registerWrite.first == GS_REG_XYZ2) ||
			    (registerWrite.first == GS_REG_XYZ3) ||
			    (registerWrite.first == GS_REG_XYZF2) ||
			    (registerWrite.first == GS_REG_XYZF3))
			{
				if(vertexCount != 0)
				{
					vertexCount--;

					const auto& offset = currentOfs[currentPrim.nContext];

					drawingKickInfo.primType = currentPrim.nType;
					drawingKickInfo.context = currentPrim.nContext;
					drawingKickInfo.vertex[vertexCount].x = ((registerWrite.second >> 0) & 0xFFFF) - offset.nOffsetX;
					drawingKickInfo.vertex[vertexCount].y = ((registerWrite.second >> 16) & 0xFFFF) - offset.nOffsetY;

					if(vertexCount == 0)
					{
						bool drawingKick = (registerWrite.first == GS_REG_XYZ2) || (registerWrite.first == GS_REG_XYZF2);
						if(drawingKick)
						{
							m_drawingKicks.insert(std::make_pair(cmdIndex, drawingKickInfo));
						}
						vertexCount = g_nextVertexCounts[currentPrim.nType];
						switch(currentPrim.nType)
						{
						case CGSHandler::PRIM_LINESTRIP:
							memcpy(&drawingKickInfo.vertex[1], &drawingKickInfo.vertex[0], sizeof(DRAWINGKICK_INFO::VERTEX));
							break;
						case CGSHandler::PRIM_TRIANGLESTRIP:
							memcpy(&drawingKickInfo.vertex[2], &drawingKickInfo.vertex[1], sizeof(DRAWINGKICK_INFO::VERTEX));
							memcpy(&drawingKickInfo.vertex[1], &drawingKickInfo.vertex[0], sizeof(DRAWINGKICK_INFO::VERTEX));
							break;
						case CGSHandler::PRIM_TRIANGLEFAN:
							memcpy(&drawingKickInfo.vertex[1], &drawingKickInfo.vertex[0], sizeof(DRAWINGKICK_INFO::VERTEX));
							break;
						}
					}
				}
			}

			cmdIndex++;
		}
	}
}

const DrawingKickInfoMap& CFrameDump::GetDrawingKicks() const
{
	return m_drawingKicks;
}
