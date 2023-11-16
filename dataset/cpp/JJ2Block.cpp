﻿#include "JJ2Block.h"

#include "../../nCine/IO/CompressionUtils.h"

#include <cstring>

using namespace nCine;

namespace Jazz2::Compatibility
{
	JJ2Block::JJ2Block(const std::unique_ptr<Stream>& s, int32_t length, int32_t uncompressedLength)
		: _length(0), _offset(0)
	{
		std::unique_ptr<uint8_t[]> tmpBuffer = std::make_unique<uint8_t[]>(length);
		s->Read(tmpBuffer.get(), length);

		if (uncompressedLength > 0) {
			_length -= 2;
			_buffer = std::make_unique<uint8_t[]>(uncompressedLength);
			auto result = CompressionUtils::Inflate(tmpBuffer.get() + 2, length, _buffer.get(), uncompressedLength);
			_length = (result == DecompressionResult::Success ? uncompressedLength : 0);
		} else {
			_buffer = std::move(tmpBuffer);
			_length = length;
		}
	}

	void JJ2Block::SeekTo(int32_t offset)
	{
		_offset = offset;
	}

	void JJ2Block::DiscardBytes(int32_t length)
	{
		_offset += length;
	}

	bool JJ2Block::ReadBool()
	{
		if (_offset >= _length) {
			_offset = INT32_MAX;
			return false;
		}

		return _buffer[_offset++] != 0x00;
	}

	uint8_t JJ2Block::ReadByte()
	{
		if (_offset >= _length) {
			_offset = INT32_MAX;
			return 0;
		}

		return _buffer[_offset++];
	}

	int16_t JJ2Block::ReadInt16()
	{
		if (_offset > _length - 2) {
			_offset = INT32_MAX;
			return false;
		}

		int16_t result = *(int16_t*)&_buffer[_offset];
		_offset += 2;
		return result;
	}

	uint16_t JJ2Block::ReadUInt16()
	{
		if (_offset > _length - 2) {
			_offset = INT32_MAX;
			return false;
		}

		uint16_t result = *(uint16_t*)&_buffer[_offset];
		_offset += 2;
		return result;
	}

	int32_t JJ2Block::ReadInt32()
	{
		if (_offset > _length - 4) {
			_offset = INT32_MAX;
			return false;
		}
		//int32_t result = *(int32_t*)&_buffer[_offset];
		int32_t result = (int32_t)(_buffer[_offset] | (_buffer[_offset + 1] << 8) | (_buffer[_offset + 2] << 16) | (_buffer[_offset + 3] << 24));
		_offset += 4;
		return result;
	}

	uint32_t JJ2Block::ReadUInt32()
	{
		if (_offset > _length - 4) {
			_offset = INT32_MAX;
			return false;
		}

		//uint32_t result = *(uint32_t*)&_buffer[_offset];
		uint32_t result = _buffer[_offset] | (_buffer[_offset + 1] << 8) | (_buffer[_offset + 2] << 16) | (_buffer[_offset + 3] << 24);
		_offset += 4;
		return result;
	}


	int32_t JJ2Block::ReadUint7bitEncoded()
	{
		int result = 0;

		while (true) {
			if (_offset >= _length) {
				_offset = INT32_MAX;
				break;
			}

			uint8_t current = _buffer[_offset++];
			result |= (current & 0x7F);
			if (current >= 0x80) {
				result <<= 7;
			} else {
				break;
			}
		}

		return result;
	}

	float JJ2Block::ReadFloat()
	{
		if (_offset > _length - 4) {
			_offset = INT32_MAX;
			return false;
		}

		float result = *(float*)&_buffer[_offset];
		_offset += 4;
		return result;
	}

	float JJ2Block::ReadFloatEncoded()
	{
		return ((float)ReadInt32() / 65536.0f);
	}

	void JJ2Block::ReadRawBytes(uint8_t* dst, int32_t length)
	{
		int bytesLeft = _length - _offset;
		bool endOfStream = false;
		if (length > bytesLeft) {
			length = bytesLeft;
			endOfStream = true;
		}

		if (length > 0) {
			std::memcpy(dst, _buffer.get() + _offset, length);
		}

		if (endOfStream) {
			_offset = INT32_MAX;
		} else {
			_offset += length;
		}
	}

	StringView JJ2Block::ReadString(int32_t length, bool trimToNull)
	{
		int bytesLeft = _length - _offset;
		bool endOfStream = false;
		if (length > bytesLeft) {
			length = bytesLeft;
			endOfStream = true;
		}

		int realLength = length;
		if (trimToNull) {
			for (int i = 0; i < realLength; i++) {
				if (_buffer[_offset + i] == '\0') {
					realLength = i;
					break;
				}
			}
		} else {
			while (realLength > 0 && (_buffer[_offset + realLength - 1] == '\0' || _buffer[_offset + realLength - 1] == ' ')) {
				realLength--;
			}
		}

		StringView result((char*)&_buffer[_offset], realLength);

		if (endOfStream) {
			_offset = INT32_MAX;
		} else {
			_offset += length;
		}

		return result;
	}
}