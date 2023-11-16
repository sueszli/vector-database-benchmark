// Copyright (C) 2023 Jérôme "Lynix" Leclercq (lynix680@gmail.com)
// This file is part of the "Nazara Engine - Core module"
// For conditions of distribution and use, see copyright notice in Config.hpp

#include <Nazara/Core/Hash/SHA512.hpp>
#include <Nazara/Core/Hash/SHA/Internal.hpp>
#include <Nazara/Core/Debug.hpp>

namespace Nz
{
	SHA512Hasher::SHA512Hasher()
	{
		m_state = new SHA_CTX;
	}

	SHA512Hasher::~SHA512Hasher()
	{
		delete m_state;
	}

	void SHA512Hasher::Append(const UInt8* data, std::size_t len)
	{
		SHA512_Update(m_state, data, len);
	}

	void SHA512Hasher::Begin()
	{
		SHA512_Init(m_state);
	}

	ByteArray SHA512Hasher::End()
	{
		UInt8 digest[SHA512_DIGEST_LENGTH];

		SHA512_End(m_state, digest);

		return ByteArray(digest, SHA512_DIGEST_LENGTH);
	}

	std::size_t SHA512Hasher::GetDigestLength() const
	{
		return SHA512_DIGEST_LENGTH;
	}

	const char* SHA512Hasher::GetHashName() const
	{
		return "SHA512";
	}
}
