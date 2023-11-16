// Copyright (C) 2023 Alexandre Janniaux
// This file is part of the "Nazara Engine - Core module"
// For conditions of distribution and use, see copyright notice in Config.hpp

#include <Nazara/Core/Posix/DynLibImpl.hpp>
#include <NazaraUtils/Algorithm.hpp>
#include <dlfcn.h>
#include <cstring>
#include <Nazara/Core/Debug.hpp>

namespace Nz
{
	DynLibImpl::~DynLibImpl()
	{
		if (m_handle)
			dlclose(m_handle);
	}

	DynLibFunc DynLibImpl::GetSymbol(const char* symbol, std::string* errorMessage) const
	{
		dlerror(); // Clear error flag

		void* ptr = dlsym(m_handle, symbol);
		if (!ptr)
			*errorMessage = dlerror();

		static_assert(sizeof(DynLibFunc) == sizeof(void*));

		// poor man's std::bit_cast
		DynLibFunc funcPtr;
		std::memcpy(&funcPtr, &ptr, sizeof(funcPtr));

		return funcPtr;
	}

	bool DynLibImpl::Load(const std::filesystem::path& libraryPath, std::string* errorMessage)
	{
		dlerror(); // Clear error flag
		m_handle = dlopen(Nz::PathToString(libraryPath).data(), RTLD_LAZY | RTLD_GLOBAL);

		if (!m_handle)
		{
			*errorMessage = dlerror();
			return false;
		}

		return true;
	}
}
