// Copyright (C) 2023 Jérôme "Lynix" Leclercq (lynix680@gmail.com)
// This file is part of the "Nazara Engine - Core module"
// For conditions of distribution and use, see copyright notice in Config.hpp

#pragma once

#ifndef NAZARA_CORE_VIRTUALDIRECTORYFILESYSTEMRESOLVER_HPP
#define NAZARA_CORE_VIRTUALDIRECTORYFILESYSTEMRESOLVER_HPP

#include <NazaraUtils/Prerequisites.hpp>
#include <Nazara/Core/Config.hpp>
#include <Nazara/Core/VirtualDirectory.hpp>
#include <filesystem>

namespace Nz
{
	class NAZARA_CORE_API VirtualDirectoryFilesystemResolver : public VirtualDirectoryResolver
	{
		public:
			inline VirtualDirectoryFilesystemResolver(std::filesystem::path physicalPath, OpenModeFlags fileOpenMode = OpenMode::ReadOnly | OpenMode::Defer);
			VirtualDirectoryFilesystemResolver(const VirtualDirectoryFilesystemResolver&) = delete;
			VirtualDirectoryFilesystemResolver(VirtualDirectoryFilesystemResolver&&) = delete;
			~VirtualDirectoryFilesystemResolver() = default;

			void ForEach(std::weak_ptr<VirtualDirectory> parent, FunctionRef<bool(std::string_view name, VirtualDirectory::Entry&& entry)> callback) const override;

			std::optional<VirtualDirectory::Entry> Resolve(std::weak_ptr<VirtualDirectory> parent, const std::string_view* parts, std::size_t partCount) const override;

			VirtualDirectoryFilesystemResolver& operator=(const VirtualDirectoryFilesystemResolver&) = delete;
			VirtualDirectoryFilesystemResolver& operator=(VirtualDirectoryFilesystemResolver&&) = delete;

		private:
			std::filesystem::path m_physicalPath;
			OpenModeFlags m_fileOpenMode;
	};
}

#include <Nazara/Core/VirtualDirectoryFilesystemResolver.inl>

#endif // NAZARA_CORE_VIRTUALDIRECTORYFILESYSTEMRESOLVER_HPP
