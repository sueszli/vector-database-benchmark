// Copyright (C) 2023 Jérôme "Lynix" Leclercq (lynix680@gmail.com)
// This file is part of the "Nazara Engine - Core module"
// For conditions of distribution and use, see copyright notice in Config.hpp

#include <Nazara/Core/AppFilesystemComponent.hpp>
#include <Nazara/Core/VirtualDirectoryFilesystemResolver.hpp>
#include <Nazara/Core/Debug.hpp>

namespace Nz
{
	const VirtualDirectoryPtr& AppFilesystemComponent::Mount(std::string_view name, std::filesystem::path filepath)
	{
		return Mount(name, std::make_shared<VirtualDirectory>(std::make_shared<VirtualDirectoryFilesystemResolver>(std::move(filepath))));
	}

	const VirtualDirectoryPtr& AppFilesystemComponent::Mount(std::string_view name, VirtualDirectoryPtr directory)
	{
		if (name.empty())
		{
			m_rootDirectory = std::move(directory);
			return m_rootDirectory;
		}

		if (!m_rootDirectory)
			m_rootDirectory = std::make_shared<VirtualDirectory>();

		return m_rootDirectory->StoreDirectory(name, std::move(directory)).directory;
	}

	void AppFilesystemComponent::MountDefaultDirectories()
	{
		m_rootDirectory = std::make_shared<VirtualDirectory>(std::make_shared<VirtualDirectoryFilesystemResolver>(std::filesystem::current_path()));
	}
}
