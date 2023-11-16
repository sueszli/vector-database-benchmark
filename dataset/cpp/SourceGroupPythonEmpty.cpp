#include "SourceGroupPythonEmpty.h"

#include "ApplicationSettings.h"
#include "FileManager.h"
#include "IndexerCommandCustom.h"
#include "ProjectSettings.h"
#include "RefreshInfo.h"
#include "ResourcePaths.h"
#include "SourceGroupSettingsPythonEmpty.h"
#include "SqliteIndexStorage.h"
#include "utility.h"

SourceGroupPythonEmpty::SourceGroupPythonEmpty(std::shared_ptr<SourceGroupSettingsPythonEmpty> settings)
	: m_settings(settings)
{
}

bool SourceGroupPythonEmpty::allowsPartialClearing() const
{
	return false;
}

bool SourceGroupPythonEmpty::allowsShallowIndexing() const
{
	return true;
}


std::set<FilePath> SourceGroupPythonEmpty::filterToContainedFilePaths(
	const std::set<FilePath>& filePaths) const
{
	return SourceGroup::filterToContainedFilePaths(
		filePaths,
		std::set<FilePath>(),
		utility::toSet(m_settings->getSourcePathsExpandedAndAbsolute()),
		m_settings->getExcludeFiltersExpandedAndAbsolute());
}

std::set<FilePath> SourceGroupPythonEmpty::getAllSourceFilePaths() const
{
	FileManager fileManager;
	fileManager.update(
		m_settings->getSourcePathsExpandedAndAbsolute(),
		m_settings->getExcludeFiltersExpandedAndAbsolute(),
		m_settings->getSourceExtensions());
	return fileManager.getAllSourceFilePaths();
}

std::vector<std::shared_ptr<IndexerCommand>> SourceGroupPythonEmpty::getIndexerCommands(
	const RefreshInfo& info) const
{
	std::vector<std::wstring> args;
	args.push_back(L"index");

	args.push_back(L"--source-file-path");
	args.push_back(L"%{SOURCE_FILE_PATH}");
	args.push_back(L"--database-file-path");
	args.push_back(L"%{DATABASE_FILE_PATH}");

	if (!m_settings->getEnvironmentPath().empty())
	{
		args.push_back(L"--environment-path");
		args.push_back(m_settings->getEnvironmentPathExpandedAndAbsolute().wstr());
	}

	if (ApplicationSettings::getInstance()->getVerboseIndexerLoggingEnabled())
	{
		args.push_back(L"--verbose");
	}

	if (info.shallow)
	{
		args.push_back(L"--shallow");
	}

	std::vector<std::shared_ptr<IndexerCommand>> indexerCommands;
	for (const FilePath& sourceFilePath: getAllSourceFilePaths())
	{
		if (info.filesToIndex.find(sourceFilePath) != info.filesToIndex.end())
		{
			indexerCommands.push_back(std::make_shared<IndexerCommandCustom>(
				INDEXER_COMMAND_PYTHON,
				ResourcePaths::getPythonIndexerFilePath().wstr(),
				args,
				m_settings->getProjectSettings()->getProjectFilePath(),
				m_settings->getProjectSettings()->getTempDBFilePath(),
				std::to_wstring(SqliteIndexStorage::getStorageVersion()),
				sourceFilePath,
				true));
		}
	}

	return indexerCommands;
}

std::shared_ptr<SourceGroupSettings> SourceGroupPythonEmpty::getSourceGroupSettings()
{
	return m_settings;
}

std::shared_ptr<const SourceGroupSettings> SourceGroupPythonEmpty::getSourceGroupSettings() const
{
	return m_settings;
}
