#include "SourceGroupJavaEmpty.h"

#include "SourceGroupSettingsJavaEmpty.h"
#include "utilityJava.h"

SourceGroupJavaEmpty::SourceGroupJavaEmpty(std::shared_ptr<SourceGroupSettingsJavaEmpty> settings)
	: m_settings(settings)
{
}

bool SourceGroupJavaEmpty::prepareIndexing()
{
	if (!utility::prepareJavaEnvironmentAndDisplayOccurringErrors())
	{
		return false;
	}
	return true;
}

std::vector<FilePath> SourceGroupJavaEmpty::getAllSourcePaths() const
{
	return m_settings->getSourcePathsExpandedAndAbsolute();
}

std::vector<FilePath> SourceGroupJavaEmpty::doGetClassPath() const
{
	return utility::getClassPath(
		m_settings->getClasspathExpandedAndAbsolute(),
		m_settings->getUseJreSystemLibrary(),
		getAllSourceFilePaths());
}

std::shared_ptr<SourceGroupSettings> SourceGroupJavaEmpty::getSourceGroupSettings()
{
	return m_settings;
}

std::shared_ptr<const SourceGroupSettings> SourceGroupJavaEmpty::getSourceGroupSettings() const
{
	return m_settings;
}
