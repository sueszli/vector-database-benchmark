#include "utilityCxxHeaderDetection.h"

#include <QSettings>
#include <QSysInfo>

#include "FileSystem.h"
#include "utilityApp.h"
#include "utilityString.h"

namespace utility
{
std::vector<std::wstring> getCxxHeaderPaths(const std::string& compilerName)
{
	std::vector<std::wstring> paths;

	const utility::ProcessOutput out = utility::executeProcess(
		utility::decodeFromUtf8(compilerName), {L"-x", L"c++", L"-v", L"-E", L"/dev/null"});
	if (out.exitCode == 0)
	{
		std::wstring standardHeaders = utility::substrBetween<std::wstring>(
			out.output, L"#include <...> search starts here:\n", L"\nEnd of search list");

		if (!standardHeaders.empty())
		{
			for (const std::wstring& s: utility::splitToVector(standardHeaders, L'\n'))
			{
				paths.push_back(utility::trim(s));
			}
		}
	}

	return paths;
}

std::vector<FilePath> getWindowsSdkHeaderSearchPaths(ApplicationArchitectureType architectureType)
{
	std::vector<FilePath> headerSearchPaths;

	std::vector<std::string> windowsSdkVersions;
	windowsSdkVersions.push_back("v8.1A");
	windowsSdkVersions.push_back("v8.1");
	windowsSdkVersions.push_back("v8.0A");
	windowsSdkVersions.push_back("v7.1A");
	windowsSdkVersions.push_back("v7.0A");

	for (size_t i = 0; i < windowsSdkVersions.size(); i++)
	{
		const FilePath sdkPath = getWindowsSdkRootPathUsingRegistry(
			architectureType, windowsSdkVersions[i]);
		if (sdkPath.exists())
		{
			const FilePath sdkIncludePath = sdkPath.getConcatenated(L"include/");
			if (sdkIncludePath.exists())
			{
				bool usingSubdirectories = false;
				for (const std::wstring& subDirectory: {L"shared", L"um", L"winrt"})
				{
					const FilePath sdkSubdirectory = sdkIncludePath.getConcatenated(subDirectory);
					if (sdkSubdirectory.exists())
					{
						headerSearchPaths.push_back(sdkSubdirectory);
						usingSubdirectories = true;
					}
				}

				if (!usingSubdirectories)
				{
					headerSearchPaths.push_back(sdkIncludePath);
				}
				break;
			}
		}
	}
	{
		const FilePath sdkPath = getWindowsSdkRootPathUsingRegistry(architectureType, "v10.0");
		if (sdkPath.exists())
		{
			for (const FilePath& versionPath:
				 FileSystem::getDirectSubDirectories(sdkPath.getConcatenated(L"include/")))
			{
				const FilePath ucrtPath = versionPath.getConcatenated(L"ucrt");
				if (ucrtPath.exists())
				{
					headerSearchPaths.push_back(ucrtPath);
					break;
				}
			}
		}
	}

	return headerSearchPaths;
}

FilePath getWindowsSdkRootPathUsingRegistry(
	ApplicationArchitectureType architectureType, const std::string& sdkVersion)
{
	QString key = QStringLiteral("HKEY_LOCAL_MACHINE\\SOFTWARE\\");
	if (architectureType == APPLICATION_ARCHITECTURE_X86_32)
	{
		key += QStringLiteral("Wow6432Node\\");
	}
	key += QStringLiteral("Microsoft\\Microsoft SDKs\\Windows\\") + sdkVersion.c_str();

	QSettings expressKey(
		key, QSettings::NativeFormat);	  // NativeFormat means from Registry on Windows.
	QString value = expressKey.value(QStringLiteral("InstallationFolder")).toString();

	FilePath path(value.toStdWString());
	if (path.exists())
	{
		return path;
	}

	return FilePath();
}
}	 // namespace utility
