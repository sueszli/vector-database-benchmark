#include <algorithm>
#include "AppConfig.h"
#include "BootablesProcesses.h"
#include "BootablesDbClient.h"
#include "TheGamesDbClient.h"
#include "DiskUtils.h"
#include "PathUtils.h"
#include "StringUtils.h"
#include "string_format.h"
#include "StdStreamUtils.h"
#include "http/HttpClientFactory.h"
#ifdef __ANDROID__
#include "android/ContentUtils.h"
#endif

//Jobs
// Scan for new games (from input directory)
// Remove games that might not be available anymore
// Extract game ids from disk images
// Pull disc cover URLs and titles from GamesDb/TheGamesDb

//#define SCAN_LOG

static void BootableLog(const char* format, ...)
{
#ifdef SCAN_LOG
	static FILE* logStream = nullptr;
	if(!logStream)
	{
		auto logPath = CAppConfig::GetBasePath() / "bootables.log";
		logStream = fopen(logPath.string().c_str(), "wb");
	}
	va_list args;
	va_start(args, format);
	vfprintf(logStream, format, args);
	va_end(args);
	fflush(logStream);
#endif
}

bool IsBootableExecutablePath(const fs::path& filePath)
{
	auto extension = StringUtils::ToLower(filePath.extension().string());
	return (extension == ".elf");
}

bool IsBootableDiscImagePath(const fs::path& filePath)
{
	const auto& supportedExtensions = DiskUtils::GetSupportedExtensions();
	auto extension = StringUtils::ToLower(filePath.extension().string());
	auto extensionIterator = supportedExtensions.find(extension);
	return extensionIterator != std::end(supportedExtensions);
}

bool IsBootableArcadeDefPath(const fs::path& filePath)
{
	auto extension = filePath.extension().string();
	return (extension == ".arcadedef");
}

bool DoesBootableExist(const fs::path& filePath)
{
	//TODO: Properly support S3 paths. Also, beware when implementing this because Android
	//      might complain about network access being done on the main thread.
	static const char* s3ImagePathPrefix = "//s3/";
	if(filePath.string().find(s3ImagePathPrefix) == 0) return true;
#ifdef __ANDROID__
	if(Framework::Android::CContentUtils::IsContentPath(filePath))
	{
		return Framework::Android::CContentUtils::DoesFileExist(filePath);
	}
#endif
	return fs::exists(filePath);
}

bool TryRegisterBootable(const fs::path& path)
{
	try
	{
		std::string serial;
		if(
		    !BootablesDb::CClient::GetInstance().BootableExists(path) &&
		    !IsBootableExecutablePath(path) &&
		    !(IsBootableDiscImagePath(path) && DiskUtils::TryGetDiskId(path, &serial)) &&
		    !IsBootableArcadeDefPath(path))
		{
			return false;
		}
		BootablesDb::CClient::GetInstance().RegisterBootable(path, path.filename().string().c_str(), serial.c_str());
		return true;
	}
	catch(...)
	{
		return false;
	}
}

bool TryUpdateLastBootedTime(const fs::path& path)
{
	try
	{
		BootablesDb::CClient::GetInstance().SetLastBootedTime(path, std::time(nullptr));
		return true;
	}
	catch(...)
	{
		return false;
	}
}

void ScanBootables(const fs::path& parentPath, bool recursive)
{
	BootableLog("Entering ScanBootables(path = '%s', recursive = %d);\r\n",
	            parentPath.string().c_str(), static_cast<int>(recursive));
	try
	{
		std::error_code ec;
		for(auto pathIterator = fs::directory_iterator(parentPath, ec);
		    pathIterator != fs::directory_iterator(); pathIterator.increment(ec))
		{
			auto& path = pathIterator->path();
			BootableLog("Checking '%s'... ", path.string().c_str());
			try
			{
				if(ec)
				{
					BootableLog(" failed to get status: %s.\r\n", ec.message().c_str());
					continue;
				}
				if(recursive && fs::is_directory(path))
				{
					BootableLog("is directory.\r\n");
					ScanBootables(path, recursive);
					continue;
				}
				BootableLog("registering... ");
				bool success = TryRegisterBootable(path);
				BootableLog("result = %d\r\n", static_cast<int>(success));
			}
			catch(const std::exception& exception)
			{
				//Failed to process a path, keep going
				BootableLog(" exception: %s\r\n", exception.what());
			}
		}
	}
	catch(const std::exception& exception)
	{
		BootableLog("Caught an exception while trying to list directory: %s\r\n", exception.what());
	}
	BootableLog("Exiting ScanBootables(path = '%s', recursive = %d);\r\n",
	            parentPath.string().c_str(), static_cast<int>(recursive));
}

std::set<fs::path> GetActiveBootableDirectories()
{
	std::set<fs::path> result;
	auto bootables = BootablesDb::CClient::GetInstance().GetBootables();
	for(const auto& bootable : bootables)
	{
		auto parentPath = bootable.path.parent_path();
		static const char* s3ImagePathPrefix = "//s3/";
		if(parentPath.string().find(s3ImagePathPrefix) == std::string::npos)
			result.insert(parentPath);
	}
	return result;
}

void PurgeInexistingFiles()
{
	auto bootables = BootablesDb::CClient::GetInstance().GetBootables();
	for(const auto& bootable : bootables)
	{
		if(DoesBootableExist(bootable.path)) continue;
		BootablesDb::CClient::GetInstance().UnregisterBootable(bootable.path);
	}
}

void FetchGameTitles()
{
	auto bootables = BootablesDb::CClient::GetInstance().GetBootables();
	std::vector<std::string> serials;
	for(const auto& bootable : bootables)
	{
		if(bootable.discId.empty()) continue;

		if(bootable.coverUrl.empty() || bootable.title.empty() || bootable.overview.empty())
		{
			serials.push_back(bootable.discId);
		}
	}

	if(serials.empty()) return;

	BootableLog("Fetching info for %d games.\r\n", serials.size());

	try
	{
		auto gamesList = TheGamesDb::CClient::GetInstance().GetGames(serials);
		BootableLog("Received info for %d games.\r\n", gamesList.size());

		for(auto& game : gamesList)
		{
			for(const auto& bootable : bootables)
			{
				for(const auto& discId : game.discIds)
				{
					if(discId == bootable.discId)
					{
						BootableLog("Setting info for '%s'...\r\n", bootable.discId.c_str());

						BootablesDb::CClient::GetInstance().SetTitle(bootable.path, game.title.c_str());

						if(!game.overview.empty())
						{
							BootablesDb::CClient::GetInstance().SetOverview(bootable.path, game.overview.c_str());
						}
						if(!game.boxArtUrl.empty())
						{
							auto coverUrl = string_format("%s%s", game.baseImgUrl.c_str(), game.boxArtUrl.c_str());
							BootablesDb::CClient::GetInstance().SetCoverUrl(bootable.path, coverUrl.c_str());
						}

						break;
					}
				}
			}
		}
	}
	catch(const std::exception& exception)
	{
		BootableLog("Caught an exception while trying to fetch titles: %s\r\n", exception.what());
	}
}

void FetchGameCovers()
{
	auto coverpath(CAppConfig::GetInstance().GetBasePath() / fs::path("covers"));
	Framework::PathUtils::EnsurePathExists(coverpath);

	auto bootables = BootablesDb::CClient::GetInstance().GetBootables();
	std::vector<std::string> serials;
	for(const auto& bootable : bootables)
	{
		if(bootable.discId.empty())
			continue;

		BootableLog("Checking cover for '%s'...\r\n", bootable.discId.c_str());

		if(bootable.coverUrl.empty())
		{
			BootableLog("Bootable has no cover URL, skipping.\r\n");
			continue;
		}

		try
		{
			auto path = coverpath / (bootable.discId + ".jpg");

			BootableLog("Looking for '%s'... ", path.string().c_str());
			if(fs::exists(path))
			{
				BootableLog("Already exists, skipping.\r\n");
				continue;
			}
			BootableLog("Doesn't exist.\r\n");
			BootableLog("Downloading from '%s'...\r\n", bootable.coverUrl.c_str());

			auto requestResult =
			    [&]() {
				    auto client = Framework::Http::CreateHttpClient();
				    client->SetUrl(bootable.coverUrl);
				    return client->SendRequest();
			    }();

			BootableLog("Download yielded result %d.\r\n", requestResult.statusCode);
			if(requestResult.statusCode == Framework::Http::HTTP_STATUS_CODE::OK)
			{
				auto outputStream = Framework::CreateOutputStdStream(path.native());
				outputStream.Write(requestResult.data.GetBuffer(), requestResult.data.GetSize());
				BootableLog("Saved cover to disk.\r\n");
			}
		}
		catch(const std::exception& exception)
		{
			BootableLog("Caught an exception while trying to process cover: %s\r\n", exception.what());
		}
	}
}
