#include <iostream>

#include "AppConfig.h"
#include "CoverUtils.h"
#include "PathUtils.h"
#include "QStringUtils.h"

std::map<std::string, QPixmap> CoverUtils::cache;
std::mutex CoverUtils::m_lock;

QPixmap CoverUtils::find(std::string key)
{
	auto itr = CoverUtils::cache.find(key);
	if(itr == CoverUtils::cache.end())
	{
		return QPixmap();
	}
	return itr->second;
}

void CoverUtils::PopulatePlaceholderCover()
{
	auto itr = CoverUtils::cache.find("PH");
	if(itr == CoverUtils::cache.end())
	{
		auto pixmap = QPixmap(QString(":/assets/boxart.png")).scaledToWidth(250 / 2, Qt::SmoothTransformation);
		CoverUtils::cache.insert(std::make_pair("PH", pixmap));
	}
}

void CoverUtils::PopulateCache(std::vector<BootablesDb::Bootable> bootables)
{
	m_lock.lock();
	PopulatePlaceholderCover();

	auto coverpath(CAppConfig::GetInstance().GetBasePath() / fs::path("covers"));
	Framework::PathUtils::EnsurePathExists(coverpath);

	auto itr = CoverUtils::cache.find("PH");
	auto placeholder_size = itr->second.size();
	for(auto bootable : bootables)
	{
		if(bootable.discId.empty())
			continue;

		auto path = coverpath / (bootable.discId + ".jpg");
		try
		{
			if(fs::exists(path))
			{
				auto itr = CoverUtils::cache.find(bootable.discId.c_str());
				if(itr == CoverUtils::cache.end())
				{
					auto pixmap = QPixmap();
					if(!pixmap.load(PathToQString(path)))
					{
						pixmap.load(PathToQString(path), "png");
					}
					pixmap = pixmap.scaled(placeholder_size, Qt::IgnoreAspectRatio, Qt::SmoothTransformation);
					CoverUtils::cache.insert(std::make_pair(bootable.discId.c_str(), pixmap));
				}
			}
		}
		catch(const std::exception& ex)
		{
			//Ignore
		}
	}
	m_lock.unlock();
}
