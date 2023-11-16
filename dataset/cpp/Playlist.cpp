#include "StdStream.h"
#include "path_uncomplete.h"
#include "string_cast.h"
#include "xml/Parser.h"
#include "xml/Writer.h"
#include "xml/Utils.h"
#include "Utf8.h"
#include "Playlist.h"
#include "stricmp.h"
#include "StdStreamUtils.h"

#define PLAYLIST_NODE_TAG "Playlist"
#define PLAYLIST_ITEM_NODE_TAG "Item"
#define PLAYLIST_ITEM_PATH_ATTRIBUTE ("Path")
#define PLAYLIST_ITEM_TITLE_ATTRIBUTE ("Title")
#define PLAYLIST_ITEM_LENGTH_ATTRIBUTE ("Length")

// clang-format off
const char* CPlaylist::g_loadableExtensions[] =
{
	"psf",
	"minipsf",
	"psf2",
	"minipsf2",
	"psfp",
	"minipsfp"
};
// clang-format on

const CPlaylist::ITEM& CPlaylist::GetItem(unsigned int index) const
{
	assert(index < m_items.size());
	if(index >= m_items.size())
	{
		throw std::runtime_error("Invalid item index.");
	}
	return m_items[index];
}

int CPlaylist::FindItem(unsigned int itemId) const
{
	for(unsigned int i = 0; i < m_items.size(); i++)
	{
		const ITEM& item(m_items[i]);
		if(item.id == itemId)
		{
			return i;
		}
	}
	return -1;
}

unsigned int CPlaylist::GetItemCount() const
{
	return static_cast<unsigned int>(m_items.size());
}

bool CPlaylist::IsLoadableExtension(const std::string& extension)
{
	for(const auto& loadableExtension : g_loadableExtensions)
	{
		if(!stricmp(extension.c_str(), loadableExtension))
		{
			return true;
		}
	}
	return false;
}

void CPlaylist::PopulateItemFromTags(ITEM& item, const CPsfTags& tags)
{
	if(tags.HasTag("title"))
	{
		item.title = tags.GetTagValue("title");
	}
	else
	{
		item.title = item.path;
	}
	item.length = CPsfTags::ConvertTimeString(tags.GetTagValue("length").c_str());
}

unsigned int CPlaylist::InsertArchive(const std::wstring& path)
{
	m_archives.push_back(path);
	return static_cast<unsigned int>(m_archives.size());
}

std::wstring CPlaylist::GetArchive(unsigned int archiveId) const
{
	archiveId--;
	assert(archiveId < m_archives.size());
	if(archiveId >= m_archives.size())
	{
		return std::wstring();
	}
	else
	{
		return m_archives[archiveId];
	}
}

unsigned int CPlaylist::InsertItem(const ITEM& item)
{
	ITEM modifiedItem(item);
	modifiedItem.id = m_currentItemId++;
	m_items.push_back(modifiedItem);
	OnItemInsert(modifiedItem);
	return modifiedItem.id;
}

void CPlaylist::DeleteItem(unsigned int index)
{
	assert(index < m_items.size());
	if(index >= m_items.size())
	{
		throw std::runtime_error("Invalid item index.");
	}
	m_items.erase(m_items.begin() + index);
	OnItemDelete(index);
}

void CPlaylist::UpdateItem(unsigned int index, const CPlaylist::ITEM& item)
{
	assert(index < m_items.size());
	if(index >= m_items.size())
	{
		throw std::runtime_error("Invalid item index.");
	}
	m_items[index] = item;
	OnItemUpdate(index, item);
}

void CPlaylist::ExchangeItems(unsigned int index1, unsigned int index2)
{
	assert(index1 < m_items.size());
	assert(index2 < m_items.size());
	ITEM item1 = GetItem(index1);
	ITEM item2 = GetItem(index2);
	UpdateItem(index1, item2);
	UpdateItem(index2, item1);
}

void CPlaylist::Clear()
{
	OnItemsClear();
	m_items.clear();
	m_archives.clear();
}

void CPlaylist::Read(const fs::path& playlistPath)
{
	Clear();

	std::unique_ptr<Framework::Xml::CNode> document;
	{
		auto stream(Framework::CreateInputStdStream(playlistPath.native()));
		document = Framework::Xml::CParser::ParseDocument(stream);
	}
	if(!document)
	{
		throw std::runtime_error("Couldn't parse document.");
	}
	auto parentPath = playlistPath.parent_path();

	auto items = document->SelectNodes(PLAYLIST_NODE_TAG "/" PLAYLIST_ITEM_NODE_TAG);
	for(auto nodeIterator(std::begin(items));
	    nodeIterator != std::end(items); nodeIterator++)
	{
		auto itemNode = (*nodeIterator);
		auto itemPath = fs::path(Framework::Utf8::ConvertFrom(itemNode->GetAttribute(PLAYLIST_ITEM_PATH_ATTRIBUTE)));
		const char* title = itemNode->GetAttribute(PLAYLIST_ITEM_TITLE_ATTRIBUTE);
		const char* length = itemNode->GetAttribute(PLAYLIST_ITEM_LENGTH_ATTRIBUTE);

		if(!itemPath.is_absolute())
		{
			itemPath = parentPath / itemPath;
		}

		ITEM item;
		item.path = itemPath.wstring();
		if(title != NULL)
		{
			item.title = Framework::Utf8::ConvertFrom(title);
		}

		if(length != NULL)
		{
			item.length = atoi(length);
		}
		else
		{
			item.length = 0;
		}

		InsertItem(item);
	}
}

void CPlaylist::Write(const fs::path& playlistPath)
{
	auto document = std::make_unique<Framework::Xml::CNode>();
	auto playlistNode = document->InsertNode(std::make_unique<Framework::Xml::CNode>(PLAYLIST_NODE_TAG, true));

	auto parentPath = playlistPath.parent_path();

	for(auto itemIterator(std::begin(m_items));
	    itemIterator != std::end(m_items); itemIterator++)
	{
		const auto& item(*itemIterator);

		auto itemPath = fs::path(item.path);
		auto itemRelativePath(naive_uncomplete(itemPath, parentPath));

		auto itemNode = playlistNode->InsertNode(std::make_unique<Framework::Xml::CNode>(PLAYLIST_ITEM_NODE_TAG, true));
		itemNode->InsertAttribute(PLAYLIST_ITEM_PATH_ATTRIBUTE, Framework::Utf8::ConvertTo(itemRelativePath.wstring()).c_str());
		itemNode->InsertAttribute(PLAYLIST_ITEM_TITLE_ATTRIBUTE, Framework::Utf8::ConvertTo(item.title).c_str());
		itemNode->InsertAttribute(Framework::Xml::CreateAttributeIntValue(PLAYLIST_ITEM_LENGTH_ATTRIBUTE, item.length));
	}

	auto stream(Framework::CreateOutputStdStream(playlistPath.native()));
	Framework::Xml::CWriter::WriteDocument(stream, document.get());
}
