/*  PCSX2 - PS2 Emulator for PCs
 *  Copyright (C) 2002-2021  PCSX2 Dev Team
 *
 *  PCSX2 is free software: you can redistribute it and/or modify it under the terms
 *  of the GNU Lesser General Public License as published by the Free Software Found-
 *  ation, either version 3 of the License, or (at your option) any later version.
 *
 *  PCSX2 is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
 *  without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 *  PURPOSE.  See the GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License along with PCSX2.
 *  If not, see <http://www.gnu.org/licenses/>.
 */

#include "PrecompiledHeader.h"

#include "INISettingsInterface.h"
#include "common/FileSystem.h"
#include "common/Console.h"
#include "common/StringUtil.h"
#include <algorithm>
#include <iterator>
#include <mutex>

#ifdef _WIN32
#include <io.h> // _mktemp_s
#else
#include <stdlib.h> // mktemp
#endif

// To prevent races between saving and loading settings, particularly with game settings,
// we only allow one ini to be parsed at any point in time.
static std::mutex s_ini_load_save_mutex;

static std::string GetTemporaryFileName(const std::string& original_filename)
{
	std::string temporary_filename;
	temporary_filename.reserve(original_filename.length() + 8);
	temporary_filename.append(original_filename);

#ifdef _WIN32
	temporary_filename.append(".XXXXXXX");
	_mktemp_s(temporary_filename.data(), temporary_filename.length() + 1);
#else
	temporary_filename.append(".XXXXXX");
#if defined(__linux__) || defined(__ANDROID__) || defined(__APPLE__)
	mkstemp(temporary_filename.data());
#else
	mktemp(temporary_filename.data());
#endif
#endif

	return temporary_filename;
}

INISettingsInterface::INISettingsInterface(std::string filename)
	: m_filename(std::move(filename))
	, m_ini(true, true)
{
}

INISettingsInterface::~INISettingsInterface()
{
	if (m_dirty)
		Save();
}

bool INISettingsInterface::Load()
{
	if (m_filename.empty())
		return false;

	std::unique_lock lock(s_ini_load_save_mutex);
	SI_Error err = SI_FAIL;
	auto fp = FileSystem::OpenManagedCFile(m_filename.c_str(), "rb");
	if (fp)
		err = m_ini.LoadFile(fp.get());

	return (err == SI_OK);
}

bool INISettingsInterface::Save()
{
	if (m_filename.empty())
		return false;

	std::unique_lock lock(s_ini_load_save_mutex);
	std::string temp_filename(GetTemporaryFileName(m_filename));
	SI_Error err = SI_FAIL;
	std::FILE* fp = FileSystem::OpenCFile(temp_filename.c_str(), "wb");
	if (fp)
	{
		err = m_ini.SaveFile(fp, false);
		std::fclose(fp);

		if (err != SI_OK)
		{
			// remove temporary file
			FileSystem::DeleteFilePath(temp_filename.c_str());
		}
		else if (!FileSystem::RenamePath(temp_filename.c_str(), m_filename.c_str()))
		{
			Console.Error("Failed to rename '%s' to '%s'", temp_filename.c_str(), m_filename.c_str());
			FileSystem::DeleteFilePath(temp_filename.c_str());
			return false;
		}
	}

	if (err != SI_OK)
	{
		Console.Warning("Failed to save settings to '%s'.", m_filename.c_str());
		return false;
	}

	m_dirty = false;
	return true;
}

void INISettingsInterface::Clear()
{
	m_ini.Reset();
}

bool INISettingsInterface::GetIntValue(const char* section, const char* key, int* value) const
{
	const char* str_value = m_ini.GetValue(section, key);
	if (!str_value)
		return false;

	std::optional<int> parsed_value = StringUtil::FromChars<int>(str_value, 10);
	if (!parsed_value.has_value())
		return false;

	*value = parsed_value.value();
	return true;
}

bool INISettingsInterface::GetUIntValue(const char* section, const char* key, uint* value) const
{
	const char* str_value = m_ini.GetValue(section, key);
	if (!str_value)
		return false;

	std::optional<uint> parsed_value = StringUtil::FromChars<uint>(str_value, 10);
	if (!parsed_value.has_value())
		return false;

	*value = parsed_value.value();
	return true;
}

bool INISettingsInterface::GetFloatValue(const char* section, const char* key, float* value) const
{
	const char* str_value = m_ini.GetValue(section, key);
	if (!str_value)
		return false;

	std::optional<float> parsed_value = StringUtil::FromChars<float>(str_value);
	if (!parsed_value.has_value())
		return false;

	*value = parsed_value.value();
	return true;
}

bool INISettingsInterface::GetDoubleValue(const char* section, const char* key, double* value) const
{
	const char* str_value = m_ini.GetValue(section, key);
	if (!str_value)
		return false;

	std::optional<double> parsed_value = StringUtil::FromChars<double>(str_value);
	if (!parsed_value.has_value())
		return false;

	*value = parsed_value.value();
	return true;
}

bool INISettingsInterface::GetBoolValue(const char* section, const char* key, bool* value) const
{
	const char* str_value = m_ini.GetValue(section, key);
	if (!str_value)
		return false;

	std::optional<bool> parsed_value = StringUtil::FromChars<bool>(str_value);
	if (!parsed_value.has_value())
		return false;

	*value = parsed_value.value();
	return true;
}

bool INISettingsInterface::GetStringValue(const char* section, const char* key, std::string* value) const
{
	const char* str_value = m_ini.GetValue(section, key);
	if (!str_value)
		return false;

	value->assign(str_value);
	return true;
}

void INISettingsInterface::SetIntValue(const char* section, const char* key, int value)
{
	m_dirty = true;
	m_ini.SetValue(section, key, StringUtil::ToChars(value).c_str(), nullptr, true);
}

void INISettingsInterface::SetUIntValue(const char* section, const char* key, uint value)
{
	m_dirty = true;
	m_ini.SetValue(section, key, StringUtil::ToChars(value).c_str(), nullptr, true);
}

void INISettingsInterface::SetFloatValue(const char* section, const char* key, float value)
{
	m_dirty = true;
	m_ini.SetValue(section, key, StringUtil::ToChars(value).c_str(), nullptr, true);
}

void INISettingsInterface::SetDoubleValue(const char* section, const char* key, double value)
{
	m_dirty = true;
	m_ini.SetValue(section, key, StringUtil::ToChars(value).c_str(), nullptr, true);
}

void INISettingsInterface::SetBoolValue(const char* section, const char* key, bool value)
{
	m_dirty = true;
	m_ini.SetBoolValue(section, key, value, nullptr, true);
}

void INISettingsInterface::SetStringValue(const char* section, const char* key, const char* value)
{
	m_dirty = true;
	m_ini.SetValue(section, key, value, nullptr, true);
}

bool INISettingsInterface::ContainsValue(const char* section, const char* key) const
{
	return (m_ini.GetValue(section, key, nullptr) != nullptr);
}

void INISettingsInterface::DeleteValue(const char* section, const char* key)
{
	m_dirty = true;
	m_ini.Delete(section, key);
}

void INISettingsInterface::ClearSection(const char* section)
{
	m_dirty = true;
	m_ini.Delete(section, nullptr);
	m_ini.SetValue(section, nullptr, nullptr);
}

std::vector<std::string> INISettingsInterface::GetStringList(const char* section, const char* key) const
{
	std::list<CSimpleIniA::Entry> entries;
	if (!m_ini.GetAllValues(section, key, entries))
		return {};

	std::vector<std::string> ret;
	ret.reserve(entries.size());
	for (const CSimpleIniA::Entry& entry : entries)
		ret.emplace_back(entry.pItem);
	return ret;
}

void INISettingsInterface::SetStringList(const char* section, const char* key, const std::vector<std::string>& items)
{
	m_dirty = true;
	m_ini.Delete(section, key);

	for (const std::string& sv : items)
		m_ini.SetValue(section, key, sv.c_str(), nullptr, false);
}

bool INISettingsInterface::RemoveFromStringList(const char* section, const char* key, const char* item)
{
	m_dirty = true;
	return m_ini.DeleteValue(section, key, item, true);
}

bool INISettingsInterface::AddToStringList(const char* section, const char* key, const char* item)
{
	std::list<CSimpleIniA::Entry> entries;
	if (m_ini.GetAllValues(section, key, entries) &&
		std::find_if(entries.begin(), entries.end(),
			[item](const CSimpleIniA::Entry& e) { return (std::strcmp(e.pItem, item) == 0); }) != entries.end())
	{
		return false;
	}

	m_dirty = true;
	m_ini.SetValue(section, key, item, nullptr, false);
	return true;
}

std::vector<std::pair<std::string, std::string>> INISettingsInterface::GetKeyValueList(const char* section) const
{
	using Entry = CSimpleIniA::Entry;
	using KVEntry = std::pair<const char*, Entry>;
	std::vector<KVEntry> entries;
	std::vector<std::pair<std::string, std::string>> output;
	std::list<Entry> keys;
	if (m_ini.GetAllKeys(section, keys))
	{
		std::list<Entry> values;
		for (Entry& key : keys)
		{
			if (!m_ini.GetAllValues(section, key.pItem, values)) // [[unlikely]]
			{
				Console.Error("Got no values for a key returned from GetAllKeys!");
				continue;
			}
			for (const Entry& value : values)
				entries.emplace_back(key.pItem, value);
		}
	}
	std::sort(entries.begin(), entries.end(), [](const KVEntry& a, const KVEntry& b)
	{
		return a.second.nOrder < b.second.nOrder;
	});
	for (const KVEntry& entry : entries)
		output.emplace_back(entry.first, entry.second.pItem);
	return output;
}

void INISettingsInterface::SetKeyValueList(const char* section, const std::vector<std::pair<std::string, std::string>>& items)
{
	m_ini.Delete(section, nullptr);
	for (const std::pair<std::string, std::string>& item : items)
		m_ini.SetValue(section, item.first.c_str(), item.second.c_str(), nullptr, false);
}
