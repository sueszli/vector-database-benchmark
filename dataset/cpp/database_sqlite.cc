//
// Aspia Project
// Copyright (C) 2016-2023 Dmitry Chapyshev <dmitry@aspia.ru>
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.
//

#include "router/database_sqlite.h"

#include "base/logging.h"
#include "base/files/base_paths.h"
#include "base/strings/unicode.h"
#include "build/build_config.h"

#include <optional>

namespace router {

namespace {

//--------------------------------------------------------------------------------------------------
const char* columnTypeToString(int type)
{
    switch (type)
    {
        case SQLITE_INTEGER:
            return "SQLITE_INTEGER";

        case SQLITE_FLOAT:
            return "SQLITE_FLOAT";

        case SQLITE_BLOB:
            return "SQLITE_BLOB";

        case SQLITE_NULL:
            return "SQLITE_NULL";

        case SQLITE_TEXT:
            return "SQLITE_TEXT";

        default:
            return "UNKNOWN";
    }
}

//--------------------------------------------------------------------------------------------------
bool writeText(sqlite3_stmt* statement, const std::string& text, int column)
{
    int error_code = sqlite3_bind_text(
        statement, column, text.c_str(), static_cast<int>(text.size()), SQLITE_STATIC);
    if (error_code != SQLITE_OK)
    {
        LOG(LS_ERROR) << "sqlite3_bind_text failed: " << sqlite3_errstr(error_code)
                      << " (error code: " << error_code << " column: " << column << ")";
        return false;
    }

    return true;
}

//--------------------------------------------------------------------------------------------------
bool writeBlob(sqlite3_stmt* statement, const base::ByteArray& blob, int column)
{
    int error_code = sqlite3_bind_blob(statement,
                                       column,
                                       blob.data(),
                                       static_cast<int>(blob.size()),
                                       SQLITE_STATIC);
    if (error_code != SQLITE_OK)
    {
        LOG(LS_ERROR) << "sqlite3_bind_blob failed: " << sqlite3_errstr(error_code)
                      << " (error code: " << error_code << " column: " << column << ")";
        return false;
    }

    return true;
}

//--------------------------------------------------------------------------------------------------
bool writeInt(sqlite3_stmt* statement, int number, int column)
{
    int error_code = sqlite3_bind_int(statement, column, number);
    if (error_code != SQLITE_OK)
    {
        LOG(LS_ERROR) << "sqlite3_bind_int failed: " << sqlite3_errstr(error_code)
                      << " (error code: " << error_code << " column: " << column << ")";
        return false;
    }

    return true;
}

//--------------------------------------------------------------------------------------------------
bool writeInt64(sqlite3_stmt* statement, int64_t number, int column)
{
    int error_code = sqlite3_bind_int64(statement, column, number);
    if (error_code != SQLITE_OK)
    {
        LOG(LS_ERROR) << "sqlite3_bind_int64 failed: " << sqlite3_errstr(error_code)
                      << " (error code: " << error_code << " column: " << column << ")";
        return false;
    }

    return true;
}

//--------------------------------------------------------------------------------------------------
template <typename T>
std::optional<T> readInteger(sqlite3_stmt* statement, int column)
{
    int column_type = sqlite3_column_type(statement, column);
    if (column_type != SQLITE_INTEGER)
    {
        LOG(LS_ERROR) << "Type is not SQLITE_INTEGER: " << columnTypeToString(column_type)
                      << " (" << column_type << ")";
        return std::nullopt;
    }

    return static_cast<T>(sqlite3_column_int64(statement, column));
}

//--------------------------------------------------------------------------------------------------
std::optional<base::ByteArray> readBlob(sqlite3_stmt* statement, int column)
{
    int column_type = sqlite3_column_type(statement, column);
    if (column_type != SQLITE_BLOB)
    {
        LOG(LS_ERROR) << "Type is not SQLITE_BLOB: " << columnTypeToString(column_type)
                      << " (" << column_type << ")";
        return std::nullopt;
    }

    int blob_size = sqlite3_column_bytes(statement, column);
    if (blob_size <= 0)
    {
        LOG(LS_ERROR) << "Field has an invalid size: " << blob_size;
        return std::nullopt;
    }

    const void* blob = sqlite3_column_blob(statement, column);
    if (!blob)
    {
        LOG(LS_ERROR) << "Failed to get the pointer to the field";
        return std::nullopt;
    }

    return base::fromData(blob, static_cast<size_t>(blob_size));
}

//--------------------------------------------------------------------------------------------------
std::optional<std::string> readText(sqlite3_stmt* statement, int column)
{
    int column_type = sqlite3_column_type(statement, column);
    if (column_type != SQLITE_TEXT)
    {
        LOG(LS_ERROR) << "Type is not SQLITE_TEXT: " << columnTypeToString(column_type)
                      << " (" << column_type << ")";
        return std::nullopt;
    }

    int string_size = sqlite3_column_bytes(statement, column);
    if (string_size <= 0)
    {
        LOG(LS_ERROR) << "Field has an invalid size: " << string_size;
        return std::nullopt;
    }

    const uint8_t* string = sqlite3_column_text(statement, column);
    if (!string)
    {
        LOG(LS_ERROR) << "Failed to get the pointer to the field";
        return std::nullopt;
    }

    return std::string(reinterpret_cast<const char*>(string), static_cast<size_t>(string_size));
}

//--------------------------------------------------------------------------------------------------
std::optional<std::u16string> readText16(sqlite3_stmt* statement, int column)
{
    std::optional<std::string> str = readText(statement, column);
    if (!str.has_value())
        return std::nullopt;

    return base::utf16FromUtf8(*str);
}

//--------------------------------------------------------------------------------------------------
std::optional<base::User> readUser(sqlite3_stmt* statement)
{
    std::optional<int64_t> entry_id = readInteger<int64_t>(statement, 0);
    if (!entry_id.has_value())
    {
        LOG(LS_ERROR) << "Failed to get field 'id'";
        return std::nullopt;
    }

    std::optional<std::u16string> name = readText16(statement, 1);
    if (!name.has_value())
    {
        LOG(LS_ERROR) << "Failed to get field 'name'";
        return std::nullopt;
    }

    std::optional<std::string> group = readText(statement, 2);
    if (!group.has_value())
    {
        LOG(LS_ERROR) << "Failed to get field 'group'";
        return std::nullopt;
    }

    std::optional<base::ByteArray> salt = readBlob(statement, 3);
    if (!salt.has_value())
    {
        LOG(LS_ERROR) << "Failed to get field 'salt'";
        return std::nullopt;
    }

    std::optional<base::ByteArray> verifier = readBlob(statement, 4);
    if (!verifier.has_value())
    {
        LOG(LS_ERROR) << "Failed to get field 'verifier'";
        return std::nullopt;
    }

    std::optional<uint32_t> sessions = readInteger<uint32_t>(statement, 5);
    if (!sessions.has_value())
    {
        LOG(LS_ERROR) << "Failed to get field 'sessions'";
        return std::nullopt;
    }

    std::optional<uint32_t> flags = readInteger<uint32_t>(statement, 6);
    if (!flags.has_value())
    {
        LOG(LS_ERROR) << "Failed to get field 'flags'";
        return std::nullopt;
    }

    base::User user;

    user.entry_id  = *entry_id;
    user.name      = std::move(*name);
    user.group     = std::move(*group);
    user.salt      = std::move(*salt);
    user.verifier  = std::move(*verifier);
    user.sessions  = *sessions;
    user.flags     = *flags;

    return std::move(user);
}

} // namespace

//--------------------------------------------------------------------------------------------------
DatabaseSqlite::DatabaseSqlite(sqlite3* db)
    : db_(db)
{
    DCHECK(db_);
}

//--------------------------------------------------------------------------------------------------
DatabaseSqlite::~DatabaseSqlite()
{
    sqlite3_close(db_);
}

//--------------------------------------------------------------------------------------------------
// static
std::unique_ptr<DatabaseSqlite> DatabaseSqlite::create()
{
    std::filesystem::path dir_path = databaseDirectory();
    if (dir_path.empty())
    {
        LOG(LS_ERROR) << "Invalid directory path";
        return nullptr;
    }

    std::error_code error_code;
    std::filesystem::file_status dir_status = std::filesystem::status(dir_path, error_code);
    if (std::filesystem::exists(dir_status))
    {
        if (!std::filesystem::is_directory(dir_status))
        {
            LOG(LS_ERROR) << "Unable to create directory for database. Need to delete file '"
                          << dir_path << "'";
            return nullptr;
        }
    }
    else
    {
        if (!std::filesystem::create_directories(dir_path, error_code))
        {
            LOG(LS_ERROR) << "Unable to create directory for database: "
                          << base::utf16FromLocal8Bit(error_code.message());
            return nullptr;
        }
    }

    std::filesystem::path file_path = filePath();
    if (file_path.empty())
    {
        LOG(LS_ERROR) << "Invalid file path";
        return nullptr;
    }

    if (std::filesystem::exists(file_path, error_code))
    {
        LOG(LS_ERROR) << "Database file already exists";
        return nullptr;
    }

    std::unique_ptr<DatabaseSqlite> db = open();
    if (!db)
        return nullptr;

    const char kSql[] = "BEGIN TRANSACTION;"
        "CREATE TABLE IF NOT EXISTS \"users\" ("
            "\"id\" INTEGER UNIQUE,"
            "\"name\" TEXT NOT NULL UNIQUE,"
            "\"group\" TEXT NOT NULL,"
            "\"salt\" BLOB NOT NULL,"
            "\"verifier\" BLOB NOT NULL,"
            "\"sessions\" INTEGER DEFAULT 0,"
            "\"flags\" INTEGER DEFAULT 0,"
            "PRIMARY KEY(\"id\" AUTOINCREMENT));"
        "CREATE TABLE IF NOT EXISTS \"hosts\" ("
            "\"id\" INTEGER UNIQUE,"
            "\"key\" BLOB NOT NULL UNIQUE,"
            "PRIMARY KEY(\"id\" AUTOINCREMENT));"
        "COMMIT;";

    char* error_string = nullptr;
    int ret = sqlite3_exec(db->db_, kSql, nullptr, nullptr, &error_string);
    if (ret != SQLITE_OK)
    {
        LOG(LS_ERROR) << "sqlite3_exec failed: " << error_string;
        return nullptr;
    }

    return db;
}

//--------------------------------------------------------------------------------------------------
// static
std::unique_ptr<DatabaseSqlite> DatabaseSqlite::open()
{
    std::filesystem::path file_path = filePath();
    if (file_path.empty())
    {
        LOG(LS_ERROR) << "Invalid file path";
        return nullptr;
    }

    std::string file_path_utf8 = file_path.u8string();
    LOG(LS_INFO) << "Opening database: " << file_path_utf8;

    sqlite3* db = nullptr;

    int error_code = sqlite3_open(file_path_utf8.c_str(), &db);
    if (error_code != SQLITE_OK)
    {
        LOG(LS_ERROR) << "sqlite3_open failed: " << sqlite3_errstr(error_code)
                      << " (" << error_code << ")";
        return nullptr;
    }

    return std::unique_ptr<DatabaseSqlite>(new DatabaseSqlite(db));
}

//--------------------------------------------------------------------------------------------------
// static
std::filesystem::path DatabaseSqlite::filePath()
{
    std::filesystem::path file_path = databaseDirectory();
    if (file_path.empty())
        return std::filesystem::path();

    file_path.append(u"router.db3");
    return file_path;
}

//--------------------------------------------------------------------------------------------------
std::vector<base::User> DatabaseSqlite::userList() const
{
    const char kQuery[] = "SELECT * FROM users";

    sqlite3_stmt* statement;
    int error_code = sqlite3_prepare(db_,
                                     kQuery,
                                     static_cast<int>(std::size(kQuery)),
                                     &statement,
                                     nullptr);
    if (error_code != SQLITE_OK)
    {
        LOG(LS_ERROR) << "sqlite3_prepare failed: " << sqlite3_errstr(error_code)
                      << " (" << error_code << ")";
        return {};
    }

    std::vector<base::User> users;
    for (;;)
    {
        error_code = sqlite3_step(statement);
        if (error_code != SQLITE_ROW)
            break;

        std::optional<base::User> user = readUser(statement);
        if (user.has_value())
            users.emplace_back(std::move(*user));
    }

    sqlite3_finalize(statement);
    return users;
}

//--------------------------------------------------------------------------------------------------
bool DatabaseSqlite::addUser(const base::User& user)
{
    if (!user.isValid())
    {
        LOG(LS_ERROR) << "Not valid user";
        return false;
    }

    static const char kQuery[] =
        "INSERT INTO users ('id', 'name', 'group', 'salt', 'verifier', 'sessions', 'flags') "
        "VALUES (NULL, ?, ?, ?, ?, ?, ?)";

    sqlite3_stmt* statement = nullptr;
    int error_code = sqlite3_prepare(db_,
                                     kQuery,
                                     static_cast<int>(std::size(kQuery)),
                                     &statement,
                                     nullptr);
    if (error_code != SQLITE_OK)
    {
        LOG(LS_ERROR) << "sqlite3_prepare failed: " << sqlite3_errstr(error_code)
                      << " (" << error_code << ")";
        return false;
    }

    std::string username = base::utf8FromUtf16(user.name);
    bool result = false;

    do
    {
        if (!writeText(statement, username, 1))
            break;

        if (!writeText(statement, user.group, 2))
            break;

        if (!writeBlob(statement, user.salt, 3))
            break;

        if (!writeBlob(statement, user.verifier, 4))
            break;

        if (!writeInt(statement, static_cast<int>(user.sessions), 5))
            break;

        if (!writeInt(statement, static_cast<int>(user.flags), 6))
            break;

        error_code = sqlite3_step(statement);
        if (error_code != SQLITE_DONE)
        {
            LOG(LS_ERROR) << "sqlite3_step failed: " << sqlite3_errstr(error_code)
                          << " (" << error_code << ")";
            break;
        }

        result = true;
    }
    while (false);

    sqlite3_finalize(statement);
    return result;
}

//--------------------------------------------------------------------------------------------------
bool DatabaseSqlite::modifyUser(const base::User& user)
{
    if (!user.isValid())
    {
        LOG(LS_ERROR) << "Not valid user";
        return false;
    }

    static const char kQuery[] =
        "UPDATE users SET ('name', 'group', 'salt', 'verifier', 'sessions', 'flags') = "
        "(?, ?, ?, ?, ?, ?) WHERE id=?";

    sqlite3_stmt* statement = nullptr;
    int error_code = sqlite3_prepare(db_,
                                     kQuery,
                                     static_cast<int>(std::size(kQuery)),
                                     &statement,
                                     nullptr);
    if (error_code != SQLITE_OK)
    {
        LOG(LS_ERROR) << "sqlite3_prepare failed: " << sqlite3_errstr(error_code)
                      << " (" << error_code << ")";
        return false;
    }

    std::string username = base::utf8FromUtf16(user.name);
    bool result = false;

    do
    {
        if (!writeText(statement, username, 1))
            break;

        if (!writeText(statement, user.group, 2))
            break;

        if (!writeBlob(statement, user.salt, 3))
            break;

        if (!writeBlob(statement, user.verifier, 4))
            break;

        if (!writeInt(statement, static_cast<int>(user.sessions), 5))
            break;

        if (!writeInt(statement, static_cast<int>(user.flags), 6))
            break;

        if (!writeInt64(statement, user.entry_id, 7))
            break;

        error_code = sqlite3_step(statement);
        if (error_code != SQLITE_DONE)
        {
            LOG(LS_ERROR) << "sqlite3_step failed: " << sqlite3_errstr(error_code)
                          << " (" << error_code << ")";
            break;
        }

        result = true;
    }
    while (false);

    sqlite3_finalize(statement);
    return result;
}

//--------------------------------------------------------------------------------------------------
bool DatabaseSqlite::removeUser(int64_t entry_id)
{
    static const char kQuery[] = "DELETE FROM users WHERE id=?";

    sqlite3_stmt* statement = nullptr;
    int error_code = sqlite3_prepare(db_,
                                     kQuery,
                                     static_cast<int>(std::size(kQuery)),
                                     &statement,
                                     nullptr);
    if (error_code != SQLITE_OK)
    {
        LOG(LS_ERROR) << "sqlite3_prepare failed: " << sqlite3_errstr(error_code);
        return false;
    }

    bool result = false;

    do
    {
        if (!writeInt64(statement, entry_id, 1))
            break;

        error_code = sqlite3_step(statement);
        if (error_code != SQLITE_DONE)
        {
            LOG(LS_ERROR) << "sqlite3_step failed: " << sqlite3_errstr(error_code)
                          << " (" << error_code << ")";
            break;
        }

        result = true;
    }
    while (false);

    sqlite3_finalize(statement);
    return result;
}

//--------------------------------------------------------------------------------------------------
base::User DatabaseSqlite::findUser(std::u16string_view username)
{
    const char kQuery[] = "SELECT * FROM users WHERE name=?";

    sqlite3_stmt* statement = nullptr;
    int error_code = sqlite3_prepare(db_,
                                     kQuery,
                                     static_cast<int>(std::size(kQuery)),
                                     &statement,
                                     nullptr);
    if (error_code != SQLITE_OK)
    {
        LOG(LS_ERROR) << "sqlite3_prepare failed: " << sqlite3_errstr(error_code)
                      << " (" << error_code << ")";
        return base::User::kInvalidUser;
    }

    std::string username_utf8 = base::utf8FromUtf16(username);
    std::optional<base::User> user;

    do
    {
        if (!writeText(statement, username_utf8, 1))
            break;

        if (sqlite3_step(statement) != SQLITE_ROW)
            break;

        user = readUser(statement);
    }
    while (false);

    sqlite3_finalize(statement);
    return user.value_or(base::User::kInvalidUser);
}

//--------------------------------------------------------------------------------------------------
Database::ErrorCode DatabaseSqlite::hostId(
    const base::ByteArray& key_hash, base::HostId* host_id) const
{
    if (key_hash.empty())
    {
        LOG(LS_ERROR) << "Invalid key hash";
        return ErrorCode::UNKNOWN;
    }

    if (!host_id)
    {
        LOG(LS_ERROR) << "Invalid host id";
        return ErrorCode::UNKNOWN;
    }

    *host_id = base::kInvalidHostId;

    const char kQuery[] = "SELECT * FROM hosts WHERE key=?";

    sqlite3_stmt* statement;
    int error_code = sqlite3_prepare(db_,
                                     kQuery,
                                     static_cast<int>(std::size(kQuery)),
                                     &statement,
                                     nullptr);
    if (error_code != SQLITE_OK)
    {
        LOG(LS_ERROR) << "sqlite3_prepare failed: " << sqlite3_errstr(error_code)
                      << " (" << error_code << ")";
        return ErrorCode::UNKNOWN;
    }

    ErrorCode result = ErrorCode::UNKNOWN;

    do
    {
        if (!writeBlob(statement, key_hash, 1))
            break;

        error_code = sqlite3_step(statement);
        if (error_code != SQLITE_ROW)
        {
            LOG(LS_ERROR) << "sqlite3_step failed: " << sqlite3_errstr(error_code)
                          << " (" << error_code << ")";
            result = ErrorCode::NO_HOST_FOUND;
            break;
        }

        std::optional<int64_t> entry_id = readInteger<int64_t>(statement, 0);
        if (!entry_id.has_value())
        {
            LOG(LS_ERROR) << "Failed to get field 'id'";
            break;
        }

        *host_id = static_cast<base::HostId>(*entry_id);
        result = ErrorCode::SUCCESS;
    }
    while (false);

    sqlite3_finalize(statement);
    return result;
}

//--------------------------------------------------------------------------------------------------
bool DatabaseSqlite::addHost(const base::ByteArray& keyHash)
{
    if (keyHash.empty())
    {
        LOG(LS_ERROR) << "Invalid parameters";
        return false;
    }

    const char kQuery[] = "INSERT INTO hosts ('id', 'key') VALUES (NULL, ?)";

    sqlite3_stmt* statement = nullptr;
    int error_code = sqlite3_prepare(db_,
                                     kQuery,
                                     static_cast<int>(std::size(kQuery)),
                                     &statement,
                                     nullptr);
    if (error_code != SQLITE_OK)
    {
        LOG(LS_ERROR) << "sqlite3_prepare failed: " << sqlite3_errstr(error_code)
                      << " (" << error_code << ")";
        return false;
    }

    bool result = false;

    do
    {
        if (!writeBlob(statement, keyHash, 1))
            break;

        error_code = sqlite3_step(statement);
        if (error_code != SQLITE_DONE)
        {
            LOG(LS_ERROR) << "sqlite3_step failed: " << sqlite3_errstr(error_code)
                          << " (" << error_code << ")";
            break;
        }

        result = true;
    }
    while (false);

    sqlite3_finalize(statement);
    return result;
}

//--------------------------------------------------------------------------------------------------
// static
std::filesystem::path DatabaseSqlite::databaseDirectory()
{
    std::filesystem::path dir_path;

#if defined(OS_WIN)
    if (!base::BasePaths::commonAppData(&dir_path))
        return std::filesystem::path();

    dir_path.append(u"aspia");
#elif (OS_LINUX)
    dir_path.append("/var/lib/aspia");
#else
    NOTIMPLEMENTED();
#endif // defined(OS_*)

    return dir_path;
}

} // namespace router
