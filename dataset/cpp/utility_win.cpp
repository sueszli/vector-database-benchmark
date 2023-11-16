/*
 * Copyright (C) by Daniel Molkentin <danimo@owncloud.com>
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
 */

#include "asserts.h"
#include "utility.h"
#include "gui/configgui.h"

#include <comdef.h>
#include <Lmcons.h>
#include <shlguid.h>
#include <shlobj.h>
#include <string>
#include <winbase.h>
#include <windows.h>
#include <winerror.h>
#include <QCoreApplication>
#include <QDir>
#include <QFile>
#include <QLibrary>
#include <QSettings>
#include <QTemporaryFile>

extern Q_CORE_EXPORT int qt_ntfs_permission_lookup;

static const char systemRunPathC[] = R"(HKEY_LOCAL_MACHINE\Software\Microsoft\Windows\CurrentVersion\Run)";
static const char runPathC[] = R"(HKEY_CURRENT_USER\Software\Microsoft\Windows\CurrentVersion\Run)";

namespace OCC {

static void setupFavLink_private(const QString &folder)
{
    // First create a Desktop.ini so that the folder and favorite link show our application's icon.
    QFile desktopIni(folder + QLatin1String("/Desktop.ini"));
    if (desktopIni.exists()) {
        qCWarning(lcUtility) << desktopIni.fileName() << "already exists, not overwriting it to set the folder icon.";
    } else {
        qCInfo(lcUtility) << "Creating" << desktopIni.fileName() << "to set a folder icon in Explorer.";
        desktopIni.open(QFile::WriteOnly);
        desktopIni.write("[.ShellClassInfo]\r\nIconResource=");
        desktopIni.write(QDir::toNativeSeparators(qApp->applicationFilePath()).toUtf8());
#ifdef APPLICATION_FOLDER_ICON_INDEX
        const auto iconIndex = APPLICATION_FOLDER_ICON_INDEX;
#else
        const auto iconIndex = "0";
#endif
        desktopIni.write(",");
        desktopIni.write(iconIndex);
        desktopIni.write("\r\n");
        desktopIni.close();

        // Set the folder as system and Desktop.ini as hidden+system for explorer to pick it.
        // https://msdn.microsoft.com/en-us/library/windows/desktop/cc144102
        DWORD folderAttrs = GetFileAttributesW((wchar_t *)folder.utf16());
        SetFileAttributesW((wchar_t *)folder.utf16(), folderAttrs | FILE_ATTRIBUTE_SYSTEM);
        SetFileAttributesW((wchar_t *)desktopIni.fileName().utf16(), FILE_ATTRIBUTE_HIDDEN | FILE_ATTRIBUTE_SYSTEM);
    }

    // Windows Explorer: Place under "Favorites" (Links)
    QString linkName;
    QDir folderDir(QDir::fromNativeSeparators(folder));

    /* Use new WINAPI functions */
    PWSTR path;

    if (SHGetKnownFolderPath(FOLDERID_Links, 0, nullptr, &path) == S_OK) {
        QString links = QDir::fromNativeSeparators(QString::fromWCharArray(path));
        linkName = QDir(links).filePath(folderDir.dirName() + QLatin1String(".lnk"));
        CoTaskMemFree(path);
    }
    qCInfo(lcUtility) << "Creating favorite link from" << folder << "to" << linkName;
    if (!QFile::link(folder, linkName))
        qCWarning(lcUtility) << "linking" << folder << "to" << linkName << "failed!";
}

static void removeFavLink_private(const QString &folder)
{
    const QDir folderDir(folder);

    // #1 Remove the Desktop.ini to reset the folder icon
    if (!QFile::remove(folderDir.absoluteFilePath(QLatin1String("Desktop.ini")))) {
        qCWarning(lcUtility) << "Remove Desktop.ini from" << folder
                             << " has failed. Make sure it exists and is not locked by another process.";
    }

    // #2 Remove the system file attribute
    const auto folderAttrs = GetFileAttributesW(folder.toStdWString().c_str());
    if (!SetFileAttributesW(folder.toStdWString().c_str(), folderAttrs & ~FILE_ATTRIBUTE_SYSTEM)) {
        qCWarning(lcUtility) << "Remove system file attribute failed for:" << folder;
    }

    // #3 Remove the link to this folder
    PWSTR path;
    if (!SHGetKnownFolderPath(FOLDERID_Links, 0, nullptr, &path) == S_OK) {
        qCWarning(lcUtility) << "SHGetKnownFolderPath for " << folder << "has failed.";
        return;
    }

    const QDir links(QString::fromWCharArray(path));
    CoTaskMemFree(path);

    const auto linkName = QDir(links).absoluteFilePath(folderDir.dirName() + QLatin1String(".lnk"));

    qCInfo(lcUtility) << "Removing favorite link from" << folder << "to" << linkName;
    if (!QFile::remove(linkName)) {
        qCWarning(lcUtility) << "Removing a favorite link from" << folder << "to" << linkName << "failed.";
    }
}

bool hasSystemLaunchOnStartup_private(const QString &appName)
{
    QString runPath = QLatin1String(systemRunPathC);
    QSettings settings(runPath, QSettings::NativeFormat);
    return settings.contains(appName);
}

bool hasLaunchOnStartup_private(const QString &appName)
{
    QString runPath = QLatin1String(runPathC);
    QSettings settings(runPath, QSettings::NativeFormat);
    return settings.contains(appName);
}

void setLaunchOnStartup_private(const QString &appName, const QString &guiName, bool enable)
{
    Q_UNUSED(guiName);
    QString runPath = QLatin1String(runPathC);
    QSettings settings(runPath, QSettings::NativeFormat);
    if (enable) {
        settings.setValue(appName, QDir::toNativeSeparators(QCoreApplication::applicationFilePath()));
    } else {
        settings.remove(appName);
    }
}

// TODO: Right now only detection on toggle/startup, not when windows theme is switched while nextcloud is running
static inline bool hasDarkSystray_private()
{
    if(Utility::registryGetKeyValue(    HKEY_CURRENT_USER,
                                        QStringLiteral(R"(Software\Microsoft\Windows\CurrentVersion\Themes\Personalize)"),
                                        QStringLiteral("SystemUsesLightTheme") ) == 1) {
        return false;
    }
    else {
        return true;
    }
}

QRect Utility::getTaskbarDimensions()
{
    APPBARDATA barData;
    barData.cbSize = sizeof(APPBARDATA);

    BOOL fResult = (BOOL)SHAppBarMessage(ABM_GETTASKBARPOS, &barData);
    if (!fResult) {
        return QRect();
    }

    RECT barRect = barData.rc;
    return QRect(barRect.left, barRect.top, (barRect.right - barRect.left), (barRect.bottom - barRect.top));
}

bool Utility::registryKeyExists(HKEY hRootKey, const QString &subKey)
{
    HKEY hKey;

    REGSAM sam = KEY_READ | KEY_WOW64_64KEY;
    LONG result = RegOpenKeyEx(hRootKey, reinterpret_cast<LPCWSTR>(subKey.utf16()), 0, sam, &hKey);

    RegCloseKey(hKey);
    return result != ERROR_FILE_NOT_FOUND;
}

QVariant Utility::registryGetKeyValue(HKEY hRootKey, const QString &subKey, const QString &valueName)
{
    QVariant value;

    HKEY hKey;

    REGSAM sam = KEY_READ | KEY_WOW64_64KEY;
    LONG result = RegOpenKeyEx(hRootKey, reinterpret_cast<LPCWSTR>(subKey.utf16()), 0, sam, &hKey);
    ASSERT(result == ERROR_SUCCESS || result == ERROR_FILE_NOT_FOUND);
    if (result != ERROR_SUCCESS)
        return value;

    DWORD type = 0, sizeInBytes = 0;
    result = RegQueryValueEx(hKey, reinterpret_cast<LPCWSTR>(valueName.utf16()), 0, &type, nullptr, &sizeInBytes);
    ASSERT(result == ERROR_SUCCESS || result == ERROR_FILE_NOT_FOUND);
    if (result == ERROR_SUCCESS) {
        switch (type) {
        case REG_DWORD:
            DWORD dword;
            Q_ASSERT(sizeInBytes == sizeof(dword));
            if (RegQueryValueEx(hKey, reinterpret_cast<LPCWSTR>(valueName.utf16()), 0, &type, reinterpret_cast<LPBYTE>(&dword), &sizeInBytes) == ERROR_SUCCESS) {
                value = int(dword);
            }
            break;
        case REG_EXPAND_SZ:
        case REG_SZ: {
            QString string;
            string.resize(sizeInBytes / sizeof(QChar));
            result = RegQueryValueEx(hKey, reinterpret_cast<LPCWSTR>(valueName.utf16()), 0, &type, reinterpret_cast<LPBYTE>(string.data()), &sizeInBytes);

            if (result == ERROR_SUCCESS) {
                int newCharSize = sizeInBytes / sizeof(QChar);
                // From the doc:
                // If the data has the REG_SZ, REG_MULTI_SZ or REG_EXPAND_SZ type, the string may not have been stored with
                // the proper terminating null characters. Therefore, even if the function returns ERROR_SUCCESS,
                // the application should ensure that the string is properly terminated before using it; otherwise, it may overwrite a buffer.
                if (string.at(newCharSize - 1) == QLatin1Char('\0'))
                    string.resize(newCharSize - 1);
                value = string;
            }
            break;
        }
        case REG_BINARY: {
            QByteArray buffer;
            buffer.resize(sizeInBytes);
            result = RegQueryValueEx(hKey, reinterpret_cast<LPCWSTR>(valueName.utf16()), 0, &type, reinterpret_cast<LPBYTE>(buffer.data()), &sizeInBytes);
            if (result == ERROR_SUCCESS) {
                value = buffer.at(12);
            }
            break;
        }
        default:
            Q_UNREACHABLE();
        }
    }
    ASSERT(result == ERROR_SUCCESS || result == ERROR_FILE_NOT_FOUND);

    RegCloseKey(hKey);
    return value;
}

bool Utility::registrySetKeyValue(HKEY hRootKey, const QString &subKey, const QString &valueName, DWORD type, const QVariant &value)
{
    HKEY hKey;
    // KEY_WOW64_64KEY is necessary because CLSIDs are "Redirected and reflected only for CLSIDs that do not specify InprocServer32 or InprocHandler32."
    // https://msdn.microsoft.com/en-us/library/windows/desktop/aa384253%28v=vs.85%29.aspx#redirected__shared__and_reflected_keys_under_wow64
    // This shouldn't be an issue in our case since we use shell32.dll as InprocServer32, so we could write those registry keys for both 32 and 64bit.
    // FIXME: Not doing so at the moment means that explorer will show the cloud provider, but 32bit processes' open dialogs (like the ownCloud client itself) won't show it.
    REGSAM sam = KEY_WRITE | KEY_WOW64_64KEY;
    LONG result = RegCreateKeyEx(hRootKey, reinterpret_cast<LPCWSTR>(subKey.utf16()), 0, nullptr, 0, sam, nullptr, &hKey, nullptr);
    ASSERT(result == ERROR_SUCCESS);
    if (result != ERROR_SUCCESS)
        return false;

    result = -1;
    switch (type) {
    case REG_DWORD: {
        DWORD dword = value.toInt();
        result = RegSetValueEx(hKey, reinterpret_cast<LPCWSTR>(valueName.utf16()), 0, type, reinterpret_cast<const BYTE *>(&dword), sizeof(dword));
        break;
    }
    case REG_EXPAND_SZ:
    case REG_SZ: {
        QString string = value.toString();
        result = RegSetValueEx(hKey, reinterpret_cast<LPCWSTR>(valueName.utf16()), 0, type, reinterpret_cast<const BYTE *>(string.constData()), (string.size() + 1) * sizeof(QChar));
        break;
    }
    default:
        Q_UNREACHABLE();
    }
    ASSERT(result == ERROR_SUCCESS);

    RegCloseKey(hKey);
    return result == ERROR_SUCCESS;
}

bool Utility::registryDeleteKeyTree(HKEY hRootKey, const QString &subKey)
{
    HKEY hKey;
    REGSAM sam = DELETE | KEY_ENUMERATE_SUB_KEYS | KEY_QUERY_VALUE | KEY_SET_VALUE | KEY_WOW64_64KEY;
    LONG result = RegOpenKeyEx(hRootKey, reinterpret_cast<LPCWSTR>(subKey.utf16()), 0, sam, &hKey);
    ASSERT(result == ERROR_SUCCESS);
    if (result != ERROR_SUCCESS)
        return false;

    result = RegDeleteTree(hKey, nullptr);
    RegCloseKey(hKey);
    ASSERT(result == ERROR_SUCCESS);

    result |= RegDeleteKeyEx(hRootKey, reinterpret_cast<LPCWSTR>(subKey.utf16()), sam, 0);
    ASSERT(result == ERROR_SUCCESS);

    return result == ERROR_SUCCESS;
}

bool Utility::registryDeleteKeyValue(HKEY hRootKey, const QString &subKey, const QString &valueName)
{
    HKEY hKey;
    REGSAM sam = KEY_WRITE | KEY_WOW64_64KEY;
    LONG result = RegOpenKeyEx(hRootKey, reinterpret_cast<LPCWSTR>(subKey.utf16()), 0, sam, &hKey);
    ASSERT(result == ERROR_SUCCESS);
    if (result != ERROR_SUCCESS)
        return false;

    result = RegDeleteValue(hKey, reinterpret_cast<LPCWSTR>(valueName.utf16()));
    ASSERT(result == ERROR_SUCCESS);

    RegCloseKey(hKey);
    return result == ERROR_SUCCESS;
}

bool Utility::registryWalkSubKeys(HKEY hRootKey, const QString &subKey, const std::function<void(HKEY, const QString &)> &callback)
{
    HKEY hKey;
    REGSAM sam = KEY_READ | KEY_WOW64_64KEY;
    LONG result = RegOpenKeyEx(hRootKey, reinterpret_cast<LPCWSTR>(subKey.utf16()), 0, sam, &hKey);
    ASSERT(result == ERROR_SUCCESS);
    if (result != ERROR_SUCCESS)
        return false;

    DWORD maxSubKeyNameSize;
    // Get the largest keyname size once instead of relying each call on ERROR_MORE_DATA.
    result = RegQueryInfoKey(hKey, nullptr, nullptr, nullptr, nullptr, &maxSubKeyNameSize, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
    ASSERT(result == ERROR_SUCCESS);
    if (result != ERROR_SUCCESS) {
        RegCloseKey(hKey);
        return false;
    }

    QString subKeyName;
    subKeyName.reserve(maxSubKeyNameSize + 1);

    DWORD retCode = ERROR_SUCCESS;
    for (DWORD i = 0; retCode == ERROR_SUCCESS; ++i) {
        Q_ASSERT(unsigned(subKeyName.capacity()) > maxSubKeyNameSize);
        // Make the previously reserved capacity official again.
        subKeyName.resize(subKeyName.capacity());
        DWORD subKeyNameSize = subKeyName.size();
        retCode = RegEnumKeyEx(hKey, i, reinterpret_cast<LPWSTR>(subKeyName.data()), &subKeyNameSize, nullptr, nullptr, nullptr, nullptr);

        ASSERT(result == ERROR_SUCCESS || retCode == ERROR_NO_MORE_ITEMS);
        if (retCode == ERROR_SUCCESS) {
            // subKeyNameSize excludes the trailing \0
            subKeyName.resize(subKeyNameSize);
            // Pass only the sub keyname, not the full path.
            callback(hKey, subKeyName);
        }
    }

    RegCloseKey(hKey);
    return retCode != ERROR_NO_MORE_ITEMS;
}

bool Utility::registryWalkValues(HKEY hRootKey, const QString &subKey, const std::function<void(const QString &, bool *)> &callback)
{
    HKEY hKey;
    REGSAM sam = KEY_QUERY_VALUE;
    LONG result = RegOpenKeyEx(hRootKey, reinterpret_cast<LPCWSTR>(subKey.utf16()), 0, sam, &hKey);
    ASSERT(result == ERROR_SUCCESS);
    if (result != ERROR_SUCCESS) {
        return false;
    }

    DWORD maxValueNameSize = 0;
    result = RegQueryInfoKey(hKey, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, &maxValueNameSize, nullptr, nullptr, nullptr);
    ASSERT(result == ERROR_SUCCESS);
    if (result != ERROR_SUCCESS) {
        RegCloseKey(hKey);
        return false;
    }

    QString valueName;
    valueName.reserve(maxValueNameSize + 1);

    DWORD retCode = ERROR_SUCCESS;
    bool done = false;
    for (DWORD i = 0; retCode == ERROR_SUCCESS; ++i) {
        Q_ASSERT(unsigned(valueName.capacity()) > maxValueNameSize);
        valueName.resize(valueName.capacity());
        DWORD valueNameSize = valueName.size();
        retCode = RegEnumValue(hKey, i, reinterpret_cast<LPWSTR>(valueName.data()), &valueNameSize, nullptr, nullptr, nullptr, nullptr);

        ASSERT(result == ERROR_SUCCESS || retCode == ERROR_NO_MORE_ITEMS);
        if (retCode == ERROR_SUCCESS) {
            valueName.resize(valueNameSize);
            callback(valueName, &done);

            if (done) {
                break;
            }
        }
    }

    RegCloseKey(hKey);
    return retCode != ERROR_NO_MORE_ITEMS;
}

DWORD Utility::convertSizeToDWORD(size_t &convertVar)
{
    if( convertVar > UINT_MAX ) {
        //throw std::bad_cast();
        convertVar = UINT_MAX; // intentionally default to wrong value here to not crash: exception handling TBD
    }
    return static_cast<DWORD>(convertVar);
}

void Utility::UnixTimeToFiletime(time_t t, FILETIME *filetime)
{
    LONGLONG ll = Int32x32To64(t, 10000000) + 116444736000000000;
    filetime->dwLowDateTime = (DWORD) ll;
    filetime->dwHighDateTime = ll >>32;
}

void Utility::FiletimeToLargeIntegerFiletime(FILETIME *filetime, LARGE_INTEGER *hundredNSecs)
{
    hundredNSecs->LowPart = filetime->dwLowDateTime;
    hundredNSecs->HighPart = filetime->dwHighDateTime;
}

void Utility::UnixTimeToLargeIntegerFiletime(time_t t, LARGE_INTEGER *hundredNSecs)
{
    LONGLONG ll = Int32x32To64(t, 10000000) + 116444736000000000;
    hundredNSecs->LowPart = (DWORD) ll;
    hundredNSecs->HighPart = ll >>32;
}

bool Utility::canCreateFileInPath(const QString &path)
{
    Q_ASSERT(!path.isEmpty());
    const auto pathWithSlash = !path.endsWith(QLatin1Char('/'))
        ? path + QLatin1Char('/')
        : path;
    QTemporaryFile testFile(pathWithSlash + QStringLiteral("~$write-test-file-XXXXXX"));
    return testFile.open();
}

QString Utility::formatWinError(long errorCode)
{
    return QStringLiteral("WindowsError: %1: %2").arg(QString::number(errorCode, 16), QString::fromWCharArray(_com_error(errorCode).ErrorMessage()));
}

QString Utility::getCurrentUserName()
{
    TCHAR username[UNLEN + 1] = {0};
    DWORD len = sizeof(username) / sizeof(TCHAR);
    
    if (!GetUserName(username, &len)) {
        qCWarning(lcUtility) << "Could not retrieve Windows user name." << formatWinError(GetLastError());
    }

    return QString::fromWCharArray(username);
}

void Utility::registerUriHandlerForLocalEditing() { /* URI handler is registered via Nextcloud.wxs */ }

Utility::NtfsPermissionLookupRAII::NtfsPermissionLookupRAII()
{
    qt_ntfs_permission_lookup++;
}

Utility::NtfsPermissionLookupRAII::~NtfsPermissionLookupRAII()
{
    qt_ntfs_permission_lookup--;
}

} // namespace OCC
