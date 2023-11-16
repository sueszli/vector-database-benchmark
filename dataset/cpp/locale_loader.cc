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

#include "qt_base/locale_loader.h"

#include "qt_base/qt_logging.h"

#include <QCoreApplication>
#include <QDir>
#include <QLocale>
#include <QTranslator>

namespace qt_base {

namespace {

const QString kTranslationsDir = QStringLiteral(":/tr/");

} // namespace

//--------------------------------------------------------------------------------------------------
LocaleLoader::LocaleLoader()
{
    const QStringList qm_file_list =
        QDir(kTranslationsDir).entryList(QStringList("*.qm"), QDir::Files);

    for (const auto& qm_file : qm_file_list)
    {
        QString locale_name = qm_file.chopped(3); // Remove file extension (*.qm).

        if (locale_name.right(2).isUpper())
        {
            // xx_XX (language / country).
            locale_name = locale_name.right(5);
        }
        else
        {
            // xx (language only).
            locale_name = locale_name.right(2);
        }

        LOG(LS_INFO) << "Translation file added: " << qm_file << " (" << locale_name << ")";

        if (locale_list_.contains(locale_name))
            locale_list_[locale_name].push_back(qm_file);
        else
            locale_list_.insert(locale_name, QStringList(qm_file));
    }
}

//--------------------------------------------------------------------------------------------------
LocaleLoader::~LocaleLoader()
{
    removeTranslators();
}

//--------------------------------------------------------------------------------------------------
LocaleLoader::LocaleList LocaleLoader::localeList() const
{
    LocaleList list;

    auto add_locale = [&](const QString& locale_code)
    {
        QLocale locale(locale_code);
        QString name;

        if (locale_code.length() == 2)
        {
            name = QLocale::languageToString(locale.language());
        }
        else
        {
            name = QLocale::languageToString(locale.language())
                + " (" + QLocale::countryToString(locale.country()) + ")";
        }

        list.push_back(Locale(locale_code, name));
    };

    add_locale("en");

    for (auto it = locale_list_.constBegin(); it != locale_list_.constEnd(); ++it)
        add_locale(it.key());

    std::sort(list.begin(), list.end(), [](const Locale& a, const Locale& b)
    {
        return QString::compare(a.second, b.second, Qt::CaseInsensitive) < 0;
    });

    return list;
}

//--------------------------------------------------------------------------------------------------
bool LocaleLoader::contains(const QString& locale) const
{
    return locale_list_.contains(locale);
}

//--------------------------------------------------------------------------------------------------
void LocaleLoader::installTranslators(const QString& locale)
{
    removeTranslators();

    LOG(LS_INFO) << "Install translators for: " << locale;

    auto file_list = locale_list_.constFind(locale);
    if (file_list == locale_list_.constEnd())
        return;

    for (const auto& file : file_list.value())
    {
        std::unique_ptr<QTranslator> translator = std::make_unique<QTranslator>();

        if (translator->load(file, kTranslationsDir))
        {
            if (QCoreApplication::installTranslator(translator.get()))
                translator_list_.push_back(translator.release());
        }
    }
}

//--------------------------------------------------------------------------------------------------
void LocaleLoader::removeTranslators()
{
    LOG(LS_INFO) << "Cleanup translators";

    for (auto it = translator_list_.begin(); it != translator_list_.end(); ++it)
    {
        QCoreApplication::removeTranslator(*it);
        delete *it;
    }

    translator_list_.clear();
}

} // namespace qt_base
