/*
 * Copyright (C) 2009, 2010, 2011 Nicolas Bonnefon and other contributors
 *
 * This file is part of glogg.
 *
 * glogg is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * glogg is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with glogg.  If not, see <http://www.gnu.org/licenses/>.
 */

// This file implements class SavedSearch

#include <QDataStream>
#include <QSettings>

#include "log.h"
#include "savedsearches.h"

void SavedSearches::addRecent( const QString& text )
{
    // We're not interested in blank lines
    if ( text.isEmpty() )
        return;

    savedSearches_.removeAll( text );
    savedSearches_.push_front( text );

    trim();
}

QStringList SavedSearches::recentSearches() const
{
    return savedSearches_;
}

int SavedSearches::historySize() const
{
    return historySize_;
}

void SavedSearches::setHistorySize( int historySize )
{
    historySize_ = historySize;
    trim();
}

void SavedSearches::clear()
{
    savedSearches_.clear();
}

void SavedSearches::trim()
{
    while ( savedSearches_.size() > historySize_ )
        savedSearches_.pop_back();
}

//
// Persistable virtual functions implementation
//

void SavedSearches::saveToStorage( QSettings& settings ) const
{
    LOG_DEBUG << "SavedSearches::saveToStorage";

    settings.beginGroup( "SavedSearches" );
    settings.setValue( "version", SAVEDSEARCHES_VERSION );
    settings.remove( "searchHistory" );
    settings.beginWriteArray( "searchHistory" );
    for ( int i = 0; i < savedSearches_.size(); ++i ) {
        settings.setArrayIndex( i );
        settings.setValue( "string", savedSearches_.at( i ) );
    }
    settings.endArray();
    settings.setValue( "historySize", historySize_ );
    settings.endGroup();
}

void SavedSearches::retrieveFromStorage( QSettings& settings )
{
    LOG_DEBUG << "SavedSearches::retrieveFromStorage";

    savedSearches_.clear();

    if ( settings.contains( "SavedSearches/version" ) ) {
        settings.beginGroup( "SavedSearches" );
        if ( settings.value( "version" ).toInt() == SAVEDSEARCHES_VERSION ) {
            int size = settings.beginReadArray( "searchHistory" );
            for ( int i = 0; i < size; ++i ) {
                settings.setArrayIndex( i );
                QString search = settings.value( "string" ).toString();
                savedSearches_.append( search );
            }
            settings.endArray();
            historySize_ = settings.value( "historySize", MaxNumberOfRecentSearches ).toInt();
        }
        else {
            LOG_ERROR << "Unknown version of saved searches, ignoring it...";
        }
        settings.endGroup();
    }
}
