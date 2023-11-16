/*
 * Copyright (C) 2016 -- 2019 Anton Filimonov and other contributors
 *
 * This file is part of klogg.
 *
 * klogg is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * klogg is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with klogg.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <QNetworkReply>

#include "configuration.h"
#include "log.h"

#include "downloader.h"

Downloader::Downloader( QObject* parent )
    : QObject( parent )
{
    manager_.setRedirectPolicy( QNetworkRequest::NoLessSafeRedirectPolicy );
}

QString Downloader::lastError() const
{
    return lastError_;
}

void Downloader::download( const QUrl& url, QFile* outputFile )
{
    output_ = outputFile;

    QNetworkRequest request( url );

    if ( !Configuration::get().verifySslPeers() ) {
        auto sslConfiguration = QSslConfiguration::defaultConfiguration();
        sslConfiguration.setPeerVerifyMode( QSslSocket::VerifyNone );
        request.setSslConfiguration( sslConfiguration );
    }

    currentDownload_ = manager_.get( request );

    connect( currentDownload_, &QNetworkReply::downloadProgress, this,
             &Downloader::downloadProgress );

    connect( currentDownload_, &QNetworkReply::finished, this, &Downloader::downloadFinished );

    connect( currentDownload_, &QNetworkReply::readyRead, this, &Downloader::downloadReadyRead );

    LOG_INFO << "Downloading " << url.toEncoded();
}

void Downloader::downloadFinished()
{
    output_->close();
    currentDownload_->deleteLater();

    if ( currentDownload_->error() ) {
        // download failed
        LOG_ERROR << "Download failed: " << currentDownload_->errorString();
        lastError_ = currentDownload_->errorString();
        output_->remove();
        Q_EMIT finished( false );
    }
    else {
        LOG_INFO << "Download done";
        output_->close();
        Q_EMIT finished( true );
    }
}

void Downloader::downloadReadyRead()
{
    output_->write( currentDownload_->readAll() );
}