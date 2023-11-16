#ifdef PORTABLE_CONFIG

#include "updatemanager.h"
#include "services/notifymanager.h"
#include "common/unused.h"
#include <QDebug>
#include <QRegularExpression>
#include <QCoreApplication>
#include <QFileInfo>
#include <QNetworkRequest>
#include <QNetworkAccessManager>
#include <QNetworkReply>
#include <QJsonDocument>
#include <QJsonObject>
#include <QtConcurrent/QtConcurrentRun>

UpdateManager::UpdateManager(QObject *parent) :
    QObject(parent)
{
    connect(this, SIGNAL(updatingError(QString)), this, SLOT(handleUpdatingError(QString)));
    netManager = new QNetworkAccessManager(this);
}

UpdateManager::~UpdateManager()
{

}

void UpdateManager::checkForUpdates()
{
    if (!CFG_CORE.General.CheckUpdatesOnStartup.get())
        return;

    static_qstring(url, "https://sqlitestudio.pl/rest/updates");
    QNetworkRequest request(url);
    QNetworkReply* response = netManager->get(request);
    connect(response, &QNetworkReply::finished, [this, response]()
    {
        response->deleteLater();
        handleUpdatesResponse(response);
    });
}

bool UpdateManager::isPlatformEligibleForUpdate() const
{
    return getDistributionType() != DistributionType::OS_MANAGED;
}

void UpdateManager::handleUpdatesResponse(QNetworkReply* response)
{
    if (response->error() != QNetworkReply::NoError)
    {
        emit updatingError(response->errorString());
        return;
    }

    QJsonParseError parsingError;
    QJsonDocument json = QJsonDocument::fromJson(response->readAll(), &parsingError);

    if (parsingError.error != QJsonParseError::NoError)
    {
        emit updatingError(parsingError.errorString());
        return;
    }

    QString version = json["version"].toString();
    QStringList versionParts = version.split(".");
    QString alignedVersion = versionParts[0] + versionParts[1].rightJustified(2, '0') + versionParts[2].rightJustified(2, '0');
    int versionNumber = alignedVersion.toInt();

    if (SQLITESTUDIO->getVersion() >= versionNumber)
    {
        emit noUpdatesAvailable();
        return;
    }

#if defined(Q_OS_WIN)
    QString url = json["win"].toString();
#elif defined(Q_OS_LINUX)
    QString url = json["lin"].toString();
#elif defined(Q_OS_OSX)
    QString url = json["mac"].toString();
#else
    QString url = jsonQString();
#endif

    emit updateAvailable(version, url);
}

void UpdateManager::handleUpdatingError(const QString& errorMessage)
{
    NOTIFY_MANAGER->warn(tr("Could not check for updates (%1).").arg(errorMessage));
}

#endif // PORTABLE_CONFIG
