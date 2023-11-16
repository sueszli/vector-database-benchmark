#pragma once
#include <QObject>
#include "adapters_export.hpp"

namespace adapters
{

class ADAPTERS_EXPORT IFreeBooksStorageAccess : public QObject
{
    Q_OBJECT

public:
    virtual ~IFreeBooksStorageAccess() noexcept = default;

    virtual void fetchFirstBooksMetadataPageWithFilter(
        const QString& authorsAndTitle) = 0;
    virtual void fetchBooksMetadataPage(const QString& url) = 0;
    virtual void getBookCover(int id, const QString& url) = 0;
    virtual void getBookMedia(const int id, const QUuid& uuid,
                              const QString& url) = 0;

signals:
    void fetchingFirstMetadataPageSuccessful(const bool success);
    void fetchingBooksMetaDataFinished(const QByteArray& data);
    void gettingBookCoverFinished(int id, const QByteArray& data);
    void gettingBookMediaChunkReady(int id, const QUuid& uuid,
                                    const QByteArray& data,
                                    const QString& format, bool isChunkLast);
    void gettingBookMediaProgressChanged(const int gutenbergId,
                                         qint64 bytesReceived,
                                         qint64 bytesTotal);
};

}  // namespace adapters
