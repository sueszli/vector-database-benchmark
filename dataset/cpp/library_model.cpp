#include "library_model.hpp"
#include <QBuffer>
#include <QByteArray>
#include <QDebug>
#include <QList>
#include <QString>
#include "book.hpp"
#include "tag_dto.hpp"


using namespace domain::entities;

namespace adapters::data_models
{

LibraryModel::LibraryModel(const std::vector<Book>& data) :
    m_data(data)
{
}

int LibraryModel::rowCount(const QModelIndex& parent) const
{
    if(parent.isValid())
        return 0;

    return m_data.size();
}

QVariant LibraryModel::data(const QModelIndex& index, int role) const
{
    if(!index.isValid())
        return QVariant();

    const Book& book = m_data.at(index.row());
    switch(role)
    {
    case UuidRole:
        return book.getUuid().toString(QUuid::WithoutBraces);
        break;
    case TitleRole:
        return book.getTitle();
        break;
    case AuthorsRole:
        return book.getAuthors();
        break;
    case FilePathRole:
        return book.getFilePath();
        break;
    case CreatorRole:
        return book.getCreator();
        break;
    case CreationDateRole:
        return book.getCreationDate();
        break;
    case FormatRole:
        return book.getFormat();
        break;
    case LanguageRole:
        return book.getLanguage();
        break;
    case DocumentSizeRole:
        return book.getDocumentSize();
        break;
    case PagesSizeRole:
        return book.getPagesSize();
        break;
    case PageCountRole:
        return book.getPageCount();
        break;
    case CurrentPageRole:
        return book.getCurrentPage();
        break;
    case BookReadingProgressRole:
        return book.getBookReadingProgress();
        break;
    case AddedToLibraryRole:
        return book.getAddedToLibrary().toLocalTime().toString(
            Book::dateTimeStringFormat);
        break;
    case LastOpenedRole:
        return book.getLastOpened().isNull()
                   ? "Never"
                   : book.getLastOpened().toLocalTime().toString(
                         Book::dateTimeStringFormat);
        break;
    case CoverRole:
    {
        auto pathWithScheme =
            QUrl::fromLocalFile(book.getCoverPath()).toString();

        return book.hasCover() && !book.getCoverPath().isEmpty()
                   ? pathWithScheme
                   : "";
    }
    case TagsRole:
        return QVariant::fromValue(convertTagsToDtos(book.getTags()));
        break;
    case DownloadedRole:
        return book.isDownloaded();
        break;
    case MediaDownloadProgressRole:
        return book.getMediaDownloadProgress();
        break;
    case ExistsOnlyOnClientRole:
        return book.existsOnlyOnClient();
        break;
    default:
        return QVariant();
    }
}

QHash<int, QByteArray> LibraryModel::roleNames() const
{
    static QHash<int, QByteArray> roles {
        { UuidRole, "uuid" },
        { TitleRole, "title" },
        { AuthorsRole, "authors" },
        { FilePathRole, "filePath" },
        { CreatorRole, "creator" },
        { CreationDateRole, "creationDate" },
        { FormatRole, "format" },
        { DocumentSizeRole, "documentSize" },
        { PagesSizeRole, "pagesSize" },
        { PageCountRole, "pageCount" },
        { CurrentPageRole, "currentPage" },
        { BookReadingProgressRole, "bookReadingProgress" },
        { AddedToLibraryRole, "addedToLibrary" },
        { LastOpenedRole, "lastOpened" },
        { CoverRole, "cover" },
        { TagsRole, "tags" },
        { DownloadedRole, "downloaded" },
        { MediaDownloadProgressRole, "mediaDownloadProgress" },
        { ExistsOnlyOnClientRole, "existsOnlyOnClient" },
    };

    return roles;
}

void LibraryModel::processBookCover(int row)
{
    auto modelIndex = index(row, 0);

    emit dataChanged(modelIndex, modelIndex, { CoverRole });
}

QList<dtos::TagDto> LibraryModel::convertTagsToDtos(
    const QList<Tag>& tags) const
{
    QList<dtos::TagDto> tagDtos;
    for(const auto& tag : tags)
    {
        auto tagDto = dtos::TagDto {
            .uuid = tag.getUuid().toString(QUuid::WithoutBraces),
            .name = tag.getName(),
        };
        tagDtos.push_back(tagDto);
    }

    return tagDtos;
}

QVector<int> LibraryModel::getAllRoles()
{
    QVector<int> allRoles;
    int lastRole = Invalid;
    for(int i = TitleRole; i < lastRole; ++i)
    {
        allRoles.push_back(i);
    }

    return allRoles;
}

void LibraryModel::refreshTags(int row)
{
    emit dataChanged(index(row, 0), index(row, 0), { TagsRole });
}

void LibraryModel::refreshBook(int row)
{
    auto allRoles = getAllRoles();

    emit dataChanged(index(row, 0), index(row, 0), allRoles);
}

void LibraryModel::startBookClearing()
{
    beginResetModel();
}

void LibraryModel::endBookClearing()
{
    endResetModel();
}

void LibraryModel::downloadingBookMediaProgressChanged(int row)
{
    emit dataChanged(index(row, 0), index(row, 0),
                     { MediaDownloadProgressRole });
}

void LibraryModel::startInsertingRow(int index)
{
    beginInsertRows(QModelIndex(), index, index);
}

void LibraryModel::endInsertingRow()
{
    endInsertRows();
}

void LibraryModel::startDeletingBook(int index)
{
    beginRemoveRows(QModelIndex(), index, index);
}

void LibraryModel::endDeletingBook()
{
    endRemoveRows();
}

}  // namespace adapters::data_models
