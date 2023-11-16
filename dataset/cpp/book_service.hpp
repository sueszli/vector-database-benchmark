#pragma once
#include <memory>
#include "book.hpp"
#include "i_book_service.hpp"
#include "i_library_service.hpp"
#include "mupdf/classes.h"
#include "toc/filtered_toc_model.hpp"
#include "utils/book_searcher.hpp"

namespace application::services
{

class BookService : public IBookService
{
public:
    BookService(ILibraryService* libraryService);

    void setUp(QUuid uuid) override;
    mupdf::FzDocument* getFzDocument() override;

    void search(const QString& text,
                core::utils::SearchOptions searchOptions) override;
    void clearSearch() override;
    void goToNextSearchHit() override;
    void goToPreviousSearchHit() override;

    const QList<domain::entities::Highlight>& getHighlights() const override;
    void addHighlight(const domain::entities::Highlight& highlight) override;
    void removeHighlight(const QUuid& uuid) override;
    void changeHighlightColor(const QUuid& uuid, const QColor& color) override;
    void updateBook() override;
    const domain::entities::Highlight* getHighlightAtPoint(
        const QPointF& point, int page) const override;

    const QList<domain::entities::Bookmark>& getBookmarks() const override;
    void addBookmark(const domain::entities::Bookmark& bookmark) override;
    void renameBookmark(const QUuid& uuid, const QString& newName) override;
    void removeBookmark(const QUuid& uuid) override;

    void followLink(const char* uri) override;

    QString getFilePath() const override;
    int getPageCount() const override;
    int getCurrentPage() const override;
    void setCurrentPage(int newCurrentPage) override;
    float getZoom() const override;
    void setZoom(float newZoom) override;

    core::FilteredTOCModel* getTableOfContents() override;

    domain::entities::Book* getBook();
    const domain::entities::Book* getBook() const;

private:
    int getIndexOfBookmark(const QUuid& uuid) const;

    ILibraryService* m_libraryService;
    std::unique_ptr<mupdf::FzDocument> m_fzDocument = nullptr;
    std::unique_ptr<core::utils::BookSearcher> m_bookSearcher = nullptr;
    float m_zoom = 1;
    QUuid m_uuid;

    std::unique_ptr<core::TOCModel> m_TOCModel;
    std::unique_ptr<core::FilteredTOCModel> m_filteredTOCModel;
};

}  // namespace application::services
