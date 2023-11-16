#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <QModelIndex>
#include <QString>
#include "library_model.hpp"
#include "library_proxy_model.hpp"
#include "test_data/filter_by_request_test_data.hpp"
#include "test_data/filter_by_tags_test_data.hpp"
#include "test_data/sort_by_authors_test_data.hpp"
#include "test_data/sort_by_fuzzing_test_data.hpp"
#include "test_data/sort_by_last_opened_test_data.hpp"
#include "test_data/sort_by_reading_progress_data.hpp"
#include "test_data/sort_by_recently_added_test_data.hpp"
#include "test_data/sort_by_title_test_data.hpp"


using namespace testing;
using ::testing::ReturnRef;
using namespace adapters;
using namespace adapters::data_models;
using namespace domain::value_objects;
using namespace domain::entities;

namespace tests::adapters
{

//
// Sort by title
//

class ALibraryProxyModelTitleSorter
    : public ::testing::TestWithParam<test_data::SortByTitleTestData>
{
public:
    void SetUp() override
    {
    }

    LibraryProxyModel libraryProxyModel;
};

TEST_F(ALibraryProxyModelTitleSorter, SucceedsRemovingFilterTag)
{
    // Arrange
    QString firstTag = "FirstTag";
    QString secondTag = "SecondTag";
    Book firstBook("some/path.pdf", BookMetaData { .title = "FirstBook" });
    firstBook.addTag(Tag(firstTag));
    Book secondBook("other/path.pdf", BookMetaData { .title = "SecondBook" });
    secondBook.addTag(Tag(secondTag));

    std::vector<Book> bookVec { firstBook, secondBook };

    LibraryModel model(bookVec);
    libraryProxyModel.setSourceModel(&model);
    libraryProxyModel.addFilterTag(firstTag);
    libraryProxyModel.addFilterTag(secondTag);

    // Act
    libraryProxyModel.removeFilterTag(firstTag);  // Now only the secondTag

    QModelIndex parent;
    auto firstBookIsAccepted = libraryProxyModel.filterAcceptsRow(0, parent);
    auto secondBookIsAccepted = libraryProxyModel.filterAcceptsRow(1, parent);

    // Assert
    // The first tag is no accepted anymore, since we removed the filter
    EXPECT_FALSE(firstBookIsAccepted);
    // The second book is accepted since the filter tag matches
    EXPECT_TRUE(secondBookIsAccepted);
}

TEST_P(ALibraryProxyModelTitleSorter, SucceedsSortingData)
{
    // Arrange
    test_data::SortByTitleTestData data = GetParam();
    std::vector<Book> bookVec { data.first, data.second };

    LibraryModel model(bookVec);
    libraryProxyModel.setSourceModel(&model);
    libraryProxyModel.setSortRole(LibraryProxyModel::SortRole::Title);


    // Act
    QModelIndex parent;
    auto result = libraryProxyModel.lessThan(model.index(0, 0, parent),
                                             model.index(1, 0, parent));

    // Assert
    EXPECT_EQ(data.expectedResult, result);
}

//
// Sort by recently added
//

class ALibraryProxyModelRecentlyAddedSorter
    : public ::testing::TestWithParam<test_data::SortByRecentlyAddedTestData>
{
public:
    void SetUp() override
    {
    }

    LibraryProxyModel libraryProxyModel;
};

TEST_P(ALibraryProxyModelRecentlyAddedSorter, SucceedsSortingData)
{
    // Arrange
    test_data::SortByRecentlyAddedTestData data = GetParam();
    std::vector<Book> bookVec { data.first, data.second };

    LibraryModel model(bookVec);
    libraryProxyModel.setSourceModel(&model);
    libraryProxyModel.setSortRole(LibraryProxyModel::SortRole::RecentlyAdded);


    // Act
    QModelIndex parent;
    auto result = libraryProxyModel.lessThan(model.index(0, 0, parent),
                                             model.index(1, 0, parent));

    // Assert
    EXPECT_EQ(data.expectedResult, result);
}

//
// Sort by authors
//

class ALibraryProxyModelAuthorsSorter
    : public ::testing::TestWithParam<test_data::SortByAuthorsTestData>
{
public:
    void SetUp() override
    {
    }

    LibraryProxyModel libraryProxyModel;
};

TEST_P(ALibraryProxyModelAuthorsSorter, SucceedsSortingData)
{
    // Arrange
    test_data::SortByAuthorsTestData data = GetParam();
    std::vector<Book> bookVec { data.first, data.second };

    LibraryModel model(bookVec);
    libraryProxyModel.setSourceModel(&model);
    libraryProxyModel.setSortRole(LibraryProxyModel::SortRole::Authors);


    // Act
    QModelIndex parent;
    auto result = libraryProxyModel.lessThan(model.index(0, 0, parent),
                                             model.index(1, 0, parent));

    // Assert
    EXPECT_EQ(data.expectedResult, result);
}

//
// Sort byLast opened
//

class ALibraryProxyModelLastOpenedSorter
    : public ::testing::TestWithParam<test_data::SortByLastOpenedTestData>
{
public:
    void SetUp() override
    {
    }

    LibraryProxyModel libraryProxyModel;
};

TEST_P(ALibraryProxyModelLastOpenedSorter, SucceedsSortingData)
{
    // Arrange
    test_data::SortByLastOpenedTestData data = GetParam();
    std::vector<Book> bookVec { data.first, data.second };

    LibraryModel model(bookVec);
    libraryProxyModel.setSourceModel(&model);
    libraryProxyModel.setSortRole(LibraryProxyModel::SortRole::LastOpened);


    // Act
    QModelIndex parent;
    auto result = libraryProxyModel.lessThan(model.index(0, 0, parent),
                                             model.index(1, 0, parent));

    // Assert
    EXPECT_EQ(data.expectedResult, result);
}

//
// Sort by fuzzing
//

class ALibraryProxyModelFuzzSorter
    : public ::testing::TestWithParam<test_data::SortByFuzzingTestData>
{
public:
    void SetUp() override
    {
    }

    LibraryProxyModel libraryProxyModel;
};

TEST_P(ALibraryProxyModelFuzzSorter, SucceedsSortingData)
{
    // Arrange
    test_data::SortByFuzzingTestData data = GetParam();
    std::vector<Book> bookVec { data.first, data.second };

    LibraryModel model(bookVec);
    libraryProxyModel.setSourceModel(&model);
    libraryProxyModel.setSortString(data.sortString);


    // Act
    QModelIndex parent;
    auto result = libraryProxyModel.lessThan(model.index(0, 0, parent),
                                             model.index(1, 0, parent));

    // Assert
    EXPECT_EQ(data.expectedResult, result);
}

//
// Sort by book progress (higher percentage read)
//

class ALibraryProxyModelReadingProgressSorter
    : public ::testing::TestWithParam<test_data::SortByReadingProgressTestData>
{
public:
    void SetUp() override
    {
    }

    LibraryProxyModel libraryProxyModel;
};

TEST_P(ALibraryProxyModelReadingProgressSorter, SucceedsSortingData)
{
    // Arrange
    test_data::SortByReadingProgressTestData data = GetParam();
    std::vector<Book> bookVec { data.first, data.second };

    LibraryModel model(bookVec);
    libraryProxyModel.setSourceModel(&model);
    libraryProxyModel.setSortRole(LibraryProxyModel::SortRole::Progress);


    // Act
    QModelIndex parent;
    auto result = libraryProxyModel.lessThan(model.index(0, 0, parent),
                                             model.index(1, 0, parent));

    // Assert
    EXPECT_EQ(data.expectedResult, result);
}

//
// Filter by tags
//

class ALibraryProxyModelTagFilter
    : public ::testing::TestWithParam<test_data::FilterByTagsTestData>
{
public:
    void SetUp() override
    {
    }

    LibraryProxyModel libraryProxyModel;
};

TEST_P(ALibraryProxyModelTagFilter, SucceedsSortingData)
{
    // Arrange
    test_data::FilterByTagsTestData data = GetParam();
    std::vector<Book> bookVec { data.book };

    LibraryModel model(bookVec);
    libraryProxyModel.setSourceModel(&model);

    for(const auto& filterTag : data.filterTags)
    {
        libraryProxyModel.addFilterTag(filterTag);
    }


    // Act
    QModelIndex parent;
    auto result = libraryProxyModel.filterAcceptsRow(0, parent);

    // Assert
    EXPECT_EQ(data.expectedResult, result);
}

//
// Filter by request
//

class ALibraryProxyModelRequestFilter
    : public ::testing::TestWithParam<test_data::FilterByRequestTestData>
{
public:
    void SetUp() override
    {
    }

    LibraryProxyModel libraryProxyModel;
};

TEST_P(ALibraryProxyModelRequestFilter, SucceedsSortingData)
{
    // Arrange
    test_data::FilterByRequestTestData data = GetParam();
    std::vector<Book> bookVec { data.book };

    LibraryModel model(bookVec);
    libraryProxyModel.setSourceModel(&model);

    libraryProxyModel.setFilterRequest(data.authors, data.format, "",
                                       data.onlyBooks, data.onlyFiles,
                                       data.read, data.unread);


    // Act
    QModelIndex parent;
    auto result = libraryProxyModel.filterAcceptsRow(0, parent);

    // Assert
    EXPECT_EQ(data.expectedResult, result);
}

// Register the test cases
INSTANTIATE_TEST_SUITE_P(TestSuite, ALibraryProxyModelTitleSorter,
                         ::testing::ValuesIn(test_data::titleSortingTestData));

INSTANTIATE_TEST_SUITE_P(
    TestSuite, ALibraryProxyModelRecentlyAddedSorter,
    ::testing::ValuesIn(test_data::recentlyAddedSortingTestData));

INSTANTIATE_TEST_SUITE_P(
    TestSuite, ALibraryProxyModelAuthorsSorter,
    ::testing::ValuesIn(test_data::authorsSortingTestData));

INSTANTIATE_TEST_SUITE_P(
    TestSuite, ALibraryProxyModelLastOpenedSorter,
    ::testing::ValuesIn(test_data::lastOpenedSortingTestData));

INSTANTIATE_TEST_SUITE_P(
    TestSuite, ALibraryProxyModelReadingProgressSorter,
    ::testing::ValuesIn(test_data::readingProgressSortingTestData));

INSTANTIATE_TEST_SUITE_P(TestSuite, ALibraryProxyModelFuzzSorter,
                         ::testing::ValuesIn(test_data::fuzzSortingTestData));

INSTANTIATE_TEST_SUITE_P(TestSuite, ALibraryProxyModelTagFilter,
                         ::testing::ValuesIn(test_data::tagFilteringTestData));

INSTANTIATE_TEST_SUITE_P(
    TestSuite, ALibraryProxyModelRequestFilter,
    ::testing::ValuesIn(test_data::requestFilteringTestData));

}  // namespace tests::adapters