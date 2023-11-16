#include "catch.hpp"

#include "language_packages.h"

#if BUILD_PYTHON_LANGUAGE_PACKAGE

#	include <memory>
#	include <fstream>

#	include "FileSystem.h"
#	include "IndexerCommandCustom.h"
#	include "PersistentStorage.h"
#	include "ResourcePaths.h"
#	include "SqliteIndexStorage.h"
#	include "TaskExecuteCustomCommands.h"
#	include "TestStorage.h"
#	include "utilityApp.h"

namespace
{
void deleteAllContents(const FilePath& tempPath)
{
	if (tempPath.recheckExists())
	{
		for (const FilePath& path: FileSystem::getFilePathsFromDirectory(tempPath))
		{
			FileSystem::remove(path);
		}
	}
}

std::shared_ptr<TestStorage> parseCode(std::string code)
{
	const FilePath rootPath = FilePath(L"data/PythonIndexerTestSuite/temp/").makeAbsolute();

	if (!rootPath.exists())
	{
		FileSystem::createDirectory(rootPath);
	}

	deleteAllContents(rootPath);

	const FilePath sourceFilePath = rootPath.getConcatenated(L"test.py");
	const FilePath tempDbPath = rootPath.getConcatenated(L"temp.srctrldb");

	{
		std::ofstream codeFile;
		codeFile.open(sourceFilePath.str());
		codeFile << code;
		codeFile.close();
	}

	{
		const std::set<FilePath> indexedPaths = {rootPath};
		const std::set<FilePathFilter> excludeFilters;
		const std::set<FilePathFilter> includeFilters;
		const FilePath workingDirectory(L".");

		std::vector<std::wstring> args;
		args.push_back(L"index");
		args.push_back(L"--source-file-path");
		args.push_back(L"%{SOURCE_FILE_PATH}");
		args.push_back(L"--database-file-path");
		args.push_back(L"%{DATABASE_FILE_PATH}");
		args.push_back(L"--shallow");

		std::shared_ptr<IndexerCommandCustom> indexerCommand = std::make_shared<IndexerCommandCustom>(
			INDEXER_COMMAND_PYTHON,
			FilePath("../app")
				.getConcatenated(ResourcePaths::getPythonIndexerFilePath())
				.makeAbsolute()
				.makeCanonical()
				.wstr(),
			args,
			rootPath,
			tempDbPath,
			std::to_wstring(SqliteIndexStorage::getStorageVersion()),
			sourceFilePath,
			true);

		const utility::ProcessOutput out = utility::executeProcess(
			indexerCommand->getCommand(), indexerCommand->getArguments(), rootPath, false, -1, true);

		if (!out.error.empty())
		{
			FAIL(
				"Error occurred while running the indexer: \"" + utility::encodeToUtf8(out.error) +
				"\". Process output was: \"" + utility::encodeToUtf8(out.output) + "\"");
		}
		REQUIRE(out.exitCode == 0);
	}

	std::shared_ptr<TestStorage> testStorage;
	{
		std::shared_ptr<PersistentStorage> persistentStorage = std::make_shared<PersistentStorage>(
			tempDbPath, FilePath());
		persistentStorage->setup();
		persistentStorage->buildCaches();
		TaskExecuteCustomCommands::runPythonPostProcessing(*(persistentStorage.get()));

		testStorage = TestStorage::create(persistentStorage);
	}

	deleteAllContents(rootPath);
	return testStorage;
}
}	 // namespace

TEST_CASE("python post processing regards class name in call context when adding ambiguous edges")
{
	std::shared_ptr<TestStorage> storage = parseCode(
		"class A:\n"
		"	def __init__(self):\n"
		"		pass\n"
		"\n"
		"class A1(A):\n"
		"	def __init__(self):\n"
		"		A.__init__()\n"
		"\n"
		"class B:\n"
		"	def __init__(self):\n"
		"		pass\n");

	REQUIRE(storage->calls.size() == 1);
	REQUIRE(utility::containsElement<std::wstring>(
		storage->calls, L"test.A1.__init__ -> test.A.__init__"));

	REQUIRE(!utility::containsElement<std::wstring>(
		storage->calls, L"test.A1.__init__ -> test.A1.__init__"));
	REQUIRE(!utility::containsElement<std::wstring>(
		storage->calls, L"test.A1.__init__ -> test.B.__init__"));
}

TEST_CASE("python post processing regards super() in call context when adding ambiguous edges")
{
	std::shared_ptr<TestStorage> storage = parseCode(
		"class A:\n"
		"	def __init__(self):\n"
		"		pass\n"
		"\n"
		"class A1(A):\n"
		"	def __init__(self):\n"
		"		super().__init__()\n"
		"\n"
		"class B:\n"
		"	def __init__(self):\n"
		"		pass\n");

	REQUIRE(storage->calls.size() == 2);
	REQUIRE(utility::containsElement<std::wstring>(
		storage->calls, L"test.A1.__init__ -> test.A.__init__"));

	REQUIRE(!utility::containsElement<std::wstring>(
		storage->calls, L"test.A1.__init__ -> test.A1.__init__"));
	REQUIRE(!utility::containsElement<std::wstring>(
		storage->calls, L"test.A1.__init__ -> test.B.__init__"));
}

#endif	  // BUILD_PYTHON_LANGUAGE_PACKAGE
