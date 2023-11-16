#include "TaskExecuteCustomCommands.h"

#include "ApplicationSettings.h"
#include "Blackboard.h"
#include "DialogView.h"
#include "ElementComponentKind.h"
#include "FileSystem.h"
#include "IndexerCommandCustom.h"
#include "IndexerCommandProvider.h"
#include "MessageErrorCountClear.h"
#include "MessageErrorCountUpdate.h"
#include "MessageIndexingStatus.h"
#include "MessageShowStatus.h"
#include "MessageStatus.h"
#include "PersistentStorage.h"
#include "SourceLocationCollection.h"
#include "SourceLocationFile.h"
#include "TextAccess.h"
#include "utility.h"
#include "utilityApp.h"
#include "utilityFile.h"
#include "utilityString.h"

void TaskExecuteCustomCommands::runPythonPostProcessing(PersistentStorage& storage)
{
	std::vector<Id> unsolvedLocationIds;
	for (const StorageSourceLocation location: storage.getStorageSourceLocations())
	{
		if (intToLocationType(location.type) ==
			LOCATION_UNSOLVED)	  // FIXME: this doesn't catch unsolved qualifiers -> convert
								  // Qualifier location type to qualifier edge
		{
			unsolvedLocationIds.push_back(location.id);
		}
	}

	std::shared_ptr<SourceLocationCollection> locationCollection =
		storage.getSourceLocationsForLocationIds(unsolvedLocationIds);

	std::map<std::wstring, std::vector<StorageNode>> nodeNameToStorageNodes;
	if (locationCollection->getSourceLocationCount() > 0)
	{
		for (const StorageNode& node: storage.getStorageNodes())
		{
			nodeNameToStorageNodes[NameHierarchy::deserialize(node.serializedName).back().getName()]
				.push_back(node);
		}
	}

	struct DataToInsert
	{
		StorageEdgeData edgeData;
		Id sourceLocationId;
	};

	storage.setMode(SqliteIndexStorage::STORAGE_MODE_READ);

	std::vector<DataToInsert> dataToInsert;
	std::vector<StorageOccurrence> occurrencesToDelete;
	locationCollection->forEachSourceLocationFile([&nodeNameToStorageNodes,
												   &storage,
												   &dataToInsert,
												   &occurrencesToDelete](
													  std::shared_ptr<SourceLocationFile> locationFile) {
		const FilePath filePath = locationFile->getFilePath();
		if (filePath.empty())
		{
			return;
		}
		if (!filePath.exists())
		{
			LOG_WARNING(L"Skipping post processing for non-existing file: " + filePath.wstr());
			return;
		}

		std::shared_ptr<TextAccess> textAccess = storage.getFileContent(filePath, false);
		if (textAccess)
		{
			std::map<std::wstring, std::vector<std::wstring>> childToParentNodesMap;

			locationFile->forEachStartSourceLocation([textAccess,
													  &childToParentNodesMap,
													  &nodeNameToStorageNodes,
													  &storage,
													  &dataToInsert,
													  &occurrencesToDelete](
														 const SourceLocation* startLoc) {
				if (!startLoc)
				{
					return;
				}
				const SourceLocation* endLoc = startLoc->getOtherLocation();
				if (!endLoc)
				{
					return;
				}

				const std::string tokenLine = textAccess->getLine(
					static_cast<unsigned int>(startLoc->getLineNumber()));
				const std::wstring token = utility::decodeFromUtf8(tokenLine.substr(
					startLoc->getColumnNumber() - 1,
					endLoc->getColumnNumber() - startLoc->getColumnNumber() + 1));

				std::string prefixString = tokenLine.substr(0, startLoc->getColumnNumber() - 1);

				std::wstring definitionContextName = L"";
				{
					std::regex regex("\\s([^\\.()\\s]+)\\.$");
					std::smatch matches;
					std::regex_search(prefixString, matches, regex);
					if (!matches.empty())
					{
						definitionContextName = utility::decodeFromUtf8(matches.str(1));
					}
				}
				{
					std::regex regex("\\s(super\\(\\))\\.$");
					std::smatch matches;
					std::regex_search(prefixString, matches, regex);
					if (!matches.empty())
					{
						for (const Id elementId: startLoc->getTokenIds())
						{
							const StorageEdge edge = storage.getEdgeById(elementId);
							if (edge.id != 0)
							{
								NameHierarchy nameHierarchy = storage.getNameHierarchyForNodeId(
									edge.sourceNodeId);
								if (nameHierarchy.size() > 1)
								{
									std::wstring name = nameHierarchy
															.getRange(0, nameHierarchy.size() - 1)
															.back()
															.getName();

									if (!childToParentNodesMap[name].empty())
									{
										definitionContextName = childToParentNodesMap[name].front();
									}
								}
							}
						}
					}
				}
				{
					std::vector<StorageNode> targetNodes;
					if (!definitionContextName.empty())
					{
						for (const StorageNode& node: nodeNameToStorageNodes[token])
						{
							NameHierarchy nameHierarchy = NameHierarchy::deserialize(
								node.serializedName);
							if (nameHierarchy.size() > 1 &&
								nameHierarchy.getRange(0, nameHierarchy.size() - 1).back().getName() ==
									definitionContextName)
							{
								targetNodes.push_back(node);
							}
						}
					}
					if (targetNodes.empty())
					{
						targetNodes = nodeNameToStorageNodes[token];
					}

					for (const StorageNode& targetNode: targetNodes)
					{
						for (const Id elementId: startLoc->getTokenIds())
						{
							const StorageEdge edge = storage.getEdgeById(elementId);
							if (edge.id != 0)	 // for node elements this condition will fail
							{
								if (Edge::intToType(edge.type) == Edge::EDGE_INHERITANCE)
								{
									if (intToNodeKind(targetNode.type) == NODE_CLASS)
									{
										const NameHierarchy chileName =
											storage.getNameHierarchyForNodeId(edge.sourceNodeId);
										const NameHierarchy parentName = NameHierarchy::deserialize(
											targetNode.serializedName);
										childToParentNodesMap[chileName.back().getName()].push_back(
											parentName.back().getName());
									}
									else
									{
										continue;
									}
								}

								dataToInsert.push_back(
									{StorageEdgeData(edge.type, edge.sourceNodeId, targetNode.id),
									 startLoc->getLocationId()});
								occurrencesToDelete.push_back(
									StorageOccurrence(edge.id, startLoc->getLocationId()));
							}
						}
					}
				}
			});
		}
	});

	storage.setMode(SqliteIndexStorage::STORAGE_MODE_WRITE);

	storage.startInjection();

	std::vector<StorageEdge> edgesToInsert;
	for (const DataToInsert& data: dataToInsert)
	{
		edgesToInsert.push_back(StorageEdge(0, data.edgeData));
	}
	const std::vector<Id> ambiguousEdgeIds = storage.addEdges(edgesToInsert);

	if (ambiguousEdgeIds.size() == dataToInsert.size())
	{
		for (size_t i = 0; i < ambiguousEdgeIds.size(); i++)
		{
			storage.addElementComponent(StorageElementComponent(
				ambiguousEdgeIds[i],
				elementComponentKindToInt(ElementComponentKind::IS_AMBIGUOUS),
				L""));
			storage.addOccurrence(
				StorageOccurrence(ambiguousEdgeIds[i], dataToInsert[i].sourceLocationId));
		}
		storage.setMode(SqliteIndexStorage::STORAGE_MODE_CLEAR);

		storage.removeOccurrences(occurrencesToDelete);
		std::set<Id> edgeIds;
		for (const StorageOccurrence& occurrence: occurrencesToDelete)
		{
			edgeIds.insert(occurrence.elementId);
		}
		storage.removeElementsWithoutOccurrences(utility::toVector(edgeIds));

		storage.finishInjection();
		LOG_INFO("Finished Python post processing.");
	}
	else
	{
		LOG_ERROR("Error occurred while running Python post processing. Rolling back all changes.");
		storage.rollbackInjection();
	}
}

TaskExecuteCustomCommands::TaskExecuteCustomCommands(
	std::unique_ptr<IndexerCommandProvider> indexerCommandProvider,
	std::shared_ptr<PersistentStorage> storage,
	std::shared_ptr<DialogView> dialogView,
	size_t indexerThreadCount,
	const FilePath& projectDirectory)
	: m_indexerCommandProvider(std::move(indexerCommandProvider))
	, m_storage(storage)
	, m_dialogView(dialogView)
	, m_indexerThreadCount(indexerThreadCount)
	, m_projectDirectory(projectDirectory)
	, m_indexerCommandCount(m_indexerCommandProvider->size())
	, m_hasPythonCommands(false)
{
}

void TaskExecuteCustomCommands::doEnter(std::shared_ptr<Blackboard> blackboard)
{
	m_dialogView->hideUnknownProgressDialog();
	m_start = TimeStamp::now();

	if (m_indexerCommandProvider)
	{
		for (const FilePath& sourceFilePath:
			 utility::partitionFilePathsBySize(m_indexerCommandProvider->getAllSourceFilePaths(), 2))
		{
			if (std::shared_ptr<IndexerCommandCustom> indexerCommand =
					std::dynamic_pointer_cast<IndexerCommandCustom>(
						m_indexerCommandProvider->consumeCommandForSourceFilePath(sourceFilePath)))
			{
				if (m_targetDatabaseFilePath.empty())
				{
					m_targetDatabaseFilePath = indexerCommand->getDatabaseFilePath();
				}
#if BUILD_PYTHON_LANGUAGE_PACKAGE
				if (indexerCommand->getIndexerCommandType() == INDEXER_COMMAND_PYTHON)
				{
					m_hasPythonCommands = true;
				}
#endif	  // BUILD_PYTHON_LANGUAGE_PACKAGE

				if (indexerCommand->getRunInParallel())
				{
					m_parallelCommands.push_back(indexerCommand);
				}
				else
				{
					m_serialCommands.push_back(indexerCommand);
				}
			}
		}
		// reverse because we pull elements from the back of these vectors
		std::reverse(m_parallelCommands.begin(), m_parallelCommands.end());
		std::reverse(m_serialCommands.begin(), m_serialCommands.end());
	}
}

Task::TaskState TaskExecuteCustomCommands::doUpdate(std::shared_ptr<Blackboard> blackboard)
{
	if (m_interrupted)
	{
		return STATE_SUCCESS;
	}

	m_dialogView->updateCustomIndexingDialog(0, 0, m_indexerCommandProvider->size(), {});

	std::vector<std::shared_ptr<std::thread>> indexerThreads;
	for (size_t i = 1 /*this method is counting as the first thread*/; i < m_indexerThreadCount; i++)
	{
		indexerThreads.push_back(std::make_shared<std::thread>(
			&TaskExecuteCustomCommands::executeParallelIndexerCommands,
			this,
			static_cast<int>(i),
			blackboard));
	}

	while (!m_interrupted && !m_serialCommands.empty())
	{
		std::shared_ptr<IndexerCommandCustom> indexerCommand = m_serialCommands.back();
		m_serialCommands.pop_back();
		runIndexerCommand(indexerCommand, blackboard, m_storage);
	}

	executeParallelIndexerCommands(0, blackboard);

	for (std::shared_ptr<std::thread> indexerThread: indexerThreads)
	{
		indexerThread->join();
	}
	indexerThreads.clear();

	// clear errors here, because otherwise injecting into the main storage will show them twice
	MessageErrorCountClear().dispatch();

	{
		PersistentStorage targetStorage(m_targetDatabaseFilePath, FilePath());
		targetStorage.setup();
		targetStorage.setMode(SqliteIndexStorage::STORAGE_MODE_WRITE);
		targetStorage.buildCaches();
		for (const FilePath& sourceDatabaseFilePath: m_sourceDatabaseFilePaths)
		{
			{
				PersistentStorage sourceStorage(sourceDatabaseFilePath, FilePath());
				sourceStorage.setMode(SqliteIndexStorage::STORAGE_MODE_READ);
				sourceStorage.buildCaches();
				targetStorage.inject(&sourceStorage);
			}
			FileSystem::remove(sourceDatabaseFilePath);
		}

		if (m_hasPythonCommands &&
			ApplicationSettings::getInstance()->getPythonPostProcessingEnabled())
		{
			LOG_INFO("Starting Python post processing.");
			m_dialogView->showUnknownProgressDialog(
				L"Finish Indexing", L"Run Python Post Processing");

			targetStorage.clearCaches();
			targetStorage.buildCaches();
			runPythonPostProcessing(targetStorage);
		}
	}

	return STATE_SUCCESS;
}

void TaskExecuteCustomCommands::doExit(std::shared_ptr<Blackboard> blackboard)
{
	m_storage.reset();
	const float duration = static_cast<float>(TimeStamp::durationSeconds(m_start));
	blackboard->update<float>(
		"index_time", [duration](float currentDuration) { return currentDuration + duration; });
}

void TaskExecuteCustomCommands::doReset(std::shared_ptr<Blackboard> blackboard) {}

void TaskExecuteCustomCommands::handleMessage(MessageIndexingInterrupted* message)
{
	LOG_INFO("Interrupting custom command execution.");

	m_interrupted = true;

	m_dialogView->showUnknownProgressDialog(
		L"Interrupting Indexing", L"Waiting for running\ncommand to finish");
}

void TaskExecuteCustomCommands::executeParallelIndexerCommands(
	int threadId, std::shared_ptr<Blackboard> blackboard)
{
	std::shared_ptr<PersistentStorage> storage;
	while (!m_interrupted)
	{
		std::shared_ptr<IndexerCommandCustom> indexerCommand;
		{
			std::lock_guard<std::mutex> lock(m_parallelCommandsMutex);
			if (m_parallelCommands.empty())
			{
				return;
			}
			indexerCommand = m_parallelCommands.back();
			m_parallelCommands.pop_back();
		}

		if (threadId == 0)
		{
			storage = m_storage;
		}
		else
		{
			FilePath databaseFilePath = indexerCommand->getDatabaseFilePath();
			databaseFilePath = databaseFilePath.getParentDirectory().concatenate(
				databaseFilePath.fileName() + L"_thread" + std::to_wstring(threadId));

			bool databaseFilePathKnown = true;
			{
				std::lock_guard<std::mutex> lock(m_sourceDatabaseFilePathsMutex);
				if (m_sourceDatabaseFilePaths.find(databaseFilePath) ==
					m_sourceDatabaseFilePaths.end())
				{
					m_sourceDatabaseFilePaths.insert(databaseFilePath);
					databaseFilePathKnown = false;
				}
			}

			if (!databaseFilePathKnown)
			{
				if (databaseFilePath.exists())
				{
					LOG_WARNING(
						L"Temporary storage \"" + databaseFilePath.wstr() +
						L"\" already exists on file system. File will be removed to avoid "
						L"conflicts.");
					FileSystem::remove(databaseFilePath);
				}
				storage = std::make_shared<PersistentStorage>(databaseFilePath, FilePath());
				storage->setup();
				storage->setMode(SqliteIndexStorage::STORAGE_MODE_WRITE);
				storage->buildCaches();
			}

			indexerCommand->setDatabaseFilePath(databaseFilePath);
		}

		runIndexerCommand(indexerCommand, blackboard, storage);
	}
}

void TaskExecuteCustomCommands::runIndexerCommand(
	std::shared_ptr<IndexerCommandCustom> indexerCommand,
	std::shared_ptr<Blackboard> blackboard,
	std::shared_ptr<PersistentStorage> storage)
{
	if (indexerCommand)
	{
		int indexedSourceFileCount = 0;
		blackboard->get("indexed_source_file_count", indexedSourceFileCount);

		const FilePath sourcePath = indexerCommand->getSourceFilePath();

		m_dialogView->updateCustomIndexingDialog(
			indexedSourceFileCount + 1, indexedSourceFileCount, m_indexerCommandCount, {sourcePath});
		MessageIndexingStatus(true, indexedSourceFileCount * 100 / m_indexerCommandCount).dispatch();

		const std::wstring command = indexerCommand->getCommand();
		const std::vector<std::wstring> arguments = indexerCommand->getArguments();

		LOG_INFO(
			"Start processing command \"" +
			utility::encodeToUtf8(command + L" " + utility::join(arguments, L" ")) + "\"");

		const ErrorCountInfo previousErrorCount = storage ? storage->getErrorCount()
														  : ErrorCountInfo();

		LOG_INFO("Starting to index");
		const utility::ProcessOutput out = utility::executeProcess(
			command, arguments, m_projectDirectory, false, -1, true);
		LOG_INFO("Finished indexing");

		if (storage)
		{
			std::vector<ErrorInfo> errors = storage->getErrorInfos();
			const ErrorCountInfo currentErrorCount(errors);
			if (currentErrorCount.total > previousErrorCount.total)
			{
				const ErrorCountInfo diff(
					currentErrorCount.total - previousErrorCount.total,
					currentErrorCount.fatal - previousErrorCount.fatal);

				ErrorCountInfo errorCount;	  // local copy to release lock early
				{
					std::lock_guard<std::mutex> lock(m_errorCountMutex);
					m_errorCount.total += diff.total;
					m_errorCount.fatal += diff.fatal;
					errorCount = m_errorCount;
				}

				errors.erase(errors.begin(), errors.begin() + previousErrorCount.total);
				MessageErrorCountUpdate(errorCount, errors).dispatch();
			}
		}

		if (out.exitCode == 0 && out.error.empty())
		{
			std::wstring message = L"Process returned successfully.\n";
			LOG_INFO(message);
		}
		else
		{
			std::wstring statusText = L"command \"" + indexerCommand->getCommand() + L" " +
				utility::join(arguments, L" ") + L"\" returned";
			if (out.exitCode != 0)
			{
				statusText += L" code \"" + std::to_wstring(out.exitCode) + L"\"";
			}
			if (!out.error.empty())
			{
				statusText += L" with message \"" + out.error + L"\"";
			}
			statusText += L".";

			LOG_ERROR(statusText);
			MessageShowStatus().dispatch();
			MessageStatus(statusText, true, false, true).dispatch();
		}

		indexedSourceFileCount++;
		blackboard->update<int>("indexed_source_file_count", [=](int count) { return count + 1; });
	}
}
