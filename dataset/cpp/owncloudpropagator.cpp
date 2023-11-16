/*
 * Copyright (C) by Olivier Goffart <ogoffart@owncloud.com>
 * Copyright (C) by Klaas Freitag <freitag@owncloud.com>
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
 * or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License
 * for more details.
 */

#include "owncloudpropagator.h"
#include "common/syncjournaldb.h"
#include "common/syncjournalfilerecord.h"
#include "propagatedownload.h"
#include "propagateupload.h"
#include "propagateremotedelete.h"
#include "propagateremotemove.h"
#include "propagateremotemkdir.h"
#include "bulkpropagatorjob.h"
#include "updatefiledropmetadata.h"
#include "propagatorjobs.h"
#include "filesystem.h"
#include "common/utility.h"
#include "account.h"
#include "common/asserts.h"
#include "discoveryphase.h"
#include "syncfileitem.h"

#ifdef Q_OS_WIN
#include <windef.h>
#include <winbase.h>
#endif

#include <QStack>
#include <QFileInfo>
#include <QDir>
#include <QLoggingCategory>
#include <QTimer>
#include <QObject>
#include <QTimerEvent>
#include <QRegularExpression>
#include <qmath.h>

namespace OCC {

Q_LOGGING_CATEGORY(lcPropagator, "nextcloud.sync.propagator", QtInfoMsg)
Q_LOGGING_CATEGORY(lcDirectory, "nextcloud.sync.propagator.directory", QtInfoMsg)
Q_LOGGING_CATEGORY(lcRootDirectory, "nextcloud.sync.propagator.root.directory", QtInfoMsg)
Q_LOGGING_CATEGORY(lcCleanupPolls, "nextcloud.sync.propagator.cleanuppolls", QtInfoMsg)

qint64 criticalFreeSpaceLimit()
{
    qint64 value = 50 * 1000 * 1000LL;

    static bool hasEnv = false;
    static qint64 env = qgetenv("OWNCLOUD_CRITICAL_FREE_SPACE_BYTES").toLongLong(&hasEnv);
    if (hasEnv) {
        value = env;
    }

    return qBound(0LL, value, freeSpaceLimit());
}

qint64 freeSpaceLimit()
{
    qint64 value = 250 * 1000 * 1000LL;

    static bool hasEnv = false;
    static qint64 env = qgetenv("OWNCLOUD_FREE_SPACE_BYTES").toLongLong(&hasEnv);
    if (hasEnv) {
        value = env;
    }

    return value;
}

OwncloudPropagator::~OwncloudPropagator() = default;


int OwncloudPropagator::maximumActiveTransferJob()
{
    if (_downloadLimit != 0
        || _uploadLimit != 0
        || !_syncOptions._parallelNetworkJobs) {
        // disable parallelism when there is a network limit.
        return 1;
    }
    return qMin(3, qCeil(_syncOptions._parallelNetworkJobs / 2.));
}

/* The maximum number of active jobs in parallel  */
int OwncloudPropagator::hardMaximumActiveJob()
{
    if (!_syncOptions._parallelNetworkJobs)
        return 1;
    return _syncOptions._parallelNetworkJobs;
}

PropagateItemJob::~PropagateItemJob()
{
    if (auto p = propagator()) {
        // Normally, every job should clean itself from the _activeJobList. So this should not be
        // needed. But if a job has a bug or is deleted before the network jobs signal get received,
        // we might risk end up with dangling pointer in the list which may cause crashes.
        p->_activeJobList.removeAll(this);
    }
}

static qint64 getMinBlacklistTime()
{
    return qMax(qEnvironmentVariableIntValue("OWNCLOUD_BLACKLIST_TIME_MIN"),
        25); // 25 seconds
}

static qint64 getMaxBlacklistTime()
{
    int v = qEnvironmentVariableIntValue("OWNCLOUD_BLACKLIST_TIME_MAX");
    if (v > 0)
        return v;
    return 24 * 60 * 60; // 1 day
}

/** Creates a blacklist entry, possibly taking into account an old one.
 *
 * The old entry may be invalid, then a fresh entry is created.
 */
static SyncJournalErrorBlacklistRecord createBlacklistEntry(
    const SyncJournalErrorBlacklistRecord &old, const SyncFileItem &item)
{
    SyncJournalErrorBlacklistRecord entry;
    entry._file = item._file;
    entry._errorString = item._errorString;
    entry._lastTryModtime = item._modtime;
    entry._lastTryEtag = item._etag;
    entry._lastTryTime = Utility::qDateTimeToTime_t(QDateTime::currentDateTimeUtc());
    entry._renameTarget = item._renameTarget;
    entry._retryCount = old._retryCount + 1;
    entry._requestId = item._requestId;

    static qint64 minBlacklistTime(getMinBlacklistTime());
    static qint64 maxBlacklistTime(qMax(getMaxBlacklistTime(), minBlacklistTime));

    // The factor of 5 feels natural: 25s, 2 min, 10 min, ~1h, ~5h, ~24h
    entry._ignoreDuration = old._ignoreDuration * 5;

    if (item._httpErrorCode == 403) {
        qCWarning(lcPropagator) << "Probably firewall error: " << item._httpErrorCode << ", blacklisting up to 1h only";
        entry._ignoreDuration = qMin(entry._ignoreDuration, qint64(60 * 60));

    } else if (item._httpErrorCode == 413 || item._httpErrorCode == 415) {
        qCWarning(lcPropagator) << "Fatal Error condition" << item._httpErrorCode << ", maximum blacklist ignore time!";
        entry._ignoreDuration = maxBlacklistTime;
    }

    entry._ignoreDuration = qBound(minBlacklistTime, entry._ignoreDuration, maxBlacklistTime);

    if (item._status == SyncFileItem::SoftError) {
        // Track these errors, but don't actively suppress them.
        entry._ignoreDuration = 0;
    }

    if (item._httpErrorCode == 507) {
        entry._errorCategory = SyncJournalErrorBlacklistRecord::InsufficientRemoteStorage;
    }

    return entry;
}

/** Updates, creates or removes a blacklist entry for the given item.
 *
 * May adjust the status or item._errorString.
 */
void blacklistUpdate(SyncJournalDb *journal, SyncFileItem &item)
{
    SyncJournalErrorBlacklistRecord oldEntry = journal->errorBlacklistEntry(item._file);

    bool mayBlacklist =
        item._errorMayBeBlacklisted // explicitly flagged for blacklisting
        || ((item._status == SyncFileItem::NormalError
                || item._status == SyncFileItem::SoftError
                || item._status == SyncFileItem::DetailError)
               && item._httpErrorCode != 0 // or non-local error
               );

    // No new entry? Possibly remove the old one, then done.
    if (!mayBlacklist) {
        if (oldEntry.isValid()) {
            journal->wipeErrorBlacklistEntry(item._file);
        }
        return;
    }

    auto newEntry = createBlacklistEntry(oldEntry, item);
    journal->setErrorBlacklistEntry(newEntry);

    // Suppress the error if it was and continues to be blacklisted.
    // An ignoreDuration of 0 mean we're tracking the error, but not actively
    // suppressing it.
    if (item._hasBlacklistEntry && newEntry._ignoreDuration > 0) {
        item._status = SyncFileItem::BlacklistedError;

        qCInfo(lcPropagator) << "blacklisting " << item._file
                             << " for " << newEntry._ignoreDuration
                             << ", retry count " << newEntry._retryCount;

        return;
    }

    // Some soft errors might become louder on repeat occurrence
    if (item._status == SyncFileItem::SoftError
        && newEntry._retryCount > 1) {
        qCWarning(lcPropagator) << "escalating soft error on " << item._file
                                << " to normal error, " << item._httpErrorCode;
        item._status = SyncFileItem::NormalError;
        return;
    }
}

void PropagateItemJob::done(const SyncFileItem::Status statusArg, const QString &errorString, const ErrorCategory category)
{
    // Duplicate calls to done() are a logic error
    ENFORCE(_state != Finished);
    _state = Finished;

    _item->_status = statusArg;

    if (_item->_isRestoration) {
        if (_item->_status == SyncFileItem::Success
            || _item->_status == SyncFileItem::Conflict) {
            _item->_status = SyncFileItem::Restoration;
        } else {
            _item->_errorString += tr("; Restoration Failed: %1").arg(errorString);
        }
    } else {
        if (_item->_errorString.isEmpty()) {
            _item->_errorString = errorString;
        }
    }

    if (propagator()->_abortRequested && (_item->_status == SyncFileItem::NormalError
                                          || _item->_status == SyncFileItem::FatalError)) {
        // an abort request is ongoing. Change the status to Soft-Error
        _item->_status = SyncFileItem::SoftError;
    }

    // Blacklist handling
    switch (_item->_status) {
    case SyncFileItem::SoftError:
    case SyncFileItem::FatalError:
    case SyncFileItem::NormalError:
    case SyncFileItem::DetailError:
        // Check the blacklist, possibly adjusting the item (including its status)
        blacklistUpdate(propagator()->_journal, *_item);
        break;
    case SyncFileItem::Success:
    case SyncFileItem::Restoration:
        if (_item->_hasBlacklistEntry) {
            // wipe blacklist entry.
            propagator()->_journal->wipeErrorBlacklistEntry(_item->_file);
            // remove a blacklist entry in case the file was moved.
            if (_item->_originalFile != _item->_file) {
                propagator()->_journal->wipeErrorBlacklistEntry(_item->_originalFile);
            }
        }
        break;
    case SyncFileItem::Conflict:
    case SyncFileItem::FileIgnored:
    case SyncFileItem::NoStatus:
    case SyncFileItem::BlacklistedError:
    case SyncFileItem::FileLocked:
    case SyncFileItem::FileNameInvalid:
    case SyncFileItem::FileNameInvalidOnServer:
    case SyncFileItem::FileNameClash:
        // nothing
        break;
    }

    if (_item->hasErrorStatus())
        qCWarning(lcPropagator) << "Could not complete propagation of" << _item->destination() << "by" << this << "with status" << _item->_status << "and error:" << _item->_errorString;
    else
        qCInfo(lcPropagator) << "Completed propagation of" << _item->destination() << "by" << this << "with status" << _item->_status;
    emit propagator()->itemCompleted(_item, category);
    emit finished(_item->_status);

    if (_item->_status == SyncFileItem::FatalError) {
        // Abort all remaining jobs.
        propagator()->abort();
    }
}

void PropagateItemJob::slotRestoreJobFinished(SyncFileItem::Status status)
{
    QString msg;
    if (_restoreJob) {
        msg = _restoreJob->restoreJobMsg();
        _restoreJob->setRestoreJobMsg();
    }

    if (status == SyncFileItem::Success || status == SyncFileItem::Conflict
        || status == SyncFileItem::Restoration) {
        done(SyncFileItem::SoftError, msg, ErrorCategory::GenericError);
    } else {
        done(status, tr("A file or folder was removed from a read only share, but restoring failed: %1").arg(msg), ErrorCategory::GenericError);
    }
}

bool PropagateItemJob::hasEncryptedAncestor() const
{
    if (!propagator()->account()->capabilities().clientSideEncryptionAvailable()) {
        return false;
    }

    const auto path = _item->_file;
    const auto slashPosition = path.lastIndexOf('/');
    const auto parentPath = slashPosition >= 0 ? path.left(slashPosition) : QString();

    auto pathComponents = parentPath.split('/');
    while (!pathComponents.isEmpty()) {
        SyncJournalFileRecord rec;
        const auto pathCompontentsJointed = pathComponents.join('/');
        if (!propagator()->_journal->getFileRecord(pathCompontentsJointed, &rec)) {
            qCWarning(lcPropagator) << "could not get file from local DB" << pathCompontentsJointed;
        }

        if (rec.isValid() && rec.isE2eEncrypted()) {
            return true;
        }
        pathComponents.removeLast();
    }

    return false;
}

// ================================================================================

PropagateItemJob *OwncloudPropagator::createJob(const SyncFileItemPtr &item)
{
    bool deleteExisting = item->_instruction == CSYNC_INSTRUCTION_TYPE_CHANGE;
    switch (item->_instruction) {
    case CSYNC_INSTRUCTION_REMOVE:
        if (item->_direction == SyncFileItem::Down)
            return new PropagateLocalRemove(this, item);
        else
            return new PropagateRemoteDelete(this, item);
    case CSYNC_INSTRUCTION_NEW:
    case CSYNC_INSTRUCTION_TYPE_CHANGE:
    case CSYNC_INSTRUCTION_CONFLICT:
        if (item->isDirectory()) {
            // CONFLICT has _direction == None
            if (item->_direction != SyncFileItem::Up) {
                auto job = new PropagateLocalMkdir(this, item);
                job->setDeleteExistingFile(deleteExisting);
                return job;
            } else {
                auto job = new PropagateRemoteMkdir(this, item);
                job->setDeleteExisting(deleteExisting);
                return job;
            }
        } //fall through
    case CSYNC_INSTRUCTION_SYNC:
        if (item->_direction != SyncFileItem::Up) {
            auto job = new PropagateDownloadFile(this, item);
            job->setDeleteExistingFolder(deleteExisting);
            return job;
        } else {
            if (deleteExisting || !isDelayedUploadItem(item)) {
                auto job = createUploadJob(item, deleteExisting);
                return job.release();
            } else {
                pushDelayedUploadTask(item);
                return nullptr;
            }
        }
    case CSYNC_INSTRUCTION_RENAME:
        if (item->_direction == SyncFileItem::Up) {
            return new PropagateRemoteMove(this, item);
        } else {
            return new PropagateLocalRename(this, item);
        }
    case CSYNC_INSTRUCTION_IGNORE:
    case CSYNC_INSTRUCTION_ERROR:
        return new PropagateIgnoreJob(this, item);
    default:
        return nullptr;
    }
    return nullptr;
}

std::unique_ptr<PropagateUploadFileCommon> OwncloudPropagator::createUploadJob(SyncFileItemPtr item, bool deleteExisting)
{
    auto job = std::unique_ptr<PropagateUploadFileCommon>{};

    if (item->_size > syncOptions()._initialChunkSize && account()->capabilities().chunkingNg()) {
        // Item is above _initialChunkSize, thus will be classified as to be chunked
        job = std::make_unique<PropagateUploadFileNG>(this, item);
    } else {
        job = std::make_unique<PropagateUploadFileV1>(this, item);
    }

    job->setDeleteExisting(deleteExisting);

    removeFromBulkUploadBlackList(item->_file);

    return job;
}

void OwncloudPropagator::pushDelayedUploadTask(SyncFileItemPtr item)
{
    _delayedTasks.push_back(item);
}

void OwncloudPropagator::resetDelayedUploadTasks()
{
    _scheduleDelayedTasks = false;
    _delayedTasks.clear();
}

void OwncloudPropagator::adjustDeletedFoldersWithNewChildren(SyncFileItemVector &items)
{
    /* 
       process each item that is new and is a directory and make sure every parent in its tree has the instruction CSYNC_INSTRUCTION_NEW
       instead of CSYNC_INSTRUCTION_REMOVE
       NOTE: We are iterating backwards to take advantage of optimization later, when searching for the parent of current it
    */
    for (auto it = std::crbegin(items); it != std::crend(items); ++it) {
        if ((*it)->_instruction != CSYNC_INSTRUCTION_NEW || (*it)->_direction != SyncFileItem::Up || !(*it)->isDirectory() || (*it)->_file == QStringLiteral("/")) {
            continue;
        }

        // #1 get root folder name for the current item that we need to reupload
        const auto folderPathSplit = (*it)->_file.split(QLatin1Char('/'), Qt::SkipEmptyParts);
        if (folderPathSplit.isEmpty()) {
            continue;
        }
        const auto itemRootFolderName = folderPathSplit.first();
        if (itemRootFolderName.isEmpty()) {
            continue;
        }
        // #2 iterate backwards (for optimization) and find the root folder by name
        const auto itemRootFolderReverseIt = std::find_if(it, std::crend(items), [&itemRootFolderName](const auto &currentItem) {
            return currentItem->_file == itemRootFolderName;
        });

        if (itemRootFolderReverseIt == std::rend(items)) {
            continue;
        }

        // #3 convert reverse iterator to normal iterator
        const auto itemFolderIt = (itemRootFolderReverseIt + 1).base();

        // #4 if the root folder is set to be removed, then we will need to fix this by reuploading every folder in
        // the tree, including the root
        if (itemFolderIt == std::end(items)) {
            continue;
        }

        auto nextFolderInTreeIt = itemFolderIt;
        do {
            // #5 Iterate forward from the CSYNC_INSTRUCTION_NEW folder's root, and make sure every folder in it's tree is set to CSYNC_INSTRUCTION_NEW
            if ((*nextFolderInTreeIt)->isDirectory()
                && (*nextFolderInTreeIt)->_instruction == CSYNC_INSTRUCTION_REMOVE
                && (*nextFolderInTreeIt)->_direction == SyncFileItem::Down
                && (*it)->_file.startsWith(QString((*nextFolderInTreeIt)->_file) + QLatin1Char('/'))) {

                qCWarning(lcPropagator) << "WARNING: New directory to upload " << (*it)->_file
                    << "is in the removed directories tree " << (*nextFolderInTreeIt)->_file
                    << " This should not happen! But, we are going to reupload the entire folder structure.";

                (*nextFolderInTreeIt)->_instruction = CSYNC_INSTRUCTION_NEW;
                (*nextFolderInTreeIt)->_direction = SyncFileItem::Up;
            }
            ++nextFolderInTreeIt;
        } while (nextFolderInTreeIt != std::end(items) && (*nextFolderInTreeIt)->_file != (*it)->_file);
    }
}

qint64 OwncloudPropagator::smallFileSize()
{
    const qint64 smallFileSize = 100 * 1024; //default to 1 MB. Not dynamic right now.
    return smallFileSize;
}

void OwncloudPropagator::start(SyncFileItemVector &&items)
{
    Q_ASSERT(std::is_sorted(items.begin(), items.end()));

    _abortRequested = false;

    /* This builds all the jobs needed for the propagation.
     * Each directory is a PropagateDirectory job, which contains the files in it.
     * In order to do that we loop over the items. (which are sorted by destination)
     * When we enter a directory, we can create the directory job and push it on the stack. */

    const auto regex = syncOptions().fileRegex();
    if (regex.isValid()) {
        QSet<QStringRef> names;
        for (auto &i : items) {
            if (regex.match(i->_file).hasMatch()) {
                int index = -1;
                QStringRef ref;
                do {
                    ref = i->_file.midRef(0, index);
                    names.insert(ref);
                    index = ref.lastIndexOf(QLatin1Char('/'));
                } while (index > 0);
            }
        }
        items.erase(std::remove_if(items.begin(),
                                   items.end(),
                                   [&names](auto i) {
                                       return !names.contains(QStringRef{&i->_file});
                                   }),
                    items.end());
    }

    QStringList files;

    for (const auto &item : items) {
        files.push_back(item->_file);
    }

    // process each item that is new and is a directory and make sure every parent in its tree has the instruction NEW instead of REMOVE
    adjustDeletedFoldersWithNewChildren(items);

    resetDelayedUploadTasks();
    _rootJob.reset(new PropagateRootDirectory(this));
    QStack<QPair<QString /* directory name */, PropagateDirectory * /* job */>> directories;
    directories.push(qMakePair(QString(), _rootJob.data()));
    QVector<PropagatorJob *> directoriesToRemove;
    QString removedDirectory;
    QString maybeConflictDirectory;
    foreach (const SyncFileItemPtr &item, items) {
        if (!removedDirectory.isEmpty() && item->_file.startsWith(removedDirectory)) {
            // this is an item in a directory which is going to be removed.
            auto *delDirJob = qobject_cast<PropagateDirectory *>(directoriesToRemove.first());

            const auto isNewDirectory = item->isDirectory() &&
                    (item->_instruction == CSYNC_INSTRUCTION_NEW || item->_instruction == CSYNC_INSTRUCTION_TYPE_CHANGE);

            if (item->_instruction == CSYNC_INSTRUCTION_REMOVE || isNewDirectory) {
                // If it is a remove it is already taken care of by the removal of the parent directory

                // If it is a new directory then it is inside a deleted directory... That can happen if
                // the directory etag was not fetched properly on the previous sync because the sync was
                // aborted while uploading this directory (which is now removed).  We can ignore it.

                // increase the number of subjobs that would be there.
                if (delDirJob) {
                    delDirJob->increaseAffectedCount();
                }
                continue;
            } else if (item->_instruction == CSYNC_INSTRUCTION_IGNORE) {
                continue;
            } else if (item->_instruction == CSYNC_INSTRUCTION_RENAME) {
                // all is good, the rename will be executed before the directory deletion
            } else {
                qCWarning(lcPropagator) << "WARNING:  Job within a removed directory?  This should not happen!"
                                        << item->_file << item->_instruction;
            }
        }

        // If a CONFLICT item contains files these can't be processed because
        // the conflict handling is likely to rename the directory. This can happen
        // when there's a new local directory at the same time as a remote file.
        if (!maybeConflictDirectory.isEmpty()) {
            if (item->destination().startsWith(maybeConflictDirectory)) {
                qCInfo(lcPropagator) << "Skipping job inside CONFLICT directory"
                                     << item->_file << item->_instruction;
                item->_instruction = CSYNC_INSTRUCTION_NONE;
                continue;
            } else {
                maybeConflictDirectory.clear();
            }
        }

        while (!item->destination().startsWith(directories.top().first)) {
            directories.pop();
        }

        if (item->isDirectory()) {
            startDirectoryPropagation(item,
                                      directories,
                                      directoriesToRemove,
                                      removedDirectory,
                                      items);
        } else if (!directories.top().second->_item->_isFileDropDetected) {
            startFilePropagation(item,
                                 directories,
                                 directoriesToRemove,
                                 removedDirectory,
                                 maybeConflictDirectory);
        }
    }

    foreach (PropagatorJob *it, directoriesToRemove) {
        _rootJob->appendDirDeletionJob(it);
    }

    connect(_rootJob.data(), &PropagatorJob::finished, this, &OwncloudPropagator::emitFinished);

    _jobScheduled = false;
    scheduleNextJob();
}

void OwncloudPropagator::startDirectoryPropagation(const SyncFileItemPtr &item,
                                                   QStack<QPair<QString, PropagateDirectory *>> &directories,
                                                   QVector<PropagatorJob *> &directoriesToRemove,
                                                   QString &removedDirectory,
                                                   const SyncFileItemVector &items)
{
    auto directoryPropagationJob = std::make_unique<PropagateDirectory>(this, item);

    if (item->_instruction == CSYNC_INSTRUCTION_TYPE_CHANGE
        && item->_direction == SyncFileItem::Up) {
        // Skip all potential uploads to the new folder.
        // Processing them now leads to problems with permissions:
        // checkForPermissions() has already run and used the permissions
        // of the file we're about to delete to decide whether uploading
        // to the new dir is ok...
        foreach (const SyncFileItemPtr &dirItem, items) {
            if (dirItem->destination().startsWith(item->destination() + "/")) {
                dirItem->_instruction = CSYNC_INSTRUCTION_NONE;
                _anotherSyncNeeded = true;
            }
        }
    }

    if (item->_instruction == CSYNC_INSTRUCTION_REMOVE) {
        // We do the removal of directories at the end, because there might be moves from
        // these directories that will happen later.
        directoriesToRemove.prepend(directoryPropagationJob.get());
        removedDirectory = item->_file + "/";

        // We should not update the etag of parent directories of the removed directory
        // since it would be done before the actual remove (issue #1845)
        // NOTE: Currently this means that we don't update those etag at all in this sync,
        //       but it should not be a problem, they will be updated in the next sync.
        for (int i = 0; i < directories.size(); ++i) {
            if (directories[i].second->_item->_instruction == CSYNC_INSTRUCTION_UPDATE_METADATA) {
                directories[i].second->_item->_instruction = CSYNC_INSTRUCTION_NONE;
            }
        }
    } else {
        const auto currentDirJob = directories.top().second;
        currentDirJob->appendJob(directoryPropagationJob.get());
    }
    if (item->_isFileDropDetected) {
        directoryPropagationJob->appendJob(new UpdateFileDropMetadataJob(this, item->_file));
        item->_instruction = CSYNC_INSTRUCTION_NONE;
        _anotherSyncNeeded = true;
    } else if (item->_isEncryptedMetadataNeedUpdate) {
        SyncJournalFileRecord record;
        const auto isUnexpectedMetadataFormat = _journal->getFileRecord(item->_file, &record)
            && record._e2eEncryptionStatus == SyncJournalFileRecord::EncryptionStatus::EncryptedMigratedV1_2;
        if (isUnexpectedMetadataFormat && _account->shouldSkipE2eeMetadataChecksumValidation()) {
            qCDebug(lcPropagator) << "Getting unexpected metadata format, but allowing to continue as shouldSkipE2eeMetadataChecksumValidation is set.";
        } else if (isUnexpectedMetadataFormat && !_account->shouldSkipE2eeMetadataChecksumValidation()) {
            qCDebug(lcPropagator) << "could have upgraded metadata";
            item->_instruction = CSyncEnums::CSYNC_INSTRUCTION_ERROR;
            item->_errorString = tr("Error with the metadata. Getting unexpected metadata format.");
            item->_status = SyncFileItem::NormalError;
            emit itemCompleted(item, OCC::ErrorCategory::GenericError);
        } else {
            directoryPropagationJob->appendJob(new UpdateFileDropMetadataJob(this, item->_file));
            item->_instruction = CSYNC_INSTRUCTION_NONE;
            _anotherSyncNeeded = true;
        }
    }
    directories.push(qMakePair(item->destination() + "/", directoryPropagationJob.release()));
}

void OwncloudPropagator::startFilePropagation(const SyncFileItemPtr &item,
                                              QStack<QPair<QString, PropagateDirectory *> > &directories,
                                              QVector<PropagatorJob *> &directoriesToRemove,
                                              QString &removedDirectory,
                                              QString &maybeConflictDirectory)
{
    if (item->_instruction == CSYNC_INSTRUCTION_TYPE_CHANGE) {
        // will delete directories, so defer execution
        auto job = createJob(item);
        if (job) {
            directoriesToRemove.prepend(job);
        }
        removedDirectory = item->_file + "/";
    } else {
        directories.top().second->appendTask(item);
    }

    if (item->_instruction == CSYNC_INSTRUCTION_CONFLICT) {
        // This might be a file or a directory on the local side. If it's a
        // directory we want to skip processing items inside it.
        maybeConflictDirectory = item->_file + "/";
    }
}

const SyncOptions &OwncloudPropagator::syncOptions() const
{
    return _syncOptions;
}

void OwncloudPropagator::setSyncOptions(const SyncOptions &syncOptions)
{
    _syncOptions = syncOptions;
    _chunkSize = syncOptions._initialChunkSize;
}

bool OwncloudPropagator::localFileNameClash(const QString &relFile)
{
    const QString file(_localDir + relFile);
    Q_ASSERT(!file.isEmpty());

    if (!file.isEmpty() && Utility::fsCasePreserving()) {
#ifdef Q_OS_MAC
        const QFileInfo fileInfo(file);
        if (!fileInfo.exists()) {
            return false;
        } else {
            // Need to normalize to composited form because of QTBUG-39622/QTBUG-55896
            const QString cName = fileInfo.canonicalFilePath().normalized(QString::NormalizationForm_C);
            if (file != cName && !cName.endsWith(relFile, Qt::CaseSensitive)) {
                qCWarning(lcPropagator) << "Detected case clash between" << file << "and" << cName;
                return true;
            }
        }
#elif defined(Q_OS_WIN)
        WIN32_FIND_DATA FindFileData;
        HANDLE hFind = nullptr;

        hFind = FindFirstFileW(reinterpret_cast<const wchar_t *>(FileSystem::longWinPath(file).utf16()), &FindFileData);
        if (hFind == INVALID_HANDLE_VALUE) {
            // returns false.
        } else {
            const QString realFileName = QString::fromWCharArray(FindFileData.cFileName);
            FindClose(hFind);

            if (!file.endsWith(realFileName, Qt::CaseSensitive)) {
                qCWarning(lcPropagator) << "Detected case clash between" << file << "and" << realFileName;
                return true;
            }
        }
#else
        // On Linux, the file system is case sensitive, but this code is useful for testing.
        // Just check that there is no other file with the same name and different casing.
        QFileInfo fileInfo(file);
        const QString fn = fileInfo.fileName();
        const QStringList list = fileInfo.dir().entryList({ fn });
        if (list.count() > 1 || (list.count() == 1 && list[0] != fn)) {
            qCWarning(lcPropagator) << "Detected case clash between" << file << "and" << list.constFirst();
            return true;
        }
#endif
    }
    return false;
}

bool OwncloudPropagator::hasCaseClashAccessibilityProblem(const QString &relfile)
{
#ifdef Q_OS_WIN
    bool result = false;
    const QString file(_localDir + relfile);
    WIN32_FIND_DATA FindFileData;
    HANDLE hFind = nullptr;

    hFind = FindFirstFileW(reinterpret_cast<const wchar_t *>(FileSystem::longWinPath(file).utf16()), &FindFileData);
    if (hFind != INVALID_HANDLE_VALUE) {
        QString firstFile = QString::fromWCharArray(FindFileData.cFileName);
        if (FindNextFile(hFind, &FindFileData)) {
            QString secondFile = QString::fromWCharArray(FindFileData.cFileName);
            // This extra check shouldn't be necessary, but ensures that there
            // are two different filenames that are identical when case is ignored.
            if (firstFile != secondFile
                && QString::compare(firstFile, secondFile, Qt::CaseInsensitive) == 0) {
                result = true;
                qCWarning(lcPropagator) << "Found two filepaths that only differ in case: " << firstFile << secondFile;
            }
        }
        FindClose(hFind);
    }
    return result;
#else
    Q_UNUSED(relfile);
    return false;
#endif
}

QString OwncloudPropagator::fullLocalPath(const QString &tmp_file_name) const
{
    return _localDir + tmp_file_name;
}

QString OwncloudPropagator::localPath() const
{
    return _localDir;
}

void OwncloudPropagator::scheduleNextJob()
{
    if (_jobScheduled) return; // don't schedule more than 1
    _jobScheduled = true;
    QTimer::singleShot(3, this, &OwncloudPropagator::scheduleNextJobImpl);
}

void OwncloudPropagator::scheduleNextJobImpl()
{
    // TODO: If we see that the automatic up-scaling has a bad impact we
    // need to check how to avoid this.
    // Down-scaling on slow networks? https://github.com/owncloud/client/issues/3382
    // Making sure we do up/down at same time? https://github.com/owncloud/client/issues/1633

    _jobScheduled = false;

    if (_activeJobList.count() < maximumActiveTransferJob()) {
        if (_rootJob->scheduleSelfOrChild()) {
            scheduleNextJob();
        }
    } else if (_activeJobList.count() < hardMaximumActiveJob()) {
        int likelyFinishedQuicklyCount = 0;
        // NOTE: Only counts the first 3 jobs! Then for each
        // one that is likely finished quickly, we can launch another one.
        // When a job finishes another one will "move up" to be one of the first 3 and then
        // be counted too.
        for (int i = 0; i < maximumActiveTransferJob() && i < _activeJobList.count(); i++) {
            if (_activeJobList.at(i)->isLikelyFinishedQuickly()) {
                likelyFinishedQuicklyCount++;
            }
        }
        if (_activeJobList.count() < maximumActiveTransferJob() + likelyFinishedQuicklyCount) {
            qCDebug(lcPropagator) << "Can pump in another request! activeJobs =" << _activeJobList.count();
            if (_rootJob->scheduleSelfOrChild()) {
                scheduleNextJob();
            }
        }
    }
}

void OwncloudPropagator::reportProgress(const SyncFileItem &item, qint64 bytes)
{
    emit progress(item, bytes);
}

AccountPtr OwncloudPropagator::account() const
{
    return _account;
}

OwncloudPropagator::DiskSpaceResult OwncloudPropagator::diskSpaceCheck() const
{
    const qint64 freeBytes = Utility::freeDiskSpace(_localDir);
    if (freeBytes < 0) {
        return DiskSpaceOk;
    }

    if (freeBytes < criticalFreeSpaceLimit()) {
        return DiskSpaceCritical;
    }

    if (freeBytes - _rootJob->committedDiskSpace() < freeSpaceLimit()) {
        return DiskSpaceFailure;
    }

    return DiskSpaceOk;
}

bool OwncloudPropagator::createConflict(const SyncFileItemPtr &item,
    PropagatorCompositeJob *composite, QString *error)
{
    QString fn = fullLocalPath(item->_file);

    QString renameError;
    auto conflictModTime = FileSystem::getModTime(fn);
    if (conflictModTime <= 0) {
        *error = tr("Impossible to get modification time for file in conflict %1").arg(fn);
        return false;
    }
    QString conflictUserName;
    if (account()->capabilities().uploadConflictFiles())
        conflictUserName = account()->davDisplayName();
    QString conflictFileName = Utility::makeConflictFileName(
        item->_file, Utility::qDateTimeFromTime_t(conflictModTime), conflictUserName);
    QString conflictFilePath = fullLocalPath(conflictFileName);

    emit touchedFile(fn);
    emit touchedFile(conflictFilePath);

    if (!FileSystem::rename(fn, conflictFilePath, &renameError)) {
        // If the rename fails, don't replace it.

        // If the file is locked, we want to retry this sync when it
        // becomes available again.
        if (FileSystem::isFileLocked(fn)) {
            emit seenLockedFile(fn);
        }

        if (error)
            *error = renameError;
        return false;
    }
    qCInfo(lcPropagator) << "Created conflict file" << fn << "->" << conflictFileName;

    // Create a new conflict record. To get the base etag, we need to read it from the db.
    ConflictRecord conflictRecord;
    conflictRecord.path = conflictFileName.toUtf8();
    conflictRecord.baseModtime = item->_previousModtime;
    conflictRecord.initialBasePath = item->_file.toUtf8();

    SyncJournalFileRecord baseRecord;
    if (_journal->getFileRecord(item->_originalFile, &baseRecord) && baseRecord.isValid()) {
        conflictRecord.baseEtag = baseRecord._etag;
        conflictRecord.baseFileId = baseRecord._fileId;
    } else {
        // We might very well end up with no fileid/etag for new/new conflicts
    }

    _journal->setConflictRecord(conflictRecord);

    // Create a new upload job if the new conflict file should be uploaded
    if (account()->capabilities().uploadConflictFiles()) {
        if (composite && !QFileInfo(conflictFilePath).isDir()) {
            SyncFileItemPtr conflictItem = SyncFileItemPtr(new SyncFileItem);
            conflictItem->_file = conflictFileName;
            conflictItem->_type = ItemTypeFile;
            conflictItem->_direction = SyncFileItem::Up;
            conflictItem->_instruction = CSYNC_INSTRUCTION_NEW;
            conflictItem->_modtime = conflictModTime;
            conflictItem->_size = item->_previousSize;
            emit newItem(conflictItem);
            composite->appendTask(conflictItem);
        }
    }

    // Need a new sync to detect the created copy of the conflicting file
    _anotherSyncNeeded = true;

    return true;
}

OCC::Optional<QString> OwncloudPropagator::createCaseClashConflict(const SyncFileItemPtr &item, const QString &temporaryDownloadedFile)
{
    auto filename = QString{};

    if (item->_type == ItemType::ItemTypeFile) {
        filename = fullLocalPath(item->_file);
    } else if (item->_type == ItemType::ItemTypeVirtualFileDownload) {
        filename = fullLocalPath(item->_file + syncOptions()._vfs->fileSuffix());
    }

    const auto conflictModTime = FileSystem::getModTime(filename);
    if (conflictModTime <= 0) {
        return tr("Impossible to get modification time for file in conflict %1").arg(filename);
    }

    const auto conflictFileName = Utility::makeCaseClashConflictFileName(item->_file, Utility::qDateTimeFromTime_t(conflictModTime));
    const auto conflictFilePath = fullLocalPath(conflictFileName);

    emit touchedFile(filename);
    emit touchedFile(conflictFilePath);

    qCInfo(lcPropagator) << "rename from" << temporaryDownloadedFile << "to" << conflictFilePath;
    if (QString renameError; !FileSystem::rename(temporaryDownloadedFile, conflictFilePath, &renameError)) {
        // If the rename fails, don't replace it.

        // If the file is locked, we want to retry this sync when it
        // becomes available again.
        if (FileSystem::isFileLocked(filename)) {
            emit seenLockedFile(filename);
        }

        return renameError;
    }
    FileSystem::setFileHidden(conflictFilePath, false);
    qCInfo(lcPropagator) << "Created case clash conflict file" << filename << "->" << conflictFilePath;

    // Create a new conflict record. To get the base etag, we need to read it from the db.
    auto conflictBasePath = item->_file.toUtf8();
    if (!item->_renameTarget.isEmpty()) {
        conflictBasePath = item->_renameTarget.toUtf8();
    }
    auto conflictRecord = ConflictRecord{conflictFileName.toUtf8(), {}, item->_previousModtime, {}, conflictBasePath};

    SyncJournalFileRecord baseRecord;
    if (_journal->getFileRecord(item->_originalFile, &baseRecord) && baseRecord.isValid()) {
        conflictRecord.baseEtag = baseRecord._etag;
        conflictRecord.baseFileId = baseRecord._fileId;
    }

    _journal->setCaseConflictRecord(conflictRecord);

    // Need a new sync to detect the created copy of the conflicting file
    _anotherSyncNeeded = true;

    return {};
}

QString OwncloudPropagator::adjustRenamedPath(const QString &original) const
{
    return OCC::adjustRenamedPath(_renamedDirectories, original);
}

Result<Vfs::ConvertToPlaceholderResult, QString> OwncloudPropagator::updateMetadata(const SyncFileItem &item)
{
    return OwncloudPropagator::staticUpdateMetadata(item, _localDir, syncOptions()._vfs.data(), _journal);
}

Result<Vfs::ConvertToPlaceholderResult, QString> OwncloudPropagator::staticUpdateMetadata(const SyncFileItem &item, const QString localDir,
                                                                                          Vfs *vfs, SyncJournalDb *const journal)
{
    const QString fsPath = localDir + item.destination();
    const auto result = vfs->convertToPlaceholder(fsPath, item);
    if (!result) {
        return result.error();
    } else if (*result == Vfs::ConvertToPlaceholderResult::Locked) {
        return Vfs::ConvertToPlaceholderResult::Locked;
    }
    auto record = item.toSyncJournalFileRecordWithInode(fsPath);
    const auto dBresult = journal->setFileRecord(record);
    if (!dBresult) {
        return dBresult.error();
    }
    return Vfs::ConvertToPlaceholderResult::Ok;
}

bool OwncloudPropagator::isDelayedUploadItem(const SyncFileItemPtr &item) const
{
    const auto checkFileShouldBeEncrypted = [this] (const SyncFileItemPtr &item) -> bool {
        const auto path = item->_file;
        const auto slashPosition = path.lastIndexOf('/');
        const auto parentPath = slashPosition >= 0 ? path.left(slashPosition) : QString();

        SyncJournalFileRecord parentRec;
        bool ok = _journal->getFileRecord(parentPath, &parentRec);
        if (!ok) {
            return false;
        }

        const auto accountPtr = account();

        if (!accountPtr->capabilities().clientSideEncryptionAvailable() ||
            !parentRec.isValid() ||
            !parentRec.isE2eEncrypted()) {
            return false;
        }

        return true;
    };

    return account()->capabilities().bulkUpload() && !_scheduleDelayedTasks && !item->isEncrypted() && _syncOptions.minChunkSize() > item->_size
        && !isInBulkUploadBlackList(item->_file) && !checkFileShouldBeEncrypted(item);
}

void OwncloudPropagator::setScheduleDelayedTasks(bool active)
{
    _scheduleDelayedTasks = active;
}

void OwncloudPropagator::clearDelayedTasks()
{
    _delayedTasks.clear();
}

void OwncloudPropagator::addToBulkUploadBlackList(const QString &file)
{
    qCDebug(lcPropagator) << "black list for bulk upload" << file;
    _bulkUploadBlackList.insert(file);
}

void OwncloudPropagator::removeFromBulkUploadBlackList(const QString &file)
{
    qCDebug(lcPropagator) << "black list for bulk upload" << file;
    _bulkUploadBlackList.remove(file);
}

bool OwncloudPropagator::isInBulkUploadBlackList(const QString &file) const
{
    return _bulkUploadBlackList.contains(file);
}

// ================================================================================

PropagatorJob::PropagatorJob(OwncloudPropagator *propagator)
    : QObject(propagator)
{
}

OwncloudPropagator *PropagatorJob::propagator() const
{
    return qobject_cast<OwncloudPropagator *>(parent());
}

ErrorCategory PropagatorJob::errorCategoryFromNetworkError(const QNetworkReply::NetworkError error)
{
    auto result = ErrorCategory::NoError;
    switch (error)
    {
    case QNetworkReply::NoError:
        result = ErrorCategory::NoError;
        break;
    case QNetworkReply::TemporaryNetworkFailureError:
    case QNetworkReply::RemoteHostClosedError:
        result = ErrorCategory::NetworkError;
        break;
    case QNetworkReply::ConnectionRefusedError:
    case QNetworkReply::HostNotFoundError:
    case QNetworkReply::TimeoutError:
    case QNetworkReply::OperationCanceledError:
    case QNetworkReply::SslHandshakeFailedError:
    case QNetworkReply::NetworkSessionFailedError:
    case QNetworkReply::BackgroundRequestNotAllowedError:
    case QNetworkReply::TooManyRedirectsError:
    case QNetworkReply::InsecureRedirectError:
    case QNetworkReply::UnknownNetworkError:
    case QNetworkReply::ProxyConnectionRefusedError:
    case QNetworkReply::ProxyConnectionClosedError:
    case QNetworkReply::ProxyNotFoundError:
    case QNetworkReply::ProxyTimeoutError:
    case QNetworkReply::ProxyAuthenticationRequiredError:
    case QNetworkReply::UnknownProxyError:
    case QNetworkReply::ContentAccessDenied:
    case QNetworkReply::ContentOperationNotPermittedError:
    case QNetworkReply::ContentNotFoundError:
    case QNetworkReply::AuthenticationRequiredError:
    case QNetworkReply::ContentReSendError:
    case QNetworkReply::ContentConflictError:
    case QNetworkReply::ContentGoneError:
    case QNetworkReply::UnknownContentError:
    case QNetworkReply::ProtocolUnknownError:
    case QNetworkReply::ProtocolInvalidOperationError:
    case QNetworkReply::ProtocolFailure:
    case QNetworkReply::InternalServerError:
    case QNetworkReply::OperationNotImplementedError:
    case QNetworkReply::ServiceUnavailableError:
    case QNetworkReply::UnknownServerError:
        result = ErrorCategory::GenericError;
        break;
    }

    return result;
}

// ================================================================================

PropagatorJob::JobParallelism PropagatorCompositeJob::parallelism() const
{
    // If any of the running sub jobs is not parallel, we have to wait
    for (int i = 0; i < _runningJobs.count(); ++i) {
        if (_runningJobs.at(i)->parallelism() != FullParallelism) {
            return _runningJobs.at(i)->parallelism();
        }
    }
    return FullParallelism;
}

void PropagatorCompositeJob::slotSubJobAbortFinished()
{
    // Count that job has been finished
    _abortsCount--;

    // Emit abort if last job has been aborted
    if (_abortsCount == 0) {
        emit abortFinished();
    }
}

void PropagatorCompositeJob::appendJob(PropagatorJob *job)
{
    job->setAssociatedComposite(this);
    _jobsToDo.append(job);
}

bool PropagatorCompositeJob::scheduleSelfOrChild()
{
    if (_state == Finished) {
        return false;
    }

    // Start the composite job
    if (_state == NotYetStarted) {
        _state = Running;
    }

    // Ask all the running composite jobs if they have something new to schedule.
    for (auto runningJob : qAsConst(_runningJobs)) {
        ASSERT(runningJob->_state == Running);

        if (possiblyRunNextJob(runningJob)) {
            return true;
        }

        // If any of the running sub jobs is not parallel, we have to cancel the scheduling
        // of the rest of the list and wait for the blocking job to finish and schedule the next one.
        auto paral = runningJob->parallelism();
        if (paral == WaitForFinished) {
            return false;
        }
    }

    // Now it's our turn, check if we have something left to do.
    // First, convert a task to a job if necessary
    while (_jobsToDo.isEmpty() && !_tasksToDo.isEmpty()) {
        SyncFileItemPtr nextTask = _tasksToDo.first();
        _tasksToDo.remove(0);
        PropagatorJob *job = propagator()->createJob(nextTask);
        if (!job) {
            qCWarning(lcDirectory) << "Useless task found for file" << nextTask->destination() << "instruction" << nextTask->_instruction;
            continue;
        }
        appendJob(job);
        break;
    }
    // Then run the next job
    if (!_jobsToDo.isEmpty()) {
        PropagatorJob *nextJob = _jobsToDo.first();
        _jobsToDo.remove(0);
        _runningJobs.append(nextJob);
        return possiblyRunNextJob(nextJob);
    }

    // If neither us or our children had stuff left to do we could hang. Make sure
    // we mark this job as finished so that the propagator can schedule a new one.
    if (_jobsToDo.isEmpty() && _tasksToDo.isEmpty() && _runningJobs.isEmpty()) {
        // Our parent jobs are already iterating over their running jobs, post to the event loop
        // to avoid removing ourself from that list while they iterate.
        QMetaObject::invokeMethod(this, "finalize", Qt::QueuedConnection);
    }
    return false;
}

void PropagatorCompositeJob::slotSubJobFinished(SyncFileItem::Status status)
{
    auto *subJob = dynamic_cast<PropagatorJob *>(sender());
    ASSERT(subJob);

    // Delete the job and remove it from our list of jobs.
    subJob->deleteLater();
    int i = _runningJobs.indexOf(subJob);
    ENFORCE(i >= 0); // should only happen if this function is called more than once
    _runningJobs.remove(i);

    // Any sub job error will cause the whole composite to fail. This is important
    // for knowing whether to update the etag in PropagateDirectory, for example.
    if (status == SyncFileItem::FatalError
        || status == SyncFileItem::NormalError
        || status == SyncFileItem::SoftError
        || status == SyncFileItem::DetailError
        || status == SyncFileItem::BlacklistedError) {
        _hasError = status;
    }

    if (_jobsToDo.isEmpty() && _tasksToDo.isEmpty() && _runningJobs.isEmpty()) {
        finalize();
    } else {
        propagator()->scheduleNextJob();
    }
}

void PropagatorCompositeJob::finalize()
{
    // The propagator will do parallel scheduling and this could be posted
    // multiple times on the event loop, ignore the duplicate calls.
    if (_state == Finished)
        return;

    _state = Finished;
    emit finished(_hasError == SyncFileItem::NoStatus ? SyncFileItem::Success : _hasError);
}

qint64 PropagatorCompositeJob::committedDiskSpace() const
{
    qint64 needed = 0;
    foreach (PropagatorJob *job, _runningJobs) {
        needed += job->committedDiskSpace();
    }
    return needed;
}

// ================================================================================

PropagateDirectory::PropagateDirectory(OwncloudPropagator *propagator, const SyncFileItemPtr &item)
    : PropagatorJob(propagator)
    , _item(item)
    , _firstJob(propagator->createJob(item))
    , _subJobs(propagator)
{
    if (_firstJob) {
        connect(_firstJob.data(), &PropagatorJob::finished, this, &PropagateDirectory::slotFirstJobFinished);
        _firstJob->setAssociatedComposite(&_subJobs);
    }
    connect(&_subJobs, &PropagatorJob::finished, this, &PropagateDirectory::slotSubJobsFinished);
}

PropagatorJob::JobParallelism PropagateDirectory::parallelism() const
{
    // If any of the non-finished sub jobs is not parallel, we have to wait
    if (_firstJob && _firstJob->parallelism() != FullParallelism) {
        return WaitForFinished;
    }
    if (_subJobs.parallelism() != FullParallelism) {
        return WaitForFinished;
    }
    return FullParallelism;
}


bool PropagateDirectory::scheduleSelfOrChild()
{
    if (_state == Finished) {
        return false;
    }

    if (_state == NotYetStarted) {
        _state = Running;
    }

    if (_firstJob && _firstJob->_state == NotYetStarted) {
        return _firstJob->scheduleSelfOrChild();
    }

    if (_firstJob && _firstJob->_state == Running) {
        // Don't schedule any more job until this is done.
        return false;
    }

    return _subJobs.scheduleSelfOrChild();
}

void PropagateDirectory::slotFirstJobFinished(SyncFileItem::Status status)
{
    _firstJob.take()->deleteLater();

    if (status != SyncFileItem::Success
        && status != SyncFileItem::Restoration
        && status != SyncFileItem::Conflict) {
        if (_state != Finished) {
            // Synchronously abort
            abort(AbortType::Synchronous);
            _state = Finished;
            qCInfo(lcPropagator) << "PropagateDirectory::slotFirstJobFinished" << "emit finished" << status;
            emit finished(status);
        }
        return;
    }

    propagator()->scheduleNextJob();
}

void PropagateDirectory::slotSubJobsFinished(SyncFileItem::Status status)
{
    if (!_item->isEmpty() && status == SyncFileItem::Success) {
        // If a directory is renamed, recursively delete any stale items
        // that may still exist below the old path.
        if (_item->_instruction == CSYNC_INSTRUCTION_RENAME && _item->_originalFile != _item->_renameTarget) {
            if (!propagator()->_journal->deleteFileRecord(_item->_originalFile, true)) {
                qCWarning(lcDirectory) << "could not delete file from local DB" << _item->_originalFile;
                _state = Finished;
                status = _item->_status = SyncFileItem::FatalError;
                _item->_errorString = tr("could not delete file %1 from local DB").arg(_item->_originalFile);
                qCInfo(lcPropagator) << "PropagateDirectory::slotSubJobsFinished"
                                     << "emit finished" << status;
                emit finished(status);
                return;
            }
        }

        if (_item->_instruction == CSYNC_INSTRUCTION_NEW && _item->_direction == SyncFileItem::Down) {
            // special case for local MKDIR, set local directory mtime
            // (it's not synced later at all, but can be nice to have it set initially)

            if (_item->_modtime <= 0) {
                status = _item->_status = SyncFileItem::NormalError;
                _item->_errorString = tr("Error updating metadata due to invalid modification time");
                qCWarning(lcDirectory) << "Error writing to the database for file" << _item->_file;
            }

            FileSystem::setModTime(propagator()->fullLocalPath(_item->destination()), _item->_modtime);
        }

        // For new directories we always want to update the etag once
        // the directory has been propagated. Otherwise the directory
        // could appear locally without being added to the database.
        if (_item->_instruction == CSYNC_INSTRUCTION_RENAME
            || _item->_instruction == CSYNC_INSTRUCTION_NEW
            || _item->_instruction == CSYNC_INSTRUCTION_UPDATE_METADATA) {
            const auto result = propagator()->updateMetadata(*_item);
            if (!result) {
                status = _item->_status = SyncFileItem::FatalError;
                _item->_errorString = tr("Error updating metadata: %1").arg(result.error());
                qCWarning(lcDirectory) << "Error writing to the database for file" << _item->_file << "with" << result.error();
            } else if (*result == Vfs::ConvertToPlaceholderResult::Locked) {
                _item->_status = SyncFileItem::SoftError;
                _item->_errorString = tr("File is currently in use");
            }
        }
    }
    _state = Finished;
    qCInfo(lcPropagator) << "PropagateDirectory::slotSubJobsFinished" << "emit finished" << status;
    emit finished(status);
}

PropagateRootDirectory::PropagateRootDirectory(OwncloudPropagator *propagator)
    : PropagateDirectory(propagator, SyncFileItemPtr(new SyncFileItem))
    , _dirDeletionJobs(propagator)
{
    connect(&_dirDeletionJobs, &PropagatorJob::finished, this, &PropagateRootDirectory::slotDirDeletionJobsFinished);
}

PropagatorJob::JobParallelism PropagateRootDirectory::parallelism() const
{
    // the root directory parallelism isn't important
    return WaitForFinished;
}

void PropagateRootDirectory::abort(PropagatorJob::AbortType abortType)
{
    if (_firstJob)
        // Force first job to abort synchronously
        // even if caller allows async abort (asyncAbort)
        _firstJob->abort(AbortType::Synchronous);

    if (abortType == AbortType::Asynchronous) {
        struct AbortsFinished {
            bool subJobsFinished = false;
            bool dirDeletionFinished = false;
        };
        auto abortStatus = QSharedPointer<AbortsFinished>(new AbortsFinished);

        connect(&_subJobs, &PropagatorCompositeJob::abortFinished, this, [this, abortStatus]() {
            abortStatus->subJobsFinished = true;
            if (abortStatus->subJobsFinished && abortStatus->dirDeletionFinished)
                emit abortFinished();
        });
        connect(&_dirDeletionJobs, &PropagatorCompositeJob::abortFinished, this, [this, abortStatus]() {
            abortStatus->dirDeletionFinished = true;
            if (abortStatus->subJobsFinished && abortStatus->dirDeletionFinished)
                emit abortFinished();
        });
    }
    _subJobs.abort(abortType);
    _dirDeletionJobs.abort(abortType);
}

qint64 PropagateRootDirectory::committedDiskSpace() const
{
    return _subJobs.committedDiskSpace() + _dirDeletionJobs.committedDiskSpace();
}

void PropagateRootDirectory::appendDirDeletionJob(PropagatorJob *job)
{
    _dirDeletionJobs.appendJob(job);
}

bool PropagateRootDirectory::scheduleSelfOrChild()
{
    qCInfo(lcRootDirectory()) << "scheduleSelfOrChild" << _state << "pending uploads" << propagator()->delayedTasks().size() << "subjobs state" << _subJobs._state;

    if (_state == Finished) {
        return false;
    }

    if (PropagateDirectory::scheduleSelfOrChild() && propagator()->delayedTasks().empty()) {
        return true;
    }

    // Important: Finish _subJobs before scheduling any deletes.
    if (_subJobs._state != Finished) {
        return false;
    }

    if (!propagator()->delayedTasks().empty()) {
        return scheduleDelayedJobs();
    }

    return _dirDeletionJobs.scheduleSelfOrChild();
}

void PropagateRootDirectory::slotSubJobsFinished(SyncFileItem::Status status)
{
    qCInfo(lcRootDirectory()) << status << "slotSubJobsFinished" << _state << "pending uploads" << propagator()->delayedTasks().size() << "subjobs state" << _subJobs._state;

    if (!propagator()->delayedTasks().empty()) {
        scheduleDelayedJobs();
        return;
    }

    if (status != SyncFileItem::Success
        && status != SyncFileItem::Restoration
        && status != SyncFileItem::BlacklistedError
        && status != SyncFileItem::FileNameClash
        && status != SyncFileItem::Conflict) {
        if (_state != Finished) {
            // Synchronously abort
            abort(AbortType::Synchronous);
            _state = Finished;
            qCInfo(lcPropagator) << "PropagateRootDirectory::slotSubJobsFinished" << "emit finished" << status;
            emit finished(status);
        }
        return;
    }

    if (_errorStatus == SyncFileItem::NoStatus) {
        switch (status) {
        case SyncFileItem::NoStatus:
        case SyncFileItem::FatalError:
        case SyncFileItem::NormalError:
        case SyncFileItem::SoftError:
        case SyncFileItem::Conflict:
        case SyncFileItem::FileIgnored:
        case SyncFileItem::FileLocked:
        case SyncFileItem::Restoration:
        case SyncFileItem::FileNameInvalid:
        case SyncFileItem::FileNameInvalidOnServer:
        case SyncFileItem::DetailError:
        case SyncFileItem::Success:
            break;
        case SyncFileItem::FileNameClash:
        case SyncFileItem::BlacklistedError:
            _errorStatus = status;
            break;
        }
    }

    propagator()->scheduleNextJob();
}

void PropagateRootDirectory::slotDirDeletionJobsFinished(SyncFileItem::Status status)
{
    if (_errorStatus != SyncFileItem::NoStatus && status == SyncFileItem::Success) {
        qCInfo(lcPropagator) << "PropagateRootDirectory::slotDirDeletionJobsFinished" << "reporting previous error" << _errorStatus;
        status = _errorStatus;
    }

    _state = Finished;
    qCInfo(lcPropagator) << "PropagateRootDirectory::slotDirDeletionJobsFinished" << "emit finished" << status;
    emit finished(status);
}

bool PropagateRootDirectory::scheduleDelayedJobs()
{
    qCInfo(lcPropagator) << "PropagateRootDirectory::scheduleDelayedJobs";
    propagator()->setScheduleDelayedTasks(true);
    auto bulkPropagatorJob = std::make_unique<BulkPropagatorJob>(propagator(), propagator()->delayedTasks());
    propagator()->clearDelayedTasks();
    _subJobs.appendJob(bulkPropagatorJob.release());
    _subJobs._state = Running;
    return _subJobs.scheduleSelfOrChild();
}

// ================================================================================

CleanupPollsJob::~CleanupPollsJob() = default;

void CleanupPollsJob::start()
{
    if (_pollInfos.empty()) {
        emit finished();
        deleteLater();
        return;
    }

    auto info = _pollInfos.first();
    _pollInfos.pop_front();
    SyncFileItemPtr item(new SyncFileItem);
    item->_file = info._file;
    item->_modtime = info._modtime;
    item->_size = info._fileSize;
    auto *job = new PollJob(_account, info._url, item, _journal, _localPath, this);
    connect(job, &PollJob::finishedSignal, this, &CleanupPollsJob::slotPollFinished);
    job->start();
}

void CleanupPollsJob::slotPollFinished()
{
    auto *job = qobject_cast<PollJob *>(sender());
    ASSERT(job);
    if (job->_item->_status == SyncFileItem::FatalError) {
        emit aborted(job->_item->_errorString, ErrorCategory::GenericError);
        deleteLater();
        return;
    } else if (job->_item->_status != SyncFileItem::Success) {
        qCWarning(lcCleanupPolls) << "There was an error with file " << job->_item->_file << job->_item->_errorString;
    } else {
        if (!OwncloudPropagator::staticUpdateMetadata(*job->_item, _localPath, _vfs.data(), _journal)) {
            qCWarning(lcCleanupPolls) << "database error";
            job->_item->_status = SyncFileItem::FatalError;
            job->_item->_errorString = tr("Error writing metadata to the database");
            emit aborted(job->_item->_errorString, ErrorCategory::GenericError);
            deleteLater();
            return;
        }
        _journal->setUploadInfo(job->_item->_file, SyncJournalDb::UploadInfo());
    }
    // Continue with the next entry, or finish
    start();
}

QString OwncloudPropagator::fullRemotePath(const QString &tmp_file_name) const
{
    // TODO: should this be part of the _item (SyncFileItemPtr)?
    return _remoteFolder + tmp_file_name;
}

QString OwncloudPropagator::remotePath() const
{
    return _remoteFolder;
}

void PropagateIgnoreJob::start()
{
    auto status = _item->_status;
    if (status == SyncFileItem::NoStatus) {
        if (_item->_instruction == CSYNC_INSTRUCTION_ERROR) {
            status = SyncFileItem::NormalError;
        } else {
            status = SyncFileItem::FileIgnored;
            ASSERT(_item->_instruction == CSYNC_INSTRUCTION_IGNORE);
        }
    } else if (status == SyncFileItem::FileNameClash) {
        const auto conflictRecord = propagator()->_journal->caseConflictRecordByPath(_item->_file);
        if (conflictRecord.isValid()) {
            _item->_file = conflictRecord.initialBasePath;
        }
    }
    done(status, _item->_errorString, ErrorCategory::NoError);
}

}
