#include "cache_manager.h"
#include "clang_complete.h"
#include "message_handler.h"
#include "project.h"
#include "queue_manager.h"
#include "working_files.h"

#include <loguru/loguru.hpp>

namespace {
MethodType kMethodType = "workspace/didChangeWatchedFiles";

enum class lsFileChangeType {
  Created = 1,
  Changed = 2,
  Deleted = 3,
};
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-function"
MAKE_REFLECT_TYPE_PROXY(lsFileChangeType);
#pragma clang diagnostic pop

struct lsFileEvent {
  lsDocumentUri uri;
  lsFileChangeType type;
};
MAKE_REFLECT_STRUCT(lsFileEvent, uri, type);

struct lsDidChangeWatchedFilesParams {
  std::vector<lsFileEvent> changes;
};
MAKE_REFLECT_STRUCT(lsDidChangeWatchedFilesParams, changes);

struct In_WorkspaceDidChangeWatchedFiles : public NotificationInMessage {
  MethodType GetMethodType() const override { return kMethodType; }
  lsDidChangeWatchedFilesParams params;
};
MAKE_REFLECT_STRUCT(In_WorkspaceDidChangeWatchedFiles, params);
REGISTER_IN_MESSAGE(In_WorkspaceDidChangeWatchedFiles);

struct Handler_WorkspaceDidChangeWatchedFiles
    : BaseMessageHandler<In_WorkspaceDidChangeWatchedFiles> {
  MethodType GetMethodType() const override { return kMethodType; }
  void Run(In_WorkspaceDidChangeWatchedFiles* request) override {
    for (lsFileEvent& event : request->params.changes) {
      AbsolutePath path = event.uri.GetAbsolutePath();
      auto it = project->absolute_path_to_entry_index_.find(path);
      if (it == project->absolute_path_to_entry_index_.end())
        continue;
      const Project::Entry& entry = project->entries[it->second];
      bool is_interactive =
          working_files->GetFileByFilename(entry.filename) != nullptr;
      switch (event.type) {
        case lsFileChangeType::Created:
        case lsFileChangeType::Changed: {
          QueueManager::instance()->index_request.Enqueue(
              Index_Request(path, entry.args, is_interactive, nullopt,
                            ICacheManager::Make()),
              false /*priority*/);
          if (is_interactive)
            clang_complete->NotifySave(path);
          break;
        }
        case lsFileChangeType::Deleted:
          QueueManager::instance()->index_request.Enqueue(
              Index_Request(path, entry.args, is_interactive, std::string(),
                            ICacheManager::Make()),
              false /*priority*/);
          break;
      }
    }
  }
};
REGISTER_MESSAGE_HANDLER(Handler_WorkspaceDidChangeWatchedFiles);
}  // namespace
