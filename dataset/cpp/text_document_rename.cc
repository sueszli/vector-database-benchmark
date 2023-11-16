#include "message_handler.h"
#include "query_utils.h"
#include "queue_manager.h"

namespace {
MethodType kMethodType = "textDocument/rename";

lsWorkspaceEdit BuildWorkspaceEdit(QueryDatabase* db,
                                   WorkingFiles* working_files,
                                   QueryId::SymbolRef sym,
                                   const std::string& new_text) {
  std::unordered_map<QueryId::File, lsTextDocumentEdit> path_to_edit;

  EachOccurrence(db, sym, true, [&](QueryId::LexicalRef ref) {
    optional<lsLocation> ls_location = GetLsLocation(db, working_files, ref);
    if (!ls_location)
      return;

    QueryId::File file_id = ref.file;
    if (path_to_edit.find(file_id) == path_to_edit.end()) {
      path_to_edit[file_id] = lsTextDocumentEdit();

      QueryFile& file = db->files[file_id.id];
      if (!file.def)
        return;

      const std::string& path = file.def->path;
      path_to_edit[file_id].textDocument.uri = lsDocumentUri::FromPath(path);

      WorkingFile* working_file = working_files->GetFileByFilename(path);
      if (working_file)
        path_to_edit[file_id].textDocument.version = working_file->version;
    }

    lsTextEdit edit;
    edit.range = ls_location->range;
    edit.newText = new_text;

    // vscode complains if we submit overlapping text edits.
    auto& edits = path_to_edit[file_id].edits;
    if (std::find(edits.begin(), edits.end(), edit) == edits.end())
      edits.push_back(edit);
  });

  lsWorkspaceEdit edit;
  for (const auto& changes : path_to_edit)
    edit.documentChanges.push_back(changes.second);
  return edit;
}

struct In_TextDocumentRename : public RequestInMessage {
  MethodType GetMethodType() const override { return kMethodType; }
  struct Params {
    // The document to format.
    lsTextDocumentIdentifier textDocument;

    // The position at which this request was sent.
    lsPosition position;

    // The new name of the symbol. If the given name is not valid the
    // request must return a [ResponseError](#ResponseError) with an
    // appropriate message set.
    std::string newName;
  };
  Params params;
};
MAKE_REFLECT_STRUCT(In_TextDocumentRename::Params,
                    textDocument,
                    position,
                    newName);
MAKE_REFLECT_STRUCT(In_TextDocumentRename, id, params);
REGISTER_IN_MESSAGE(In_TextDocumentRename);

struct Out_TextDocumentRename : public lsOutMessage<Out_TextDocumentRename> {
  lsRequestId id;
  lsWorkspaceEdit result;
};
MAKE_REFLECT_STRUCT(Out_TextDocumentRename, jsonrpc, id, result);

struct Handler_TextDocumentRename : BaseMessageHandler<In_TextDocumentRename> {
  MethodType GetMethodType() const override { return kMethodType; }
  void Run(In_TextDocumentRename* request) override {
    QueryId::File file_id;
    QueryFile* file;
    if (!FindFileOrFail(db, project, request->id,
                        request->params.textDocument.uri.GetAbsolutePath(),
                        &file, &file_id)) {
      return;
    }

    WorkingFile* working_file =
        working_files->GetFileByFilename(file->def->path);

    Out_TextDocumentRename out;
    out.id = request->id;

    for (QueryId::SymbolRef sym :
         FindSymbolsAtLocation(working_file, file, request->params.position)) {
      // Found symbol. Return references to rename.
      out.result =
          BuildWorkspaceEdit(db, working_files, sym, request->params.newName);
      break;
    }

    QueueManager::WriteStdout(kMethodType, out);
  }
};
REGISTER_MESSAGE_HANDLER(Handler_TextDocumentRename);
}  // namespace
