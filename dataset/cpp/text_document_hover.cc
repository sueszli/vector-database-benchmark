#include "message_handler.h"
#include "query_utils.h"
#include "queue_manager.h"

namespace {
MethodType kMethodType = "textDocument/hover";

// Find the comments for |sym|, if any.
optional<lsMarkedString> GetComments(QueryDatabase* db,
                                     QueryId::SymbolRef sym) {
  auto make = [](std::string_view comment) -> optional<lsMarkedString> {
    lsMarkedString result;
    result.value = std::string(comment.data(), comment.length());
    return result;
  };

  optional<lsMarkedString> result;
  WithEntity(db, sym, [&](const auto& entity) {
    if (const auto* def = entity.AnyDef()) {
      if (!def->comments.empty())
        result = make(def->comments);
    }
  });
  return result;
}

// Returns the hover or detailed name for `sym`, if any.
optional<lsMarkedString> GetHoverOrName(QueryDatabase* db,
                                        const std::string& language,
                                        QueryId::SymbolRef sym) {
  auto make = [&](std::string_view comment) {
    lsMarkedString result;
    result.language = language;
    result.value = std::string(comment.data(), comment.length());
    return result;
  };

  optional<lsMarkedString> result;
  WithEntity(db, sym, [&](const auto& entity) {
    if (const auto* def = entity.AnyDef()) {
      if (!def->hover.empty())
        result = make(def->hover);
      else if (!def->detailed_name.empty())
        result = make(def->detailed_name);
    }
  });
  return result;
}

struct In_TextDocumentHover : public RequestInMessage {
  MethodType GetMethodType() const override { return kMethodType; }
  lsTextDocumentPositionParams params;
};
MAKE_REFLECT_STRUCT(In_TextDocumentHover, id, params);
REGISTER_IN_MESSAGE(In_TextDocumentHover);

struct Out_TextDocumentHover : public lsOutMessage<Out_TextDocumentHover> {
  struct Result {
    std::vector<lsMarkedString> contents;
    optional<lsRange> range;
  };

  lsRequestId id;
  optional<Result> result;
};
MAKE_REFLECT_STRUCT(Out_TextDocumentHover::Result, contents, range);
MAKE_REFLECT_STRUCT_OPTIONALS_MANDATORY(Out_TextDocumentHover,
                                        jsonrpc,
                                        id,
                                        result);

struct Handler_TextDocumentHover : BaseMessageHandler<In_TextDocumentHover> {
  MethodType GetMethodType() const override { return kMethodType; }
  void Run(In_TextDocumentHover* request) override {
    QueryFile* file;
    if (!FindFileOrFail(db, project, request->id,
                        request->params.textDocument.uri.GetAbsolutePath(),
                        &file)) {
      return;
    }

    WorkingFile* working_file =
        working_files->GetFileByFilename(file->def->path);

    Out_TextDocumentHover out;
    out.id = request->id;

    for (QueryId::SymbolRef sym :
         FindSymbolsAtLocation(working_file, file, request->params.position)) {
      // Found symbol. Return hover.
      optional<lsRange> ls_range = GetLsRange(
          working_files->GetFileByFilename(file->def->path), sym.range);
      if (!ls_range)
        continue;

      optional<lsMarkedString> comments = GetComments(db, sym);
      optional<lsMarkedString> hover =
          GetHoverOrName(db, file->def->language, sym);
      if (comments || hover) {
        out.result = Out_TextDocumentHover::Result();
        out.result->range = *ls_range;
        if (comments)
          out.result->contents.push_back(*comments);
        if (hover)
          out.result->contents.push_back(*hover);
        break;
      }
    }

    QueueManager::WriteStdout(kMethodType, out);
  }
};
REGISTER_MESSAGE_HANDLER(Handler_TextDocumentHover);
}  // namespace
