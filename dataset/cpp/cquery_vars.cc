#include "message_handler.h"
#include "query_utils.h"
#include "queue_manager.h"

namespace {
MethodType kMethodType = "$cquery/vars";

struct In_CqueryVars : public RequestInMessage {
  MethodType GetMethodType() const override { return kMethodType; }

  lsTextDocumentPositionParams params;
};
MAKE_REFLECT_STRUCT(In_CqueryVars, id, params);
REGISTER_IN_MESSAGE(In_CqueryVars);

struct Handler_CqueryVars : BaseMessageHandler<In_CqueryVars> {
  MethodType GetMethodType() const override { return kMethodType; }

  void Run(In_CqueryVars* request) override {
    QueryFile* file;
    if (!FindFileOrFail(db, project, request->id,
                        request->params.textDocument.uri.GetAbsolutePath(),
                        &file)) {
      return;
    }

    WorkingFile* working_file =
        working_files->GetFileByFilename(file->def->path);

    Out_LocationList out;
    out.id = request->id;
    for (QueryId::SymbolRef sym :
         FindSymbolsAtLocation(working_file, file, request->params.position)) {
      switch (sym.kind) {
        default:
          break;
        case SymbolKind::Var: {
          const QueryVar::Def* def = db->GetVar(sym).AnyDef();
          if (!def || !def->type)
            continue;
        }
        // fallthrough
        case SymbolKind::Type: {
          QueryType& type = db->GetType(sym);
          out.result = GetLsLocations(db, working_files,
                                      GetDeclarations(db, type.instances));
          break;
        }
      }
    }
    QueueManager::WriteStdout(kMethodType, out);
  }
};
REGISTER_MESSAGE_HANDLER(Handler_CqueryVars);
}  // namespace
