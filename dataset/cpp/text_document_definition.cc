#include "lex_utils.h"
#include "message_handler.h"
#include "query_utils.h"
#include "queue_manager.h"

#include <ctype.h>
#include <limits.h>
#include <cstdlib>

namespace {
MethodType kMethodType = "textDocument/definition";

struct In_TextDocumentDefinition : public RequestInMessage {
  MethodType GetMethodType() const override { return kMethodType; }
  lsTextDocumentPositionParams params;
};
MAKE_REFLECT_STRUCT(In_TextDocumentDefinition, id, params);
REGISTER_IN_MESSAGE(In_TextDocumentDefinition);

std::vector<QueryId::LexicalRef> GetNonDefDeclarationTargets(
    QueryDatabase* db,
    QueryId::SymbolRef sym) {
  switch (sym.kind) {
    case SymbolKind::Var: {
      std::vector<QueryId::LexicalRef> ret = GetNonDefDeclarations(db, sym);
      // If there is no declaration, jump the its type.
      if (ret.empty()) {
        for (auto& def : db->GetVar(sym).def)
          if (def.type) {
            if (auto use = GetDefinitionSpell(
                    db, SymbolIdx{*def.type, SymbolKind::Type})) {
              ret.push_back(*use);
              break;
            }
          }
      }
      return ret;
    }
    default:
      return GetNonDefDeclarations(db, sym);
  }
}

struct Handler_TextDocumentDefinition
    : BaseMessageHandler<In_TextDocumentDefinition> {
  MethodType GetMethodType() const override { return kMethodType; }
  void Run(In_TextDocumentDefinition* request) override {
    QueryId::File file_id;
    QueryFile* file;
    if (!FindFileOrFail(db, project, request->id,
                        request->params.textDocument.uri.GetAbsolutePath(),
                        &file, &file_id)) {
      return;
    }

    WorkingFile* working_file =
        working_files->GetFileByFilename(file->def->path);

    Out_LocationList out;
    out.id = request->id;

    Maybe<QueryId::LexicalRef> on_def;
    bool has_symbol = false;
    int target_line = request->params.position.line;
    int target_column = request->params.position.character;

    for (QueryId::SymbolRef sym :
         FindSymbolsAtLocation(working_file, file, request->params.position)) {
      // Found symbol. Return definition.
      has_symbol = true;

      // Special cases which are handled:
      //  - symbol has declaration but no definition (ie, pure virtual)
      //  - start at spelling but end at extent for better mouse tooltip
      //  - goto declaration while in definition of recursive type
      std::vector<QueryId::LexicalRef> uses;
      EachEntityDef(db, sym, [&](const auto& def) {
        if (def.spell && def.extent) {
          QueryId::LexicalRef spell = *def.spell;
          // If on a definition, clear |uses| to find declarations below.
          if (spell.file == file_id &&
              spell.range.Contains(target_line, target_column)) {
            on_def = spell;
            uses.clear();
          }
          // We use spelling start and extent end because this causes vscode
          // to highlight the entire definition when previewing / hoving with
          // the mouse.
          spell.range.end = def.extent->range.end;
          uses.push_back(spell);
        }
        return true;
      });

      if (uses.empty()) {
        // The symbol has no definition or the cursor is on a definition.
        uses = GetNonDefDeclarationTargets(db, sym);
        // There is no declaration but the cursor is on a definition.
        if (uses.empty() && on_def)
          uses.push_back(*on_def);
      }
      AddRange(&out.result, GetLsLocations(db, working_files, uses));
    }

    // No symbols - check for includes.
    if (out.result.empty()) {
      for (const IndexInclude& include : file->def->includes) {
        if (include.line == target_line) {
          lsLocation result;
          result.uri = lsDocumentUri::FromPath(include.resolved_path);
          out.result.push_back(result);
          has_symbol = true;
          break;
        }
      }
      // Find the best match of the identifier at point.
      if (!has_symbol) {
        lsPosition position = request->params.position;
        const std::string& buffer = working_file->buffer_content;
        std::string_view query = LexIdentifierAroundPos(position, buffer);
        std::string_view short_query = query;
        {
          auto pos = query.rfind(':');
          if (pos != std::string::npos)
            short_query = query.substr(pos + 1);
        }

        // For symbols whose short/detailed names contain |query| as a
        // substring, we use the tuple <length difference, negative position,
        // not in the same file, line distance> to find the best match.
        std::tuple<int, int, bool, int> best_score{INT_MAX, 0, true, 0};
        int best_i = -1;
        for (int i = 0; i < (int)db->symbols.size(); ++i) {
          if (db->symbols[i].kind == SymbolKind::Invalid)
            continue;

          std::string_view name = short_query.size() < query.size()
                                      ? db->GetSymbolDetailedName(i)
                                      : db->GetSymbolShortName(i);
          auto pos = name.rfind(short_query);
          if (pos == std::string::npos)
            continue;
          if (optional<QueryId::LexicalRef> use =
                  GetDefinitionSpell(db, db->symbols[i])) {
            std::tuple<int, int, bool, int> score{
                int(name.size() - short_query.size()), -int(pos),
                use->file != file_id,
                std::abs(use->range.start.line - position.line)};
            // Update the score with qualified name if the qualified name
            // occurs in |name|.
            pos = name.rfind(query);
            if (pos != std::string::npos) {
              std::get<0>(score) = int(name.size() - query.size());
              std::get<1>(score) = -int(pos);
            }
            if (score < best_score) {
              best_score = score;
              best_i = i;
            }
          }
        }
        if (best_i != -1) {
          optional<QueryId::LexicalRef> use =
              GetDefinitionSpell(db, db->symbols[best_i]);
          assert(use);
          if (auto ls_loc = GetLsLocation(db, working_files, *use))
            out.result.push_back(*ls_loc);
        }
      }
    }

    QueueManager::WriteStdout(kMethodType, out);
  }
};
REGISTER_MESSAGE_HANDLER(Handler_TextDocumentDefinition);
}  // namespace
