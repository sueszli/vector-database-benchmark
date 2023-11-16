
#include "LinkTimeRelocationsStep.hpp"
#include <cstdint>
#include <limits>
#include "jet/live/DataTypes.hpp"
#include "jet/live/LiveContext.hpp"
#include "jet/live/Utility.hpp"

namespace jet
{
    void LinkTimeRelocationsStep::reload(LiveContext* context, Program* newProgram)
    {
        context->events->addLog(LogSeverity::kDebug, "Loading link-time relocations...");

        const auto& relocs = context->programInfoLoader->getLinkTimeRelocations(context, newProgram->objFilePaths);
        auto totalRelocs = relocs.size();
        size_t appliedRelocs = 0;
        std::vector<Symbol> relocatedSymbols;
        for (const auto& reloc : relocs) {
            const Symbol* targetSymbol =
                findFunction(newProgram->symbols, reloc.targetSymbolName, reloc.targetSymbolHash);
            if (!targetSymbol) {
                context->events->addLog(LogSeverity::kError,
                    "targetSymbol not found: " + reloc.targetSymbolName + " " + std::to_string(reloc.targetSymbolHash));
                continue;
            }

            const Symbol* relocSymbol =
                findVariable(newProgram->symbols, reloc.relocationSymbolName, reloc.relocationSymbolHash);
            if (!relocSymbol) {
                context->events->addLog(LogSeverity::kError,
                    "relocSymbol not found: " + reloc.relocationSymbolName + " "
                        + std::to_string(reloc.relocationSymbolHash));
                continue;
            }

            const Symbol* oldVar = nullptr;
            auto& progs = context->programs;
            for (const auto& prog : progs) {
                oldVar = findVariable(prog.symbols, reloc.relocationSymbolName, reloc.relocationSymbolHash);
                if (oldVar) {
                    break;
                }
            }
            if (!oldVar) {
                continue;
            }

            auto relocAddressVal = targetSymbol->runtimeAddress + reloc.relocationOffsetRelativeTargetSymbolAddress;
            auto distance = std::abs(static_cast<intptr_t>(oldVar->runtimeAddress - relocSymbol->runtimeAddress));
            int64_t maxAllowedDistance = 0;
            if (reloc.size == sizeof(int32_t)) {
                maxAllowedDistance = std::numeric_limits<int32_t>::max();
            } else if (reloc.size == sizeof(int64_t)) {
                maxAllowedDistance = std::numeric_limits<int64_t>::max();
            } else {
                context->events->addLog(LogSeverity::kError, "LinkTimeRelocationsStep: WTF");
                continue;
            }
            if (distance > maxAllowedDistance) {
                context->events->addLog(LogSeverity::kWarning,
                    "Cannot apply relocation for " + relocSymbol->name
                        + ", distance doesn't fit into max allowed distance");
                continue;
            }

            auto relocAddress = reinterpret_cast<void*>(relocAddressVal);
            if (!unprotect(relocAddress, reloc.size)) {
                context->events->addLog(LogSeverity::kError, "'unprotect' failed");
                continue;
            }
            if (reloc.size == sizeof(int32_t)) {
                *reinterpret_cast<int32_t*>(relocAddress) += oldVar->runtimeAddress - relocSymbol->runtimeAddress;
            } else if (reloc.size == sizeof(int64_t)) {
                *reinterpret_cast<int64_t*>(relocAddress) += oldVar->runtimeAddress - relocSymbol->runtimeAddress;
            }
            context->events->addLog(LogSeverity::kDebug, relocSymbol->name + " was relocated");

            relocatedSymbols.push_back(*relocSymbol);
            appliedRelocs++;
        }

        for (const auto& relocSymbol : relocatedSymbols) {
            auto& newVars = newProgram->symbols.variables[relocSymbol.name];
            for (size_t i = 0; i < newVars.size(); i++) {
                if (newVars[i].hash == relocSymbol.hash) {
                    newVars.erase(newVars.begin() + i);
                    break;
                }
            }
            if (newVars.empty()) {
                newProgram->symbols.variables.erase(relocSymbol.name);
            }
        }

        context->events->addLog(LogSeverity::kDebug,
            "Done, relocated: " + std::to_string(appliedRelocs) + "/" + std::to_string(totalRelocs));
    }
}
