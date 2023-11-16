
#pragma once

#include <memory>
#include <process.hpp>
#include <teenypath.h>
#include "jet/live/ICompilationUnitsParser.hpp"

namespace jet
{
    /**
     * Compilation units parser based on `compile_commands.json` file.
     */
    class CompileCommandsCompilationUnitsParser : public ICompilationUnitsParser
    {
    public:
        std::vector<std::string> getFilesToMonitor() const override;
        std::unordered_map<std::string, CompilationUnit> parseCompilationUnits(const LiveContext* context) override;
        bool updateCompilationUnits(LiveContext* context,
            const std::string& filepath,
            std::vector<std::string>* addedCompilationUnits,
            std::vector<std::string>* modifiedCompilationUnits,
            std::vector<std::string>* removedCompilationUnits) override;

    protected:
        TeenyPath::path m_compileCommandsPath;
        TeenyPath::path m_pbxProjPath;
        std::unique_ptr<TinyProcessLib::Process> m_runningProcess;

        std::unordered_map<std::string, CompilationUnit> parseCompilationUnitsInternal(const LiveContext* context,
            const TeenyPath::path& filepath);

        /**
         * By default it tries to find `compile_commands.json` in the executable directory
         * and all parent directories recursively.
         * For custom `compile_commands.json` location you can subclass and override this method.
         */
        virtual TeenyPath::path getCompileCommandsPath(const LiveContext* context) const;

        bool isXcodeProject() const;
        void createCompileCommandsJsonFromXcodeProject(const LiveContext* context, bool wait);
    };
}
