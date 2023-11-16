
#pragma once

#include <atomic>
#include <memory>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include "jet/live/Compiler.hpp"
#include "jet/live/FileWatcher.hpp"
#include "jet/live/ILiveListener.hpp"
#include "jet/live/LiveConfig.hpp"
#include "jet/live/LiveContext.hpp"
#include "jet/live/Status.hpp"

namespace jet
{
    struct CompilationUnit;

    /**
     * Main entry point of the library.
     * Ties together all the parts of functionality.
     */
    class Live
    {
    public:
        /**
         * Initialization is performed in a background thread.
         *
         */
        Live(std::unique_ptr<ILiveListener>&& listener = {}, const LiveConfig& config = {});
        ~Live();

        /**
         * Tries to reload changed code.
         * It will wait for all pending compilation processes to finish.
         * Does nothing if there's no changes.
         * Call it only when you're done editing your code.
         */
        void tryReload();

        /**
         * Runloop method, should be periodically called by the application.
         */
        void update();

        /**
         * Checks if initialization is finished.
         */
        bool isInitialized() const;

        /**
         * Retrieves status of the library.
         */
        Status getStatus() const;

    private:
        std::unique_ptr<LiveContext> m_context;
        std::unique_ptr<FileWatcher> m_fileWatcher;
        std::unique_ptr<Compiler> m_compiler;
        int m_recreateFileWatcherAfterTicks = 0;
        const int m_recreateFileWatcherMaxTicks = 10;
        std::thread m_initThread;
        std::atomic_bool m_initialized{false};
        std::atomic_bool m_earlyExit{false};

        void loadCompilationUnits();
        void loadSymbols();
        void loadExportedSymbols();
        void loadDependencies();
        void setupFileWatcher();
        void updateDependencies(CompilationUnit& cu);
        std::unordered_set<std::string> getDirectoriesToMonitor();
        std::unordered_set<std::string> getDirectoryFilters();

        void onFileChanged(const std::string& filepath);
        void tryReloadInternal();
    };
}
