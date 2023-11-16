
#include "ExampleListener.hpp"
#include <iostream>

ExampleListener::ExampleListener(std::function<void()>&& codePreLoadCallback,
    std::function<void()>&& codePostLoadCallback)
    : m_codePreLoadCallback(std::move(codePreLoadCallback))
    , m_codePostLoadCallback(std::move(codePostLoadCallback))
{
}

void ExampleListener::onLog(jet::LogSeverity severity, const std::string& message)
{
    std::string severityString;
    switch (severity) {
        case jet::LogSeverity::kInfo: severityString.append("[I]"); break;
        case jet::LogSeverity::kWarning: severityString.append("[W]"); break;
        case jet::LogSeverity::kError: severityString.append("[E]"); break;
        default: return;  // Skipping debug messages, they are too verbose
    }
    std::cout << severityString << ": " << message << std::endl;
}

void ExampleListener::onCodePreLoad()
{
    m_codePreLoadCallback();
}

void ExampleListener::onCodePostLoad()
{
    m_codePostLoadCallback();
}
