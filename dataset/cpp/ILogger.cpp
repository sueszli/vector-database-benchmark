// SPDX-License-Identifier: MPL-2.0
// Copyright © 2020 Skyline Team and Contributors (https://github.com/skyline-emu/)

#include <sstream>
#include "ILogger.h"

namespace skyline::service::lm {
    ILogger::ILogger(const DeviceState &state, ServiceManager &manager) : BaseService(state, manager) {}

    Result ILogger::Log(type::KSession &session, ipc::IpcRequest &request, ipc::IpcResponse &response) {
        auto inputBuffer{request.inputBuf.at(0)};
        struct Data {
            u64 pid;
            u64 threadContext;
            u16 flags;
            LogLevel level;
            u8 verbosity;
            u32 payloadLength;
        } &data = inputBuffer.as<Data>();

        struct LogMessage {
            std::string_view message;
            u32 line;
            std::string_view filename;
            std::string_view function;
            std::string_view module;
            std::string_view thread;
            u64 dropCount;
            u64 time;
            std::string_view program;
        } logMessage{};

        u64 offset{sizeof(Data)};
        while ((offset + sizeof(LogFieldType) + sizeof(u8)) < inputBuffer.size()) { // The length of the last field sometimes doesn't add up to the buffer size, so we need to terminate the loop when we can't pop the type and length off the buffer
            auto fieldType{inputBuffer.subspan(offset++).as<LogFieldType>()};
            auto length{inputBuffer.subspan(offset++).as<u8>()};
            auto object{inputBuffer.subspan(offset, length)};

            switch (fieldType) {
                case LogFieldType::Start:
                    offset += length;
                    continue;
                case LogFieldType::Stop:
                    break;
                case LogFieldType::Message:
                    logMessage.message = object.as_string();
                    offset += length;
                    continue;
                case LogFieldType::Line:
                    logMessage.line = object.as<u32>();
                    offset += sizeof(u32);
                    continue;
                case LogFieldType::Filename: {
                    logMessage.filename = object.as_string();
                    auto position{logMessage.filename.find_last_of('/')};
                    logMessage.filename.remove_prefix(position != std::string::npos ? position + 1 : 0);
                    offset += length;
                    continue;
                }
                case LogFieldType::Function:
                    logMessage.function = object.as_string();
                    offset += length;
                    continue;
                case LogFieldType::Module:
                    logMessage.module = object.as_string();
                    offset += length;
                    continue;
                case LogFieldType::Thread:
                    logMessage.thread = object.as_string();
                    offset += length;
                    continue;
                case LogFieldType::DropCount:
                    logMessage.dropCount = object.as<u64>();
                    offset += sizeof(u64);
                    continue;
                case LogFieldType::Time:
                    logMessage.time = object.as<u64>();
                    offset += sizeof(u64);
                    continue;
                case LogFieldType::ProgramName:
                    logMessage.program = object.as_string();
                    offset += length;
                    continue;
            }
            break;
        }

        Logger::LogLevel hostLevel{[&data]() {
            switch (data.level) {
                case LogLevel::Trace:
                    return Logger::LogLevel::Debug;
                case LogLevel::Info:
                    return Logger::LogLevel::Info;
                case LogLevel::Warning:
                    return Logger::LogLevel::Warn;
                case LogLevel::Error:
                case LogLevel::Critical:
                    return Logger::LogLevel::Error;
            }
        }()};

        std::ostringstream message;
        if (!logMessage.filename.empty())
            message << logMessage.filename << ':';
        if (logMessage.line)
            message << 'L' << std::dec << logMessage.line << ':';
        if (!logMessage.program.empty())
            message << logMessage.program << ':';
        if (!logMessage.module.empty())
            message << logMessage.module << ':';
        if (!logMessage.function.empty())
            message << logMessage.function << "():";
        if (!logMessage.thread.empty())
            message << logMessage.thread << ':';
        if (logMessage.time)
            message << logMessage.time << "s:";
        if (!logMessage.message.empty()) {
            if (logMessage.message.ends_with('\n'))
                logMessage.message.remove_suffix(1);
            message << ' ' << logMessage.message;
        }
        if (logMessage.dropCount)
            message << " (Dropped Messages: " << logMessage.time << ')';

        Logger::Write(hostLevel, message.str());

        return {};
    }

    Result ILogger::SetDestination(type::KSession &session, ipc::IpcRequest &request, ipc::IpcResponse &response) {
        return {};
    }
}
