/*
 * CommandFactory.cpp
 * 
 * This file is part of the XShaderCompiler project (Copyright (c) 2014-2018 by Lukas Hermanns)
 * See "LICENSE.txt" for license information.
 */

#include "CommandFactory.h"
#include <tuple>


namespace Xsc
{

namespace Util
{


const CommandFactory& CommandFactory::Instance()
{
    static const CommandFactory instance;
    return instance;
}

Command* CommandFactory::Get(const std::string& name, Command::Identifier* cmdIdent) const
{
    for (const auto& cmd : commands_)
    {
        for (const auto& ident : cmd->Idents())
        {
            auto identLen = ident.name.size();
            if ( ( !ident.includesValue && name == ident.name ) ||
                 ( ident.includesValue && name.size() >= identLen && name.substr(0, identLen) == ident.name ) )
            {
                if (cmdIdent)
                    *cmdIdent = ident;
                return cmd.get();
            }
        }
    }
    return nullptr;
}


/*
 * ======= Private: =======
 */

CommandFactory::CommandFactory()
{
    MakeStandardCommands
    <
        EntryCommand,
        SecndEntryCommand,
        TargetCommand,
        VersionInCommand,
        VersionOutCommand,
        IncludePathCommand,

        #ifdef XSC_ENABLE_LANGUAGE_EXT
        LanguageExtensionCommand,
        #endif

        OutputCommand,
        WarnCommand,
        ShowASTCommand,
        ShowTimesCommand,
        ReflectCommand,
        PPOnlyCommand,
        MacroCommand,
        SemanticCommand,
        PackUniformsCommand,
        PauseCommand,
        PresettingCommand,
        VersionCommand,
        HelpCommand,
        VerboseCommand,
        ColorCommand,
        OptimizeCommand,
        ExtensionCommand,
        EnumExtensionCommand,
        ValidateCommand,
        BindingCommand,
        CommentCommand,
        WrapperCommand,
        UnrollInitializerCommand,
        ObfuscateCommand,
        RowMajorAlignmentCommand,
        AutoBindingCommand,
        AutoBindingStartSlotCommand,
        FormattingCommand,
        IndentCommand,
        PrefixCommand,
        NameManglingCommand,
        SeparateShadersCommand,
        SeparateSamplersCommand,
        DisassembleCommand,
        DisassembleExtCommand
    >();
}

template <typename T, typename... Args>
void CommandFactory::MakeCommand(Args&&... args)
{
    auto cmd = std::unique_ptr<T>(new T(std::forward<Args>(args)...));
    helpPrinter_.AppendCommandHelp(*cmd);
    commands_.emplace_back(std::move(cmd));
}

template <typename... Commands>
void CommandFactory::MakeStandardCommands()
{
    MakeStandardCommandsFirst<Commands...>();
    MakeStandardCommandsNext<Commands...>();
}

// No declaration for template specialization (not allowed with GCC)
template <>
void CommandFactory::MakeStandardCommands()
{
    // do nothing
}

template <typename FirstCommand, typename... NextCommands>
void CommandFactory::MakeStandardCommandsFirst()
{
    MakeCommand<FirstCommand>();
}

template <typename FirstCommand, typename... NextCommands>
void CommandFactory::MakeStandardCommandsNext()
{
    MakeStandardCommands<NextCommands...>();
}


} // /namespace Util

} // /namespace Xsc



// ================================================================================
