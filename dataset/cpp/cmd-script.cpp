#include "cmd-script.h"
#include "stringformat.h"
#include "console.h"
#include "variable.h"
#include "stackinfo.h"
#include "debugger.h"
#include "simplescript.h"

bool cbScriptLoad(int argc, char* argv[])
{
    if(argc < 2)
        return false;
    scriptload(argv[1]);
    return true;
}

bool cbScriptMsg(int argc, char* argv[])
{
    if(IsArgumentsLessThan(argc, 2))
        return false;
    GuiScriptMessage(stringformatinline(argv[1]).c_str());
    return true;
}

bool cbScriptMsgyn(int argc, char* argv[])
{
    if(IsArgumentsLessThan(argc, 2))
        return false;
    varset("$RESULT", GuiScriptMsgyn(stringformatinline(argv[1]).c_str()), false);
    return true;
}

bool cbScriptCmd(int argc, char* argv[])
{
    if(IsArgumentsLessThan(argc, 2))
        return false;
    auto scriptcmd = strchr(argv[0], ' ');
    if(scriptcmd == nullptr)
        return false;
    while(isspace(*scriptcmd))
        scriptcmd++;
    return scriptcmdexec(scriptcmd);
}

static bool cbGenericLog(int argc, char* argv[], void(*logputs)(const char* msg))
{
    if(argc == 1)  //just log newline
    {
        logputs("");
        return true;
    }
    if(argc == 2)  //inline logging: log "format {rax}"
    {
        logputs(stringformatinline(argv[1]).c_str());
    }
    else //log "format {0} string", arg1, arg2, argN
    {
        FormatValueVector formatArgs;
        for(auto i = 2; i < argc; i++)
            formatArgs.push_back(argv[i]);
        logputs(stringformat(argv[1], formatArgs).c_str());
    }
    return true;
}

bool cbInstrLog(int argc, char* argv[])
{
    return cbGenericLog(argc, argv, [](const char* msg)
    {
        dputs_untranslated(msg);
        scriptlog(msg);
    });
}

bool cbInstrHtmlLog(int argc, char* argv[])
{
    return cbGenericLog(argc, argv, [](const char* msg)
    {
        dprint_untranslated_html(msg);
        dprint_untranslated_html("\n");
    });
}

bool cbInstrPrintStack(int argc, char* argv[])
{
    duint csp = GetContextDataEx(hActiveThread, UE_CSP);
    std::vector<CALLSTACKENTRY> callstackVector;
    stackgetcallstack(csp, callstackVector, false);
    if(callstackVector.size() == 0)
        dputs(QT_TRANSLATE_NOOP("DBG", "No call stack."));
    else
    {
        duint cip = GetContextDataEx(hActiveThread, UE_CIP);
#ifdef _WIN64
        duint cbp = GetContextDataEx(hActiveThread, UE_RBP);
        dprintf(QT_TRANSLATE_NOOP("DBG", "%llu call stack frames (RIP = %p , RSP = %p , RBP = %p ):\n"), callstackVector.size(), cip, csp, cbp);
#else //x86
        duint cbp = GetContextDataEx(hActiveThread, UE_EBP);
        dprintf(QT_TRANSLATE_NOOP("DBG", "%u call stack frames (EIP = %p , ESP = %p , EBP = %p ):\n"), callstackVector.size(), cip, csp, cbp);
#endif //_WIN64
        for(auto & i : callstackVector)
            dprintf_untranslated("%p %s\n", i.addr, i.comment);
    }
    return true;
}
