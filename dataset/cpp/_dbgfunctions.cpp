/**
 @file _dbgfunctions.cpp

 @brief Implements the dbgfunctions class.
 */

#include "_global.h"
#include "_dbgfunctions.h"
#include "assemble.h"
#include "debugger.h"
#include "jit.h"
#include "patches.h"
#include "memory.h"
#include "disasm_fast.h"
#include "stackinfo.h"
#include "symbolinfo.h"
#include "module.h"
#include "exhandlerinfo.h"
#include "breakpoint.h"
#include "threading.h"
#include "stringformat.h"
#include "TraceRecord.h"
#include "mnemonichelp.h"
#include "handles.h"
#include "../bridge/bridgelist.h"
#include "tcpconnections.h"
#include "watch.h"
#include "animate.h"
#include "thread.h"
#include "comment.h"
#include "exception.h"
#include "database.h"
#include "dbghelp_safe.h"

static DBGFUNCTIONS _dbgfunctions;

const DBGFUNCTIONS* dbgfunctionsget()
{
    return &_dbgfunctions;
}

static bool _assembleatex(duint addr, const char* instruction, char* error, bool fillnop)
{
    return assembleat(addr, instruction, nullptr, error, fillnop);
}

static bool _sectionfromaddr(duint addr, char* section)
{
    std::vector<MODSECTIONINFO> sections;
    if(ModSectionsFromAddr(addr, &sections))
    {
        for(const auto & cur : sections)
        {
            if(addr >= cur.addr && addr < cur.addr + (cur.size + (0x1000 - 1) & ~(0x1000 - 1)))
            {
                strncpy_s(section, MAX_SECTION_SIZE * 5, cur.name, _TRUNCATE);
                return true;
            }
        }
    }
    return false;
}

static bool _patchget(duint addr)
{
    return PatchGet(addr, nullptr);
}

static bool _patchinrange(duint start, duint end)
{
    if(start > end)
        std::swap(start, end);

    for(duint i = start; i <= end; i++)
    {
        if(_patchget(i))
            return true;
    }

    return false;
}

static bool _mempatch(duint va, const unsigned char* src, duint size)
{
    return MemPatch(va, src, size, nullptr);
}

static void _patchrestorerange(duint start, duint end)
{
    if(start > end)
        std::swap(start, end);

    for(duint i = start; i <= end; i++)
        PatchDelete(i, true);

    GuiUpdatePatches();
}

static bool _patchrestore(duint addr)
{
    return PatchDelete(addr, true);
}

static void _getcallstack(DBGCALLSTACK* callstack)
{
    if(hActiveThread)
        stackgetcallstack(GetContextDataEx(hActiveThread, UE_CSP), (CALLSTACK*)callstack);
}

static void _getcallstackbythread(HANDLE thread, DBGCALLSTACK* callstack)
{
    if(thread)
        stackgetcallstackbythread(thread, (CALLSTACK*)callstack);
}

static void _getsehchain(DBGSEHCHAIN* sehchain)
{
    std::vector<duint> SEHList;
    ExHandlerGetSEH(SEHList);
    sehchain->total = SEHList.size();
    if(sehchain->total > 0)
    {
        sehchain->records = (DBGSEHRECORD*)BridgeAlloc(sehchain->total * sizeof(DBGSEHRECORD));
        for(size_t i = 0; i < sehchain->total; i++)
        {
            sehchain->records[i].addr = SEHList[i];
            MemRead(SEHList[i] + 4, &sehchain->records[i].handler, sizeof(duint));
        }
    }
    else
    {
        sehchain->records = nullptr;
    }
}

static bool _getjitauto(bool* jit_auto)
{
    return dbggetjitauto(jit_auto, notfound, NULL, NULL);
}

static bool _getcmdline(char* cmd_line, size_t* cbsize)
{
    if(!cmd_line && !cbsize)
        return false;
    char* cmdline;
    if(!dbggetcmdline(&cmdline, NULL, fdProcessInfo->hProcess))
        return false;
    if(!cmd_line && cbsize)
        *cbsize = strlen(cmdline) + sizeof(char);
    else if(cmd_line)
        memcpy(cmd_line, cmdline, strlen(cmdline) + 1);
    efree(cmdline, "_getcmdline:cmdline");
    return true;
}

static bool _setcmdline(const char* cmd_line)
{
    return dbgsetcmdline(cmd_line, nullptr);
}

static bool _getjit(char* jit, bool jit64)
{
    arch dummy;
    char jit_tmp[JIT_ENTRY_MAX_SIZE] = "";
    if(jit != NULL)
    {
        if(!dbggetjit(jit_tmp, jit64 ? x64 : x32, &dummy, NULL))
            return false;
        strcpy_s(jit, MAX_SETTING_SIZE, jit_tmp);
    }
    else // if jit input == NULL: it returns false if there are not an OLD JIT STORED.
    {
        Memory<char*> oldjit(MAX_SETTING_SIZE + 1);
        if(!BridgeSettingGet("JIT", "Old", oldjit()))
            return false;
    }

    return true;
}

static bool _getprocesslist(DBGPROCESSINFO** entries, int* count)
{
    std::vector<PROCESSENTRY32> infoList;
    std::vector<std::string> commandList;
    std::vector<std::string> winTextList;
    if(!dbglistprocesses(&infoList, &commandList, &winTextList))
        return false;
    *count = (int)infoList.size();
    if(!*count)
    {
        *entries = nullptr;
        return false;
    }
    *entries = (DBGPROCESSINFO*)BridgeAlloc(*count * sizeof(DBGPROCESSINFO));
    for(int i = 0; i < *count; i++)
    {
        (*entries)[*count - i - 1].dwProcessId = infoList.at(i).th32ProcessID;
        strncpy_s((*entries)[*count - i - 1].szExeFile, infoList.at(i).szExeFile, _TRUNCATE);
        strncpy_s((*entries)[*count - i - 1].szExeMainWindowTitle, winTextList.at(i).c_str(), _TRUNCATE);
        strncpy_s((*entries)[*count - i - 1].szExeArgs, commandList.at(i).c_str(), _TRUNCATE);
    }
    return true;
}

static void _memupdatemap()
{
    MemUpdateMap();
    GuiUpdateMemoryView();
}

static duint _getaddrfromline(const char* szSourceFile, int line, duint* disp)
{
    if(disp)
        *disp = 0;
    return 0;
}

static duint _getaddrfromlineex(duint mod, const char* szSourceFile, int line)
{
    duint addr = 0;
    if(SymGetSourceAddr(mod, szSourceFile, line, &addr))
        return addr;
    return 0;
}

static bool _getsourcefromaddr(duint addr, char* szSourceFile, int* line)
{
    char sourceFile[MAX_STRING_SIZE] = "";
    if(!SymGetSourceLine(addr, sourceFile, line))
        return false;
    if(!FileExists(sourceFile))
        return false;
    if(szSourceFile)
        strcpy_s(szSourceFile, MAX_STRING_SIZE, sourceFile);
    return true;
}

static bool _valfromstring(const char* string, duint* value)
{
    return valfromstring(string, value);
}

static bool _getbridgebp(BPXTYPE type, duint addr, BRIDGEBP* bp)
{
    BP_TYPE bptype;
    switch(type)
    {
    case bp_normal:
        bptype = BPNORMAL;
        break;
    case bp_hardware:
        bptype = BPHARDWARE;
        break;
    case bp_memory:
        bptype = BPMEMORY;
        break;
    case bp_dll:
        bptype = BPDLL;
        addr = ModHashFromName(reinterpret_cast<const char*>(addr));
        break;
    case bp_exception:
        bptype = BPEXCEPTION;
        break;
    default:
        return false;
    }
    SHARED_ACQUIRE(LockBreakpoints);
    auto bpInfo = BpInfoFromAddr(bptype, addr);
    if(!bpInfo)
        return false;
    if(bp)
    {
        BpToBridge(bpInfo, bp);
        bp->addr = addr;
    }
    return true;
}

static bool _stringformatinline(const char* format, size_t resultSize, char* result)
{
    if(!format || !result)
        return false;
    strncpy_s(result, resultSize, stringformatinline(format).c_str(), _TRUNCATE);
    return true;
}

static void _getmnemonicbrief(const char* mnem, size_t resultSize, char* result)
{
    if(!result)
        return;
    strncpy_s(result, resultSize, MnemonicHelp::getBriefDescription(mnem).c_str(), _TRUNCATE);
}

static bool _enumhandles(ListOf(HANDLEINFO) handles)
{
    std::vector<HANDLEINFO> handleV;
    if(!HandlesEnum(handleV))
        return false;
    return BridgeList<HANDLEINFO>::CopyData(handles, handleV);
}

static bool _gethandlename(duint handle, char* name, size_t nameSize, char* typeName, size_t typeNameSize)
{
    String nameS;
    String typeNameS;
    if(!HandlesGetName(HANDLE(handle), nameS, typeNameS))
        return false;
    strncpy_s(name, nameSize, nameS.c_str(), _TRUNCATE);
    strncpy_s(typeName, typeNameSize, typeNameS.c_str(), _TRUNCATE);
    return true;
}

static bool _enumtcpconnections(ListOf(TCPCONNECTIONINFO) connections)
{
    std::vector<TCPCONNECTIONINFO> connectionsV;
    if(!TcpEnumConnections(fdProcessInfo->dwProcessId, connectionsV))
        return false;
    return BridgeList<TCPCONNECTIONINFO>::CopyData(connections, connectionsV);
}

static bool _enumwindows(ListOf(WINDOW_INFO) windows)
{
    std::vector<WINDOW_INFO> windowInfoV;
    if(!HandlesEnumWindows(windowInfoV))
        return false;
    return BridgeList<WINDOW_INFO>::CopyData(windows, windowInfoV);
}

static bool _enumheaps(ListOf(HEAPINFO) heaps)
{
    std::vector<HEAPINFO> heapInfoV;
    if(!HandlesEnumHeaps(heapInfoV))
        return false;
    return BridgeList<HEAPINFO>::CopyData(heaps, heapInfoV);
}

static void _getcallstackex(DBGCALLSTACK* callstack, bool cache)
{
    auto csp = GetContextDataEx(hActiveThread, UE_CSP);
    if(!cache)
        stackupdatecallstack(csp);
    stackgetcallstack(csp, (CALLSTACK*)callstack);
}

static void _enumconstants(ListOf(CONSTANTINFO) constants)
{
    auto constantsV = ConstantList();
    BridgeList<CONSTANTINFO>::CopyData(constants, constantsV);
}

static void _enumerrorcodes(ListOf(CONSTANTINFO) errorcodes)
{
    auto errorcodesV = ErrorCodeList();
    BridgeList<CONSTANTINFO>::CopyData(errorcodes, errorcodesV);
}

static void _enumexceptions(ListOf(CONSTANTINFO) exceptions)
{
    auto exceptionsV = ExceptionList();
    BridgeList<CONSTANTINFO>::CopyData(exceptions, exceptionsV);
}

static duint _membpsize(duint addr)
{
    SHARED_ACQUIRE(LockBreakpoints);
    auto info = BpInfoFromAddr(BPMEMORY, addr);
    return info ? info->memsize : 0;
}

static bool _modrelocationsfromaddr(duint addr, ListOf(DBGRELOCATIONINFO) relocations)
{
    std::vector<MODRELOCATIONINFO> infos;
    if(!ModRelocationsFromAddr(addr, infos))
        return false;

    BridgeList<MODRELOCATIONINFO>::CopyData(relocations, infos);
    return true;
}

static bool _modrelocationsinrange(duint addr, duint size, ListOf(DBGRELOCATIONINFO) relocations)
{
    std::vector<MODRELOCATIONINFO> infos;
    if(!ModRelocationsInRange(addr, size, infos))
        return false;

    BridgeList<MODRELOCATIONINFO>::CopyData(relocations, infos);
    return true;
}

static int SymAutoComplete(const char* Search, char** Buffer, int MaxSymbols)
{
    //TODO: refactor this in a function because this pattern will become common
    std::vector<duint> mods;
    ModEnum([&mods](const MODINFO & info)
    {
        mods.push_back(info.base);
    });

    std::unordered_set<std::string> visited;

    static const bool caseSensitiveAutoComplete = settingboolget("Gui", "CaseSensitiveAutoComplete");

    int count = 0;
    std::string prefix(Search);
    for(duint base : mods)
    {
        if(count == MaxSymbols)
            break;

        SHARED_ACQUIRE(LockModules);
        auto modInfo = ModInfoFromAddr(base);
        if(!modInfo)
            continue;

        auto addName = [Buffer, MaxSymbols, &visited, &count](const std::string & name)
        {
            if(visited.count(name))
                return true;
            visited.insert(name);
            Buffer[count] = (char*)BridgeAlloc(name.size() + 1);
            memcpy(Buffer[count], name.c_str(), name.size() + 1);
            return ++count < MaxSymbols;
        };

        NameIndex::findByPrefix(modInfo->exportsByName, prefix, [modInfo, &addName](const NameIndex & index)
        {
            return addName(modInfo->exports[index.index].name);
        }, caseSensitiveAutoComplete);

        if(count == MaxSymbols)
            break;

        if(modInfo->symbols->isOpen())
        {
            modInfo->symbols->findSymbolsByPrefix(prefix, [&addName](const SymbolInfo & symInfo)
            {
                return addName(symInfo.decoratedName);
            }, caseSensitiveAutoComplete);
        }
    }

    std::stable_sort(Buffer, Buffer + count, [](const char* a, const char* b)
    {
        return (caseSensitiveAutoComplete ? strcmp : StringUtils::hackicmp)(a, b) < 0;
    });

    return count;
}

static MODULESYMBOLSTATUS _modsymbolstatus(duint base)
{
    SHARED_ACQUIRE(LockModules);
    auto modInfo = ModInfoFromAddr(base);
    if(!modInfo)
        return MODSYMUNLOADED;
    bool isOpen = modInfo->symbols->isOpen();
    bool isLoading = modInfo->symbols->isLoading();
    if(isOpen && !isLoading)
        return MODSYMLOADED;
    else if(isOpen && isLoading)
        return MODSYMLOADING;
    else if(!isOpen && symbolDownloadingBase == base)
        return MODSYMLOADING;
    else
        return MODSYMUNLOADED;
}

static void _refreshmodulelist()
{
    SymUpdateModuleList();
}

static unsigned int _getTraceRecordHitCount(duint address)
{
    return TraceRecord.getHitCount(address);
}

static TRACERECORDBYTETYPE _getTraceRecordByteType(duint address)
{
    return (TRACERECORDBYTETYPE)TraceRecord.getByteType(address);
}

static bool _setTraceRecordType(duint pageAddress, TRACERECORDTYPE type)
{
    return TraceRecord.setTraceRecordType(pageAddress, (TraceRecordManager::TraceRecordType)type);
}

static TRACERECORDTYPE _getTraceRecordType(duint pageAddress)
{
    return (TRACERECORDTYPE)TraceRecord.getTraceRecordType(pageAddress);
}

static bool _enableTraceRecording(bool enabled, const char* fileName)
{
    return TraceRecord.enableTraceRecording(enabled, fileName);
}

static bool _isTraceRecordingEnabled()
{
    return TraceRecord.isTraceRecordingEnabled();
}

void dbgfunctionsinit()
{
    _dbgfunctions.AssembleAtEx = _assembleatex;
    _dbgfunctions.SectionFromAddr = _sectionfromaddr;
    _dbgfunctions.ModNameFromAddr = ModNameFromAddr;
    _dbgfunctions.ModBaseFromAddr = ModBaseFromAddr;
    _dbgfunctions.ModBaseFromName = ModBaseFromName;
    _dbgfunctions.ModSizeFromAddr = ModSizeFromAddr;
    _dbgfunctions.ModGetParty = ModGetParty;
    _dbgfunctions.ModSetParty = ModSetParty;
    _dbgfunctions.WatchIsWatchdogTriggered = WatchIsWatchdogTriggered;
    _dbgfunctions.Assemble = assemble;
    _dbgfunctions.PatchGet = _patchget;
    _dbgfunctions.PatchInRange = _patchinrange;
    _dbgfunctions.MemPatch = _mempatch;
    _dbgfunctions.PatchRestoreRange = _patchrestorerange;
    _dbgfunctions.PatchEnum = (PATCHENUM)PatchEnum;
    _dbgfunctions.PatchRestore = _patchrestore;
    _dbgfunctions.PatchFile = (PATCHFILE)PatchFile;
    _dbgfunctions.ModPathFromAddr = ModPathFromAddr;
    _dbgfunctions.ModPathFromName = ModPathFromName;
    _dbgfunctions.DisasmFast = disasmfast;
    _dbgfunctions.MemUpdateMap = _memupdatemap;
    _dbgfunctions.GetCallStack = _getcallstack;
    _dbgfunctions.GetSEHChain = _getsehchain;
    _dbgfunctions.SymbolDownloadAllSymbols = SymDownloadAllSymbols;
    _dbgfunctions.GetJit = _getjit;
    _dbgfunctions.GetJitAuto = _getjitauto;
    _dbgfunctions.GetDefJit = dbggetdefjit;
    _dbgfunctions.GetProcessList = _getprocesslist;
    _dbgfunctions.GetPageRights = MemGetPageRights;
    _dbgfunctions.SetPageRights = MemSetPageRights;
    _dbgfunctions.PageRightsToString = MemPageRightsToString;
    _dbgfunctions.IsProcessElevated = BridgeIsProcessElevated;
    _dbgfunctions.GetCmdline = _getcmdline;
    _dbgfunctions.SetCmdline = _setcmdline;
    _dbgfunctions.FileOffsetToVa = valfileoffsettova;
    _dbgfunctions.VaToFileOffset = valvatofileoffset;
    _dbgfunctions.GetAddrFromLine = _getaddrfromline;
    _dbgfunctions.GetSourceFromAddr = _getsourcefromaddr;
    _dbgfunctions.ValFromString = _valfromstring;
    _dbgfunctions.PatchGetEx = (PATCHGETEX)PatchGet;
    _dbgfunctions.GetBridgeBp = _getbridgebp;
    _dbgfunctions.StringFormatInline = _stringformatinline;
    _dbgfunctions.GetMnemonicBrief = _getmnemonicbrief;
    _dbgfunctions.GetTraceRecordHitCount = _getTraceRecordHitCount;
    _dbgfunctions.GetTraceRecordByteType = _getTraceRecordByteType;
    _dbgfunctions.SetTraceRecordType = _setTraceRecordType;
    _dbgfunctions.GetTraceRecordType = _getTraceRecordType;
    _dbgfunctions.EnumHandles = _enumhandles;
    _dbgfunctions.GetHandleName = _gethandlename;
    _dbgfunctions.EnumTcpConnections = _enumtcpconnections;
    _dbgfunctions.GetDbgEvents = dbggetdbgevents;
    _dbgfunctions.MemIsCodePage = MemIsCodePage;
    _dbgfunctions.AnimateCommand = _dbg_animatecommand;
    _dbgfunctions.DbgSetDebuggeeInitScript = dbgsetdebuggeeinitscript;
    _dbgfunctions.DbgGetDebuggeeInitScript = dbggetdebuggeeinitscript;
    _dbgfunctions.EnumWindows = _enumwindows;
    _dbgfunctions.EnumHeaps = _enumheaps;
    _dbgfunctions.ThreadGetName = ThreadGetName;
    _dbgfunctions.IsDepEnabled = dbgisdepenabled;
    _dbgfunctions.GetCallStackEx = _getcallstackex;
    _dbgfunctions.GetUserComment = CommentGet;
    _dbgfunctions.EnumConstants = _enumconstants;
    _dbgfunctions.EnumErrorCodes = _enumerrorcodes;
    _dbgfunctions.EnumExceptions = _enumexceptions;
    _dbgfunctions.MemBpSize = _membpsize;
    _dbgfunctions.ModRelocationsFromAddr = _modrelocationsfromaddr;
    _dbgfunctions.ModRelocationAtAddr = (MODRELOCATIONATADDR)ModRelocationAtAddr;
    _dbgfunctions.ModRelocationsInRange = _modrelocationsinrange;
    _dbgfunctions.DbGetHash = DbGetHash;
    _dbgfunctions.SymAutoComplete = SymAutoComplete;
    _dbgfunctions.RefreshModuleList = _refreshmodulelist;
    _dbgfunctions.GetAddrFromLineEx = _getaddrfromlineex;
    _dbgfunctions.ModSymbolStatus = _modsymbolstatus;
    _dbgfunctions.GetCallStackByThread = _getcallstackbythread;
}
