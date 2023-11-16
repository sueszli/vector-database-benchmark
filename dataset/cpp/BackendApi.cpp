//-------------------------------------------------------------------------------------------------------
// Copyright (C) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.txt file in the project root for full license information.
//-------------------------------------------------------------------------------------------------------
#include "Backend.h"

NativeCodeGenerator *
NewNativeCodeGenerator(Js::ScriptContext * scriptContext)
{
    return HeapNew(NativeCodeGenerator, scriptContext);
}

void
DeleteNativeCodeGenerator(NativeCodeGenerator * nativeCodeGen)
{
    HeapDelete(nativeCodeGen);
}

void
CloseNativeCodeGenerator(NativeCodeGenerator * nativeCodeGen)
{
    nativeCodeGen->Close();
}

bool
IsClosedNativeCodeGenerator(NativeCodeGenerator * nativeCodeGen)
{
    return nativeCodeGen->IsClosed();
}

void SetProfileModeNativeCodeGen(NativeCodeGenerator *pNativeCodeGen, BOOL fSet)
{
    pNativeCodeGen->SetProfileMode(fSet);
}

void UpdateNativeCodeGeneratorForDebugMode(NativeCodeGenerator* nativeCodeGen)
{
    nativeCodeGen->UpdateQueueForDebugMode();
}

CriticalSection *GetNativeCodeGenCriticalSection(NativeCodeGenerator *pNativeCodeGen)
{
    return pNativeCodeGen->Processor()->GetCriticalSection();
}

///----------------------------------------------------------------------------
///
/// GenerateFunction
///
///     This is the main entry point for the runtime to call the native code
///     generator for js function.
///
///----------------------------------------------------------------------------

void
GenerateFunction(NativeCodeGenerator * nativeCodeGen, Js::FunctionBody * fn, Js::ScriptFunction * function)
{
    nativeCodeGen->GenerateFunction(fn, function);
}
InProcCodeGenAllocators* GetForegroundAllocator(NativeCodeGenerator * nativeCodeGen, PageAllocator* pageallocator)
{
    return nativeCodeGen->GetCodeGenAllocator(pageallocator);
}
#ifdef ENABLE_PREJIT
void
GenerateAllFunctions(NativeCodeGenerator * nativeCodeGen, Js::FunctionBody *fn)
{
    nativeCodeGen->GenerateAllFunctions(fn);
}
#endif
#ifdef IR_VIEWER
Js::Var
RejitIRViewerFunction(NativeCodeGenerator *nativeCodeGen, Js::FunctionBody *fn, Js::ScriptContext *scriptContext)
{
    return nativeCodeGen->RejitIRViewerFunction(fn, scriptContext);
}
#endif
#ifdef ALLOW_JIT_REPRO
HRESULT JitFromEncodedWorkItem(NativeCodeGenerator *nativeCodeGen, _In_reads_(bufferSize) const byte* buffer, _In_ uint bufferSize)
{
    return nativeCodeGen->JitFromEncodedWorkItem(buffer, bufferSize);
}
#endif

void
GenerateLoopBody(NativeCodeGenerator *nativeCodeGen, Js::FunctionBody *fn, Js::LoopHeader * loopHeader, Js::EntryPointInfo* info, uint localCount, Js::Var localSlots[])
{
    nativeCodeGen->GenerateLoopBody(fn, loopHeader, info, localCount, localSlots);
}

void
NativeCodeGenEnterScriptStart(NativeCodeGenerator * nativeCodeGen)
{
    if (nativeCodeGen)
    {
        nativeCodeGen->EnterScriptStart();
    }
}

BOOL IsIntermediateCodeGenThunk(Js::JavascriptMethod codeAddress)
{
    return NativeCodeGenerator::IsThunk(codeAddress);
}

BOOL IsAsmJsCodeGenThunk(Js::JavascriptMethod codeAddress)
{
    return NativeCodeGenerator::IsAsmJsCodeGenThunk(codeAddress);
}

CheckCodeGenFunction GetCheckCodeGenFunction(Js::JavascriptMethod codeAddress)
{
    return NativeCodeGenerator::GetCheckCodeGenFunction(codeAddress);
}

Js::JavascriptMethod GetCheckCodeGenThunk()
{
    return NativeCodeGenerator::CheckCodeGenThunk;
}

#ifdef ASMJS_PLAT
Js::JavascriptMethod GetCheckAsmJsCodeGenThunk()
{
    return NativeCodeGenerator::CheckAsmJsCodeGenThunk;
}
#endif

uint GetBailOutRegisterSaveSlotCount()
{
    // REVIEW: not all registers are used, we are allocating more space than necessary.
    return LinearScanMD::GetRegisterSaveSlotCount();
}

uint
GetBailOutReserveSlotCount()
{
    return 1; //For arguments id
}


#if DBG
void CheckIsExecutable(Js::RecyclableObject * function, Js::JavascriptMethod entrypoint)
{
    Js::ScriptContext * scriptContext = function->GetScriptContext();
    // it's easy to call the default entry point from RecyclableObject.
    AssertMsg((Js::VarIs<Js::JavascriptFunction>(function) && Js::VarTo<Js::JavascriptFunction>(function)->IsExternalFunction())
        || Js::CrossSite::IsThunk(entrypoint)
        // External object with entrypoint
        || (!Js::VarIs<Js::JavascriptFunction>(function)
            && function->IsExternal()
            && Js::JavascriptConversion::IsCallable(function))
        || !scriptContext->IsActuallyClosed()
        || (scriptContext->GetThreadContext()->IsScriptActive() && !Js::JavascriptConversion::IsCallable(function)),
        "Can't call function when the script context is closed");

    if (scriptContext->GetThreadContext()->IsScriptActive())
    {
        return;
    }
    if (function->IsExternal())
    {
        return;
    }

    Js::TypeId typeId = Js::JavascriptOperators::GetTypeId(function);
    if (typeId == Js::TypeIds_HostDispatch)
    {
        AssertMsg(false, "Has to go through CallRootFunction to start calling Javascript function");
    }
    else if (typeId == Js::TypeId::TypeIds_Function)
    {
        if (((Js::JavascriptFunction*)function)->IsExternalFunction())
        {
            return;
        }
        else if (((Js::JavascriptFunction*)function)->IsWinRTFunction())
        {
            return;
        }
        else
        {
            AssertMsg(false, "Has to go through CallRootFunction to start calling Javascript function");
        }
    }
    else
    {
        AssertMsg(false, "Has to go through CallRootFunction to start calling Javascript function");
    }
}
#endif

#ifdef PROFILE_EXEC
void
CreateProfilerNativeCodeGen(NativeCodeGenerator * nativeCodeGen, Js::ScriptContextProfiler * profiler)
{
    nativeCodeGen->CreateProfiler(profiler);
}

void
ProfilePrintNativeCodeGen(NativeCodeGenerator * nativeCodeGen)
{
    nativeCodeGen->ProfilePrint();
}

void
SetProfilerFromNativeCodeGen(NativeCodeGenerator * toNativeCodeGen, NativeCodeGenerator * fromNativeCodeGen)
{
    toNativeCodeGen->SetProfilerFromNativeCodeGen(fromNativeCodeGen);
}
#endif

void DeleteNativeCodeData(NativeCodeData * data)
{
    if (data)
    {
        HeapDelete(data);
    }
}
