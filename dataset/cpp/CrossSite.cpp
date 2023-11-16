//-------------------------------------------------------------------------------------------------------
// Copyright (C) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.txt file in the project root for full license information.
//-------------------------------------------------------------------------------------------------------
#include "RuntimeBasePch.h"
#include "Library/JavascriptProxy.h"
#include "Library/HostObjectBase.h"
#include "Types/UnscopablesWrapperObject.h"

#if ENABLE_CROSSSITE_TRACE
#define TTD_XSITE_LOG(CTX, MSG, VAR) if((CTX)->ShouldPerformRecordOrReplayAction()) \
{ \
    (CTX)->GetThreadContext()->TTDExecutionInfo->GetTraceLogger()->WriteLiteralMsg(" -XS- "); \
    (CTX)->GetThreadContext()->TTDExecutionInfo->GetTraceLogger()->WriteLiteralMsg(MSG); \
    (CTX)->GetThreadContext()->TTDExecutionInfo->GetTraceLogger()->WriteVar(VAR); \
    (CTX)->GetThreadContext()->TTDExecutionInfo->GetTraceLogger()->WriteLiteralMsg("\n"); \
}
#else
#define TTD_XSITE_LOG(CTX, MSG, VAR)
#endif

namespace Js
{

    BOOL CrossSite::NeedMarshalVar(Var instance, ScriptContext * requestContext)
    {
        if (TaggedNumber::Is(instance))
        {
            return FALSE;
        }
        RecyclableObject * object = UnsafeVarTo<RecyclableObject>(instance);
        if (object->GetScriptContext() == requestContext)
        {
            return FALSE;
        }
        if (PHASE_TRACE1(Js::Phase::MarshalPhase))
        {
            Output::Print(_u("NeedMarshalVar: %p (var sc: %p, request sc: %p)\n"), instance, object->GetScriptContext(), requestContext);
        }
        if (DynamicType::Is(object->GetTypeId()))
        {
            return !UnsafeVarTo<DynamicObject>(object)->IsCrossSiteObject() && !object->IsExternal();
        }
        return TRUE;
    }

    void CrossSite::MarshalDynamicObject(ScriptContext * scriptContext, DynamicObject * object)
    {
        Assert(!object->IsExternal() && !object->IsCrossSiteObject());

        TTD_XSITE_LOG(scriptContext, "MarshalDynamicObject", object);

        object->MarshalToScriptContext(scriptContext);
        if (object->GetTypeId() == TypeIds_Function)
        {
            AssertMsg(object != object->GetScriptContext()->GetLibrary()->GetDefaultAccessorFunction(), "default accessor marshalled");
            JavascriptFunction * function = VarTo<JavascriptFunction>(object);

            //TODO: this may be too aggressive and create x-site thunks that are't technically needed -- see uglify-2js test.

            // See if this function is one that the host needs to handle
            HostScriptContext * hostScriptContext = scriptContext->GetHostScriptContext();
            if (!hostScriptContext || !hostScriptContext->SetCrossSiteForFunctionType(function))
            {
                if (function->GetDynamicType()->GetIsLocked())
                {
                    TTD_XSITE_LOG(scriptContext, "SetCrossSiteForLockedFunctionType ", object);

                    function->GetLibrary()->SetCrossSiteForLockedFunctionType(function);
                }
                else
                {
                    TTD_XSITE_LOG(scriptContext, "setEntryPoint->CurrentCrossSiteThunk ", object);

                    function->SetEntryPoint(function->GetScriptContext()->CurrentCrossSiteThunk);
                }
            }
        }
        else if (object->GetTypeId() == TypeIds_Proxy)
        {
            RecyclableObject * target = VarTo<JavascriptProxy>(object)->GetTarget();
            if (JavascriptConversion::IsCallable(target))
            {
                Assert(JavascriptProxy::FunctionCallTrap == object->GetEntryPoint());
                TTD_XSITE_LOG(scriptContext, "setEntryPoint->CrossSiteProxyCallTrap ", object);
                object->GetDynamicType()->SetEntryPoint(CrossSite::CrossSiteProxyCallTrap);
            }
        }
    }

    void CrossSite::MarshalPrototypeChain(ScriptContext* scriptContext, DynamicObject * object)
    {
        RecyclableObject * prototype = object->GetPrototype();
        while (prototype->GetTypeId() != TypeIds_Null && prototype->GetTypeId() != TypeIds_HostDispatch)
        {
            // We should not see any static type or host dispatch here
            DynamicObject * prototypeObject = VarTo<DynamicObject>(prototype);
            if (prototypeObject->IsCrossSiteObject())
            {
                break;
            }
            if (scriptContext != prototypeObject->GetScriptContext() && !prototypeObject->IsExternal())
            {
                MarshalDynamicObject(scriptContext, prototypeObject);
            }
            if (VarIs<JavascriptProxy>(prototypeObject))
            {
                // Fetching prototype of proxy can invoke trap - which we don't want during the marshalling time.
                break;
            }
            prototype = prototypeObject->GetPrototype();
        }
    }

    void CrossSite::MarshalDynamicObjectAndPrototype(ScriptContext* scriptContext, DynamicObject * object)
    {
        MarshalDynamicObject(scriptContext, object);
        MarshalPrototypeChain(scriptContext, object);
    }

    Var CrossSite::MarshalFrameDisplay(ScriptContext* scriptContext, FrameDisplay *display)
    {
        TTD_XSITE_LOG(scriptContext, "MarshalFrameDisplay", nullptr);

        uint16 length = display->GetLength();
        FrameDisplay *newDisplay =
            RecyclerNewPlus(scriptContext->GetRecycler(), length * sizeof(Var), FrameDisplay, length);
        for (uint16 i = 0; i < length; i++)
        {
            Var value = display->GetItem(i);
            if (Js::VarIs<UnscopablesWrapperObject>(value))
            {
                // Here we are marshalling the wrappedObject and then ReWrapping th object in the new context.
                RecyclableObject* wrappedObject = Js::VarTo<UnscopablesWrapperObject>(value)->GetWrappedObject();
                ScriptContext* wrappedObjectScriptContext = wrappedObject->GetScriptContext();
                value = JavascriptOperators::ToUnscopablesWrapperObject(CrossSite::MarshalVar(scriptContext,
                  wrappedObject, wrappedObjectScriptContext), scriptContext);
            }
            else
            {
                value = CrossSite::MarshalVar(scriptContext, value);
            }
            newDisplay->SetItem(i, value);
        }

        return (Var)newDisplay;
    }

    // static
    Var CrossSite::MarshalVar(ScriptContext* scriptContext, Var value, ScriptContext* objectScriptContext)
    {
        if (scriptContext != objectScriptContext)
        {
            if (value == nullptr || Js::TaggedNumber::Is(value))
            {
                return value;
            }
            return MarshalVarInner(scriptContext, VarTo<RecyclableObject>(value), false);
        }
        return value;
    }

    // static
    Var CrossSite::MarshalVar(ScriptContext* scriptContext, Var value, bool fRequestWrapper)
    {
        // value might be null from disable implicit call
        if (value == nullptr || Js::TaggedNumber::Is(value))
        {
            return value;
        }
        Js::RecyclableObject* object =  UnsafeVarTo<RecyclableObject>(value);

        if (fRequestWrapper || scriptContext != object->GetScriptContext())
        {
            if (PHASE_TRACE1(Js::Phase::MarshalPhase))
            {
                Output::Print(_u("MarshalVar: %p (var sc: %p, request sc: %p, requestWrapper: %d)\n"), object, object->GetScriptContext(), scriptContext, fRequestWrapper);
            }

            // Do not allow marshaling if a callable object is being marshalled into a high privileged
            // script context.
            if (JavascriptConversion::IsCallable(object))
            {
                ScriptContext* objectScriptContext = object->GetScriptContext();
                if (scriptContext->GetPrivilegeLevel() < objectScriptContext->GetPrivilegeLevel())
                {
                    return scriptContext->GetLibrary()->GetUndefined();
                }
            }
            return MarshalVarInner(scriptContext, object, fRequestWrapper);
        }
        return value;
    }

    bool CrossSite::DoRequestWrapper(Js::RecyclableObject* object, bool fRequestWrapper)
    {
        return fRequestWrapper && VarIs<JavascriptFunction>(object) && VarTo<JavascriptFunction>(object)->IsExternalFunction();
    }

#if ENABLE_TTD
    void CrossSite::MarshalCrossSite_TTDInflate(DynamicObject* obj)
    {
        obj->MarshalCrossSite_TTDInflate();

        if(obj->GetTypeId() == TypeIds_Function)
        {
            AssertMsg(obj != obj->GetScriptContext()->GetLibrary()->GetDefaultAccessorFunction(), "default accessor marshalled -- I don't think this should ever happen as it is marshalled in a special case?");
            JavascriptFunction * function = VarTo<JavascriptFunction>(obj);

            //
            //TODO: what happens if the gaurd in marshal (MarshalDynamicObject) isn't true?
            //

            if(function->GetTypeHandler()->GetIsLocked())
            {
                function->GetLibrary()->SetCrossSiteForLockedFunctionType(function);
            }
            else
            {
                function->SetEntryPoint(function->GetScriptContext()->CurrentCrossSiteThunk);
            }
        }
    }
#endif

    Var CrossSite::MarshalVarInner(ScriptContext* scriptContext, __in Js::RecyclableObject* object, bool fRequestWrapper)
    {
        if (scriptContext == object->GetScriptContext())
        {
            if (DoRequestWrapper(object, fRequestWrapper))
            {
                // If we get here then we need to either wrap in the caller's type system or we need to return undefined.
                // VBScript will pass in the scriptContext (requestContext) from the JavascriptDispatch and this will be the
                // same as the object's script context and so we have to safely pretend this value doesn't exist.
                return scriptContext->GetLibrary()->GetUndefined();
            }
            return object;
        }

        AssertMsg(scriptContext->GetThreadContext() == object->GetScriptContext()->GetThreadContext(), "ScriptContexts should belong to same threadcontext for marshalling.");
        // In heapenum, we are traversing through the object graph to dump out the content of recyclable objects. The content
        // of the objects are duplicated to the heapenum result, and we are not storing/changing the object graph during heap enum.
        // We don't actually need to do cross site thunk here.
        if (scriptContext->GetRecycler()->IsHeapEnumInProgress())
        {
            return object;
        }

#if ENABLE_TTD
        if (scriptContext->IsTTDSnapshotOrInflateInProgress())
        {
            return object;
        }
#endif

        // Marshaling should not cause any re-entrancy.
        JS_REENTRANCY_LOCK(jsReentLock, scriptContext->GetThreadContext());

#if ENABLE_COPYONACCESS_ARRAY
        JavascriptLibrary::CheckAndConvertCopyOnAccessNativeIntArray<Var>(object);
#endif
        TypeId typeId = object->GetTypeId();
        AssertMsg(typeId != TypeIds_Enumerator, "enumerator shouldn't be marshalled here");

        // At the moment the mental model for UnscopablesWrapperObject Marshaling is this:
        // Are we trying to marshal a UnscopablesWrapperObject in the Frame Display? - then 1) unwrap in MarshalFrameDisplay,
        // 2) marshal the wrapped object, 3) Create a new UnscopablesWrapperObject in the current scriptContext and re-wrap.
        // We can avoid copying the UnscopablesWrapperObject because it has no properties and never should.
        // Thus creating a new UnscopablesWrapperObject per context in MarshalFrameDisplay should be kosher.
        // If it is not a FrameDisplay then we should not marshal. We can wrap cross context objects with a
        // UnscopablesWrapperObject in a different context. When we unwrap for property lookups and the wrapped object
        // is cross context, then we marshal the wrapped object into the current scriptContext, thus avoiding
        // the need to copy the UnscopablesWrapperObject itself. Thus We don't have to handle marshaling the UnscopablesWrapperObject
        // in non-FrameDisplay cases.
        AssertMsg(typeId != TypeIds_UnscopablesWrapperObject, "UnscopablesWrapperObject shouldn't be marshalled here");

        if (StaticType::Is(typeId))
        {
            TTD_XSITE_LOG(object->GetScriptContext(), "CloneToScriptContext", object);

            return object->CloneToScriptContext(scriptContext);
        }

        if (typeId == TypeIds_ModuleRoot)
        {
            RootObjectBase *moduleRoot = static_cast<RootObjectBase*>(object);
            HostObjectBase * hostObject = moduleRoot->GetHostObject();

            // When marshaling module root, all we need is the host object.
            // So, if the module root which is being marshaled has host object, marshal it.
            if (hostObject)
            {
                TTD_XSITE_LOG(object->GetScriptContext(), "hostObject", hostObject);

                Var hostDispatch = hostObject->GetHostDispatchVar();
                return CrossSite::MarshalVar(scriptContext, hostDispatch);
            }
        }

        if (typeId == TypeIds_Function)
        {
            if (object == object->GetScriptContext()->GetLibrary()->GetDefaultAccessorFunction() )
            {
                TTD_XSITE_LOG(object->GetScriptContext(), "DefaultAccessorFunction", object);

                return scriptContext->GetLibrary()->GetDefaultAccessorFunction();
            }

            if (DoRequestWrapper(object, fRequestWrapper))
            {
                TTD_XSITE_LOG(object->GetScriptContext(), "CreateWrappedExternalFunction", object);

                // Marshal as a cross-site thunk if necessary before re-wrapping in an external function thunk.
                MarshalVarInner(scriptContext, object, false);
                return scriptContext->GetLibrary()->CreateWrappedExternalFunction(static_cast<JavascriptExternalFunction*>(object));
            }
        }

        // We have an object marshaled, we need to keep track of the related script context
        // so optimization overrides can be updated as a group
        scriptContext->optimizationOverrides.Merge(&object->GetScriptContext()->optimizationOverrides);

        DynamicObject * dynamicObject = VarTo<DynamicObject>(object);
        if (!dynamicObject->IsExternal())
        {
            if (!dynamicObject->IsCrossSiteObject())
            {
                if (VarIs<JavascriptProxy>(dynamicObject))
                {
                    // We don't need to marshal the prototype chain in the case of Proxy. Otherwise we will go to the user code.
                    TTD_XSITE_LOG(object->GetScriptContext(), "MarshalDynamicObject", object);
                    MarshalDynamicObject(scriptContext, dynamicObject);
                }
                else
                {
                    TTD_XSITE_LOG(object->GetScriptContext(), "MarshalDynamicObjectAndPrototype", object);

                    MarshalDynamicObjectAndPrototype(scriptContext, dynamicObject);
                }
            }
        }
        else
        {
            MarshalPrototypeChain(scriptContext, dynamicObject);
            if (Js::JavascriptConversion::IsCallable(dynamicObject))
            {
                TTD_XSITE_LOG(object->GetScriptContext(), "MarshalToScriptContext", object);

                dynamicObject->MarshalToScriptContext(scriptContext);
            }
        }

        return dynamicObject;
    }

    bool CrossSite::IsThunk(JavascriptMethod thunk)
    {
#if defined(ENABLE_SCRIPT_PROFILING) || defined(ENABLE_SCRIPT_DEBUGGING)
        return (thunk == CrossSite::ProfileThunk || thunk == CrossSite::DefaultThunk);
#else
        return (thunk == CrossSite::DefaultThunk);
#endif
    }

#if defined(ENABLE_SCRIPT_PROFILING) || defined(ENABLE_SCRIPT_DEBUGGING)
    Var CrossSite::ProfileThunk(RecyclableObject* callable, CallInfo callInfo, ...)
    {
        JavascriptFunction* function = VarTo<JavascriptFunction>(callable);
        Assert(function->GetTypeId() == TypeIds_Function);
        Assert(function->GetEntryPoint() == CrossSite::ProfileThunk);
        RUNTIME_ARGUMENTS(args, callInfo);
        ScriptContext * scriptContext = function->GetScriptContext();
        // It is not safe to access the function body if the script context is not alive.
        scriptContext->VerifyAliveWithHostContext(!function->IsExternal(),
            scriptContext->GetThreadContext()->GetPreviousHostScriptContext());

        JavascriptMethod entryPoint;
        FunctionInfo *funcInfo = function->GetFunctionInfo();

        TTD_XSITE_LOG(callable->GetScriptContext(), "DefaultOrProfileThunk", callable);

#ifdef ENABLE_WASM
        if (VarIs<WasmScriptFunction>(function))
        {
            entryPoint = Js::AsmJsExternalEntryPoint;
        } else
#endif
        if (funcInfo->HasBody())
        {
#if ENABLE_DEBUG_CONFIG_OPTIONS
            char16 debugStringBuffer[MAX_FUNCTION_BODY_DEBUG_STRING_SIZE];
#endif
            entryPoint = VarTo<ScriptFunction>(function)->GetEntryPointInfo()->jsMethod;
            if (funcInfo->IsDeferred() && scriptContext->IsProfiling())
            {
                // if the current entrypoint is deferred parse we need to update it appropriately for the profiler mode.
                entryPoint = Js::ScriptContext::GetProfileModeThunk(entryPoint);
            }
            OUTPUT_TRACE(Js::ScriptProfilerPhase, _u("CrossSite::ProfileThunk FunctionNumber : %s, Entrypoint : 0x%08X\n"), funcInfo->GetFunctionProxy()->GetDebugNumberSet(debugStringBuffer), entryPoint);
        }
        else
        {
            entryPoint = ProfileEntryThunk;
        }


        return CommonThunk(function, entryPoint, args);
    }
#endif

    Var CrossSite::DefaultThunk(RecyclableObject* callable, CallInfo callInfo, ...)
    {
        JavascriptFunction* function = VarTo<JavascriptFunction>(callable);
        Assert(function->GetTypeId() == TypeIds_Function);
        Assert(function->GetEntryPoint() == CrossSite::DefaultThunk);
        RUNTIME_ARGUMENTS(args, callInfo);

        // It is not safe to access the function body if the script context is not alive.
        function->GetScriptContext()->VerifyAliveWithHostContext(!function->IsExternal(),
            ThreadContext::GetContextForCurrentThread()->GetPreviousHostScriptContext());

        JavascriptMethod entryPoint;
        FunctionInfo *funcInfo = function->GetFunctionInfo();

        TTD_XSITE_LOG(callable->GetScriptContext(), "DefaultOrProfileThunk", callable);

        if (funcInfo->HasBody())
        {
#ifdef ASMJS_PLAT
            if (funcInfo->GetFunctionProxy()->IsFunctionBody() &&
                funcInfo->GetFunctionBody()->GetIsAsmJsFunction())
            {
                entryPoint = Js::AsmJsExternalEntryPoint;
            }
            else
#endif
            {
                entryPoint = VarTo<ScriptFunction>(function)->GetEntryPointInfo()->jsMethod;
            }
        }
        else
        {
            entryPoint = funcInfo->GetOriginalEntryPoint();
        }
        return CommonThunk(function, entryPoint, args);
    }

    Var CrossSite::CrossSiteProxyCallTrap(RecyclableObject* function, CallInfo callInfo, ...)
    {
        RUNTIME_ARGUMENTS(args, callInfo);
        Assert(VarIs<JavascriptProxy>(function));

        return CrossSite::CommonThunk(function, JavascriptProxy::FunctionCallTrap, args);
    }

    Var CrossSite::CommonThunk(RecyclableObject* recyclableObject, JavascriptMethod entryPoint, Arguments args)
    {
        DynamicObject* function = VarTo<DynamicObject>(recyclableObject);

        FunctionInfo * functionInfo = (VarIs<JavascriptFunction>(function) ? VarTo<JavascriptFunction>(function)->GetFunctionInfo() : nullptr);
        AutoDisableRedeferral autoDisableRedeferral(functionInfo);

        ScriptContext* targetScriptContext = function->GetScriptContext();
        Assert(!targetScriptContext->IsClosed());
        Assert(function->IsExternal() || function->IsCrossSiteObject());
        Assert(targetScriptContext->GetThreadContext()->IsScriptActive());

        HostScriptContext* calleeHostScriptContext = targetScriptContext->GetHostScriptContext();
        HostScriptContext* callerHostScriptContext = targetScriptContext->GetThreadContext()->GetPreviousHostScriptContext();

        if (callerHostScriptContext == calleeHostScriptContext || (callerHostScriptContext == nullptr && !calleeHostScriptContext->HasCaller()))
        {
            BEGIN_SAFE_REENTRANT_CALL(targetScriptContext->GetThreadContext())
            {
                return JavascriptFunction::CallFunction<true>(function, entryPoint, args, true /*useLargeArgCount*/);
            }
            END_SAFE_REENTRANT_CALL
        }

#if DBG_DUMP || defined(PROFILE_EXEC) || defined(PROFILE_MEM)
        calleeHostScriptContext->EnsureParentInfo(callerHostScriptContext->GetScriptContext());
#endif

        TTD_XSITE_LOG(recyclableObject->GetScriptContext(), "CommonThunk -- Pass Through", recyclableObject);

        uint i = 0;
        if (args.Values[0] == nullptr)
        {
            i = 1;
            Assert(args.IsNewCall());
            Assert(VarIs<JavascriptProxy>(function) || (VarIs<JavascriptFunction>(function) && VarTo<JavascriptFunction>(function)->GetFunctionInfo()->GetAttributes() & FunctionInfo::SkipDefaultNewObject));
        }
        uint count = args.Info.Count;
        for (; i < count; i++)
        {
            args.Values[i] = CrossSite::MarshalVar(targetScriptContext, args.Values[i]);
        }
        if (args.HasNewTarget())
        {
            // Last value is new.target
            args.Values[count] = CrossSite::MarshalVar(targetScriptContext, args.GetNewTarget());
        }
        else if (args.HasExtraArg())
        {
            // The final eval arg is a frame display that needs to be marshaled specially.
            args.Values[count] = CrossSite::MarshalFrameDisplay(targetScriptContext, args.GetFrameDisplay());
        }


#if ENABLE_NATIVE_CODEGEN
        CheckCodeGenFunction checkCodeGenFunction = GetCheckCodeGenFunction(entryPoint);
        if (checkCodeGenFunction != nullptr)
        {
            ScriptFunction* callFunc = VarTo<ScriptFunction>(function);
            entryPoint = checkCodeGenFunction(callFunc);
            Assert(CrossSite::IsThunk(function->GetEntryPoint()));
        }
#endif

        // We need to setup the caller chain when we go across script site boundary. Property access
        // is OK, and we need to let host know who the caller is when a call is from another script site.
        // CrossSiteObject is the natural place but it is in the target site. We build up the site
        // chain through PushDispatchExCaller/PopDispatchExCaller, and we call SetCaller in the target site
        // to indicate who the caller is. We first need to get the site from the previously pushed site
        // and set that as the caller for current call, and push a new DispatchExCaller for future calls
        // off this site. GetDispatchExCaller and ReleaseDispatchExCaller is used to get the current caller.
        // currentDispatchExCaller is cached to avoid multiple allocations.
        IUnknown* sourceCaller = nullptr, *previousSourceCaller = nullptr;
        HRESULT hr = NOERROR;
        Var result = nullptr;
        BOOL wasDispatchExCallerPushed = FALSE, wasCallerSet = FALSE;

        TryFinally([&]()
        {
            hr = callerHostScriptContext->GetDispatchExCaller((void**)&sourceCaller);

            if (SUCCEEDED(hr))
            {
                hr = calleeHostScriptContext->SetCaller((IUnknown*)sourceCaller, (IUnknown**)&previousSourceCaller);
            }

            if (SUCCEEDED(hr))
            {
                wasCallerSet = TRUE;
                hr = calleeHostScriptContext->PushHostScriptContext();
            }
            if (FAILED(hr))
            {
                // CONSIDER: Should this be callerScriptContext if we failed?
                JavascriptError::MapAndThrowError(targetScriptContext, hr);
            }
            wasDispatchExCallerPushed = TRUE;

            BEGIN_SAFE_REENTRANT_CALL(targetScriptContext->GetThreadContext())
            {
                result = JavascriptFunction::CallFunction<true>(function, entryPoint, args, true /*useLargeArgCount*/);
            }
            END_SAFE_REENTRANT_CALL
            ScriptContext* callerScriptContext = callerHostScriptContext->GetScriptContext();
            result = CrossSite::MarshalVar(callerScriptContext, result);
        },
        [&](bool hasException)
        {
            if (sourceCaller != nullptr)
            {
                callerHostScriptContext->ReleaseDispatchExCaller(sourceCaller);
            }
            IUnknown* originalCaller = nullptr;
            if (wasDispatchExCallerPushed)
            {
                calleeHostScriptContext->PopHostScriptContext();
            }
            if (wasCallerSet)
            {
                calleeHostScriptContext->SetCaller(previousSourceCaller, &originalCaller);
                if (previousSourceCaller)
                {
                    previousSourceCaller->Release();
                }
                if (originalCaller)
                {
                    originalCaller->Release();
                }
            }
        });
        Assert(result != nullptr);
        return result;
    }

    // For prototype chain to install cross-site thunk.
    // When we change prototype using __proto__, those prototypes might not have cross-site thunks
    // installed even though the CEO is accessed from a different context. During ChangePrototype time
    // we don't really know where the requestContext is.
    // Force installing cross-site thunk for all prototype changes. It's a relatively less frequently used
    // scenario.
    void CrossSite::ForceCrossSiteThunkOnPrototypeChain(RecyclableObject* object)
    {
        if (TaggedNumber::Is(object))
        {
            return;
        }
        while (DynamicType::Is(object->GetTypeId()) && !VarIs<JavascriptProxy>(object))
        {
            DynamicObject* dynamicObject = UnsafeVarTo<DynamicObject>(object);
            if (!dynamicObject->IsCrossSiteObject() && !dynamicObject->IsExternal())
            {
                // force to install cross-site thunk on prototype objects.
                dynamicObject->MarshalToScriptContext(nullptr);
            }
            object = object->GetPrototype();
        }
        return;

    }
};
