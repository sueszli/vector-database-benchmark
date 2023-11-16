//-------------------------------------------------------------------------------------------------------
// Copyright (C) Microsoft Corporation and contributors. All rights reserved.
// Licensed under the MIT license. See LICENSE.txt file in the project root for full license information.
// Portions of this file are copyright 2014 Mozilla Foundation, available under the Apache 2.0 license.
//-------------------------------------------------------------------------------------------------------

//-------------------------------------------------------------------------------------------------------
// Copyright 2014 Mozilla Foundation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http ://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//-------------------------------------------------------------------------------------------------------

#include "RuntimeLanguagePch.h"
#ifdef ASMJS_PLAT
#include "ByteCode/Symbol.h"
#include "ByteCode/FuncInfo.h"
#include "ByteCode/ByteCodeWriter.h"
#include "ByteCode/ByteCodeGenerator.h"

namespace Js
{
    bool
    AsmJSCompiler::CheckIdentifier(AsmJsModuleCompiler &m, ParseNode *usepn, PropertyName name)
    {
        if (name == m.GetParser()->names()->arguments || name == m.GetParser()->names()->eval)
        {
            return m.FailName(usepn, _u("'%s' is not an allowed identifier"), name);
        }
        return true;
    }

    bool
    AsmJSCompiler::CheckModuleLevelName(AsmJsModuleCompiler &m, ParseNode *usepn, PropertyName name)
    {
        if (!CheckIdentifier(m, usepn, name))
        {
            return false;
        }
        if (name == m.GetModuleFunctionName())
        {
            return m.FailName(usepn, _u("duplicate name '%s' not allowed"), name);
        }
        //Check for all the duplicates here.
        return true;
    }


    bool
    AsmJSCompiler::CheckFunctionHead(AsmJsModuleCompiler &m, ParseNode *fn, bool isGlobal /*= true*/)
    {
        ParseNodeFnc * fnc = fn->AsParseNodeFnc();

        if (fnc->HasNonSimpleParameterList())
        {
            return m.Fail(fn, _u("default, rest & destructuring args not allowed"));
        }

        if (fnc->IsStaticMember())
        {
            return m.Fail(fn, _u("static functions are not allowed"));
        }

        if (fnc->IsGenerator())
        {
            return m.Fail(fn, _u("generator functions are not allowed"));
        }

        if (fnc->IsAsync())
        {
            return m.Fail(fn, _u("async functions are not allowed"));
        }

        if (fnc->IsLambda())
        {
            return m.Fail(fn, _u("lambda functions are not allowed"));
        }

        if (!isGlobal && fnc->nestedCount != 0)
        {
            return m.Fail(fn, _u("closure functions are not allowed"));
        }

        if (!fnc->IsAsmJsAllowed())
        {
            return m.Fail(fn, _u("invalid function flags detected"));
        }

        return true;
    }

    bool AsmJSCompiler::CheckTypeAnnotation( AsmJsModuleCompiler &m, ParseNode *coercionNode, AsmJSCoercion *coercion,
        ParseNode **coercedExpr /*= nullptr */)
    {
        switch( coercionNode->nop )
        {
        case knopRsh:
        case knopLsh:
        case knopXor:
        case knopAnd:
        case knopOr: {
            ParseNode *rhs = ParserWrapper::GetBinaryRight( coercionNode );
            *coercion = AsmJS_ToInt32;
            if( coercedExpr )
            {

                if( rhs->nop == knopInt && rhs->AsParseNodeInt()->lw == 0 )
                {
                    if( rhs->nop == knopAnd )
                    {
                        // X & 0 == 0;
                        *coercedExpr = rhs;
                    }
                    else
                    {
                        // (X|0) == (X^0) == (X<<0) == (X>>0) == X
                        *coercedExpr = ParserWrapper::GetBinaryLeft( coercionNode );
                    }
                }
                else
                {
                    *coercedExpr = coercionNode;
                }
            }
            return true;
        }
        case knopPos: {
            *coercion = AsmJS_ToNumber;
            if( coercedExpr )
            {
                *coercedExpr = ParserWrapper::GetUnaryNode( coercionNode );
            }
            return true;
        }
        case knopCall: {
            ParseNode* target;
            AsmJsFunctionDeclaration* sym;

            target = coercionNode->AsParseNodeCall()->pnodeTarget;

            if (!target || target->nop != knopName)
            {
                return m.Fail(coercionNode, _u("Call must be of the form id(...)"));
            }

            *coercion = AsmJS_FRound;
            sym = m.LookupFunction(target->name());

            if (!AsmJsMathFunction::IsFround(sym))
            {
                return m.Fail( coercionNode, _u("call must be to fround coercion") );
            }
            if( coercedExpr )
            {
                *coercedExpr = coercionNode->AsParseNodeCall()->pnodeArgs;
            }
            return true;
        }
        case knopInt:{
            *coercion = AsmJS_ToInt32;
            if( coercedExpr )
            {
                *coercedExpr = coercionNode;
            }
            return true;
        }
        case knopFlt:{
            if (ParserWrapper::IsMinInt(coercionNode))
            {
                *coercion = AsmJS_ToInt32;
            }
            else if (coercionNode->AsParseNodeFloat()->maybeInt)
            {
                return m.Fail(coercionNode, _u("Integer literal in return must be in range [-2^31, 2^31)"));
            }
            else
            {
                *coercion = AsmJS_ToNumber;
            }
            if( coercedExpr )
            {
                *coercedExpr = coercionNode ;
            }
            return true;
        }
        case knopName:{

            // in this case we are returning a constant var from the global scope
            AsmJsSymbol * constSymSource = m.LookupIdentifier(coercionNode->name());

            if (!constSymSource)
            {
                return m.Fail(coercionNode, _u("Identifier not globally declared"));
            }

            if (!AsmJsVar::Is(constSymSource) || constSymSource->isMutable())
            {
                return m.Fail(coercionNode, _u("Unannotated variables must be constant"));
            }
            AsmJsVar * constSrc = AsmJsVar::FromSymbol(constSymSource);

            if (constSrc->GetType().isSigned())
            {
                *coercion = AsmJS_ToInt32;
            }
            else if (constSrc->GetType().isDouble())
            {
                *coercion = AsmJS_ToNumber;
            }
            else
            {
                Assert(constSrc->GetType().isFloat());
                *coercion = AsmJS_FRound;
            }
            if (coercedExpr)
            {
                *coercedExpr = coercionNode;
            }
            return true;
        }
        default:;
        }

        return m.Fail( coercionNode, _u("must be of the form +x, fround(x) or x|0") );
    }

    bool
    AsmJSCompiler::CheckModuleArgument(AsmJsModuleCompiler &m, ParseNode *arg, PropertyName *name, AsmJsModuleArg::ArgType type)
    {
        if (!ParserWrapper::IsDefinition(arg))
        {
            return m.Fail(arg, _u("duplicate argument name not allowed"));
        }

        if (!CheckIdentifier(m, arg, arg->name()))
        {
            return false;
        }
        *name = arg->name();

        m.GetByteCodeGenerator()->AssignPropertyId(*name);

        AsmJsModuleArg * moduleArg = Anew(m.GetAllocator(), AsmJsModuleArg, *name, type);

        if (!m.DefineIdentifier(*name, moduleArg))
        {
            return m.Fail(arg, _u("duplicate argument name not allowed"));
        }

        if (!CheckModuleLevelName(m, arg, *name))
        {
            return false;
        }

        return true;
    }

    bool
    AsmJSCompiler::CheckModuleArguments(AsmJsModuleCompiler &m, ParseNode *fn)
    {
        ArgSlot numFormals = 0;

        ParseNode *arg1 = ParserWrapper::FunctionArgsList( fn, numFormals );
        ParseNode *arg2 = arg1 ? ParserWrapper::NextVar( arg1 ) : nullptr;
        ParseNode *arg3 = arg2 ? ParserWrapper::NextVar( arg2 ) : nullptr;

        if (numFormals > 3)
        {
            return m.Fail(fn, _u("asm.js modules takes at most 3 argument"));
        }

        PropertyName arg1Name = nullptr;
        if (numFormals >= 1 && !CheckModuleArgument(m, arg1, &arg1Name, AsmJsModuleArg::ArgType::StdLib))
        {
            return false;
        }

        m.InitStdLibArgName(arg1Name);

        PropertyName arg2Name = nullptr;
        if (numFormals >= 2 && !CheckModuleArgument(m, arg2, &arg2Name, AsmJsModuleArg::ArgType::Import))
        {
            return false;
        }
        m.InitForeignArgName(arg2Name);

        PropertyName arg3Name = nullptr;
        if (numFormals >= 3 && !CheckModuleArgument(m, arg3, &arg3Name, AsmJsModuleArg::ArgType::Heap))
        {
            return false;
        }
        m.InitBufferArgName(arg3Name);

        return true;
    }

    bool AsmJSCompiler::CheckGlobalVariableImportExpr( AsmJsModuleCompiler &m, PropertyName varName, AsmJSCoercion coercion, ParseNode *coercedExpr )
    {
        if( !ParserWrapper::IsDotMember(coercedExpr) )
        {
            return m.FailName( coercedExpr, _u("invalid import expression for global '%s'"), varName );
        }
        ParseNode *base = ParserWrapper::DotBase(coercedExpr);
        PropertyName field = ParserWrapper::DotMember(coercedExpr);

        PropertyName importName = m.GetForeignArgName();
        if (!importName || !field)
        {
            return m.Fail(coercedExpr, _u("cannot import without an asm.js foreign parameter"));
        }
        m.GetByteCodeGenerator()->AssignPropertyId(field);
        if ((base->name() != importName))
        {
            return m.FailName(coercedExpr, _u("base of import expression must be '%s'"), importName);
        }
        return m.AddGlobalVarImport(varName, field, coercion);
    }

    bool AsmJSCompiler::CheckGlobalVariableInitImport( AsmJsModuleCompiler &m, PropertyName varName, ParseNode *initNode, bool isMutable /*= true*/)
    {
        AsmJSCoercion coercion;
        ParseNode *coercedExpr = nullptr;
        if( !CheckTypeAnnotation( m, initNode, &coercion, &coercedExpr ) )
        {
            return false;
        }
        if ((ParserWrapper::IsFroundNumericLiteral(coercedExpr)) && coercion == AsmJSCoercion::AsmJS_FRound)
        {
            return m.AddNumericVar(varName, coercedExpr, true, isMutable);
        }
        return CheckGlobalVariableImportExpr( m, varName, coercion, coercedExpr );
    }

    bool AsmJSCompiler::CheckNewArrayView( AsmJsModuleCompiler &m, PropertyName varName, ParseNode *newExpr )
    {
        Assert( newExpr->nop == knopNew );
        m.SetUsesHeapBuffer(true);

        ParseNode *ctorExpr = newExpr->AsParseNodeCall()->pnodeTarget;
        ArrayBufferView::ViewType type;
        if( ParserWrapper::IsDotMember(ctorExpr) )
        {
            ParseNode *base = ParserWrapper::DotBase(ctorExpr);

            PropertyName globalName = m.GetStdLibArgName();
            if (!globalName)
            {
                return m.Fail(base, _u("cannot create array view without an asm.js global parameter"));
            }

            if (!ParserWrapper::IsNameDeclaration(base) || base->name() != globalName)
            {
                return m.FailName(base, _u("expecting '%s.*Array"), globalName);
            }
            PropertyName fieldName = ParserWrapper::DotMember(ctorExpr);
            if (!fieldName)
            {
                return m.FailName(ctorExpr, _u("Failed to define array view to var %s"), varName);
            }
            PropertyId field = fieldName->GetPropertyId();

            switch (field)
            {
            case PropertyIds::Int8Array:
                type = ArrayBufferView::TYPE_INT8;
                m.AddArrayBuiltinUse(AsmJSTypedArrayBuiltin_Int8Array);
                break;
            case PropertyIds::Uint8Array:
                type = ArrayBufferView::TYPE_UINT8;
                m.AddArrayBuiltinUse(AsmJSTypedArrayBuiltin_Uint8Array);
                break;
            case PropertyIds::Int16Array:
                type = ArrayBufferView::TYPE_INT16;
                m.AddArrayBuiltinUse(AsmJSTypedArrayBuiltin_Int16Array);
                break;
            case PropertyIds::Uint16Array:
                type = ArrayBufferView::TYPE_UINT16;
                m.AddArrayBuiltinUse(AsmJSTypedArrayBuiltin_Uint16Array);
                break;
            case PropertyIds::Int32Array:
                type = ArrayBufferView::TYPE_INT32;
                m.AddArrayBuiltinUse(AsmJSTypedArrayBuiltin_Int32Array);
                break;
            case PropertyIds::Uint32Array:
                type = ArrayBufferView::TYPE_UINT32;
                m.AddArrayBuiltinUse(AsmJSTypedArrayBuiltin_Uint32Array);
                break;
            case PropertyIds::Float32Array:
                type = ArrayBufferView::TYPE_FLOAT32;
                m.AddArrayBuiltinUse(AsmJSTypedArrayBuiltin_Float32Array);
                break;
            case PropertyIds::Float64Array:
                type = ArrayBufferView::TYPE_FLOAT64;
                m.AddArrayBuiltinUse(AsmJSTypedArrayBuiltin_Float64Array);
                break;
            default:
                return m.Fail(ctorExpr, _u("could not match typed array name"));
                break;
            }
        }
        else if (ctorExpr->nop == knopName)
        {
            AsmJsSymbol * buffFunc = m.LookupIdentifier(ctorExpr->name());

            if (!AsmJsTypedArrayFunction::Is(buffFunc))
            {
                return m.Fail(ctorExpr, _u("invalid 'new' import"));
            }
            type = AsmJsTypedArrayFunction::FromSymbol(buffFunc)->GetViewType();
            if (type == ArrayBufferView::TYPE_COUNT)
            {
                return m.Fail(ctorExpr, _u("could not match typed array name"));
            }
        }
        else
        {
            return m.Fail(newExpr, _u("invalid 'new' import"));
        }

        ParseNode *bufArg = newExpr->AsParseNodeCall()->pnodeArgs;
        if( !bufArg || !ParserWrapper::IsNameDeclaration( bufArg ) )
        {
            return m.Fail( ctorExpr, _u("array view constructor takes exactly one argument") );
        }

        PropertyName bufferName = m.GetBufferArgName();
        if( !bufferName )
        {
            return m.Fail( bufArg, _u("cannot create array view without an asm.js heap parameter") );
        }

        if( bufferName != bufArg->name() )
        {
            return m.FailName( bufArg, _u("argument to array view constructor must be '%s'"), bufferName );
        }


        if( !m.AddArrayView( varName, type ) )
        {
            return m.FailName( ctorExpr, _u("Failed to define array view to var %s"), varName );
        }
        return true;
    }

    bool
    AsmJSCompiler::CheckGlobalDotImport(AsmJsModuleCompiler &m, PropertyName varName, ParseNode *initNode)
    {
        ParseNode *base = ParserWrapper::DotBase(initNode);
        PropertyName field = ParserWrapper::DotMember(initNode);
        if( !field )
        {
            return m.Fail( initNode, _u("Global import must be in the form c.x where c is stdlib or foreign and x is a string literal") );
        }
        m.GetByteCodeGenerator()->AssignPropertyId(field);
        PropertyName lib = nullptr;
        if (ParserWrapper::IsDotMember(base))
        {
            lib = ParserWrapper::DotMember(base);
            base = ParserWrapper::DotBase(base);
            if (!lib || lib->GetPropertyId() != PropertyIds::Math)
            {
                return m.FailName(initNode, _u("'%s' should be Math, as in global.Math.xxxx"), field);
            }
        }

        if( ParserWrapper::IsNameDeclaration(base) && base->name() == m.GetStdLibArgName() )
        {
            // global.Math.xxx
            MathBuiltin mathBuiltin;
            if (m.LookupStandardLibraryMathName(field, &mathBuiltin))
            {
                switch (mathBuiltin.kind)
                {
                case MathBuiltin::Function:{
                    auto func = mathBuiltin.u.func;
                    if (func->GetName() != nullptr)
                    {
                        OutputMessage(m.GetScriptContext(), DEIT_ASMJS_FAILED, _u("Warning: Math Builtin already defined for var %s"), func->GetName()->Psz());
                    }
                    func->SetName(varName);
                    if (!m.DefineIdentifier(varName, func))
                    {
                        return m.FailName(initNode, _u("Failed to define math builtin function to var %s"), varName);
                    }
                    m.AddMathBuiltinUse(func->GetMathBuiltInFunction());
                }
                break;
                case MathBuiltin::Constant:
                    if (!m.AddNumericConst(varName, mathBuiltin.u.cst))
                    {
                        return m.FailName(initNode, _u("Failed to define math constant to var %s"), varName);
                    }
                    m.AddMathBuiltinUse(mathBuiltin.mathLibFunctionName);
                    break;
                default:
                    Assume(UNREACHED);
                }
                return true;
            }

            TypedArrayBuiltin arrayBuiltin;
            if (m.LookupStandardLibraryArrayName(field, &arrayBuiltin))
            {
                if (arrayBuiltin.mFunc->GetName() != nullptr)
                {
                    OutputMessage(m.GetScriptContext(), DEIT_ASMJS_FAILED, _u("Warning: Typed array builtin already defined for var %s"), arrayBuiltin.mFunc->GetName()->Psz());
                }
                arrayBuiltin.mFunc->SetName(varName);
                if (!m.DefineIdentifier(varName, arrayBuiltin.mFunc))
                {
                    return m.FailName(initNode, _u("Failed to define typed array builtin function to var %s"), varName);
                }
                m.AddArrayBuiltinUse(arrayBuiltin.mFunc->GetArrayBuiltInFunction());
                return true;
            }

            return m.FailName(initNode, _u("'%s' is not a standard Math builtin"), field);
        }
        else if( ParserWrapper::IsNameDeclaration(base) && base->name() == m.GetForeignArgName() )
        {
            // foreign import
            return m.AddModuleFunctionImport( varName, field );
        }

        return m.Fail(initNode, _u("expecting c.y where c is either the global or foreign parameter"));
    }

    bool
    AsmJSCompiler::CheckModuleGlobal(AsmJsModuleCompiler &m, ParseNode *var)
    {
        Assert(var->nop == knopVarDecl || var->nop == knopConstDecl);

        bool isMutable = var->nop != knopConstDecl;
        PropertyName name = var->name();

        m.GetByteCodeGenerator()->AssignPropertyId(name);
        if (m.LookupIdentifier(name))
        {
            return m.FailName(var, _u("import variable %s names must be unique"), name);
        }

        if (!CheckModuleLevelName(m, var, name))
        {
            return false;
        }

        if (!var->AsParseNodeVar()->pnodeInit)
        {
            return m.Fail(var, _u("module import needs initializer"));
        }

        ParseNode *initNode = var->AsParseNodeVar()->pnodeInit;


        if( ParserWrapper::IsNumericLiteral( initNode ) )
        {
            if (m.AddNumericVar(name, initNode, false, isMutable))
            {
                return true;
            }
            else
            {
                return m.FailName(var, _u("Failed to declare numeric var %s"), name);
            }
        }

        if (initNode->nop == knopOr || initNode->nop == knopPos || initNode->nop == knopCall)
        {
           return CheckGlobalVariableInitImport(m, name, initNode, isMutable );
        }

        if( initNode->nop == knopNew )
        {
           return CheckNewArrayView(m, var->name(), initNode);
        }

        if (ParserWrapper::IsDotMember(initNode))
        {
            return CheckGlobalDotImport(m, name, initNode);
        }


        return m.Fail( initNode, _u("Failed to recognize global variable") );
    }

    bool
    AsmJSCompiler::CheckModuleGlobals(AsmJsModuleCompiler &m)
    {
        ParseNode *varStmts;
        if( !ParserWrapper::ParseVarOrConstStatement( m.GetCurrentParserNode(), &varStmts ) )
        {
            return false;
        }

        if (!varStmts)
        {
            return true;
        }
        while (varStmts->nop == knopList)
        {
            ParseNode * pnode = ParserWrapper::GetBinaryLeft(varStmts);
            while (pnode && pnode->nop != knopEndCode)
            {
                ParseNode * decl;
                if (pnode->nop == knopList)
                {
                    decl = ParserWrapper::GetBinaryLeft(pnode);
                    pnode = ParserWrapper::GetBinaryRight(pnode);
                }
                else
                {
                    decl = pnode;
                    pnode = nullptr;
                }

                if (decl->nop == knopFncDecl)
                {
                    goto varDeclEnd;
                }
                else if (decl->nop != knopConstDecl && decl->nop != knopVarDecl)
                {
                    goto varDeclEnd;
                }

                if (decl->AsParseNodeVar()->pnodeInit && decl->AsParseNodeVar()->pnodeInit->nop == knopArray)
                {
                    // Assume we reached func tables
                    goto varDeclEnd;
                }

                if (!CheckModuleGlobal(m, decl))
                {
                    return false;
                }
            }

            if (ParserWrapper::GetBinaryRight(varStmts)->nop == knopEndCode)
            {
                // this is an error condition, but CheckFunctionsSequential will figure it out
                goto varDeclEnd;
            }
            varStmts = ParserWrapper::GetBinaryRight(varStmts);
        }
varDeclEnd:
        // we will collect information on the function tables now and come back to the functions themselves afterwards,
        // because function table information is used when processing function bodies
        ParseNode * fnNodes = varStmts;

        while (fnNodes->nop != knopEndCode && ParserWrapper::GetBinaryLeft(fnNodes)->nop == knopFncDecl)
        {
            fnNodes = ParserWrapper::GetBinaryRight(fnNodes);
        }

        if (fnNodes->nop == knopEndCode)
        {
            // if there are no function tables, we can just initialize count to 0
            m.SetFuncPtrTableCount(0);
        }
        else
        {
            m.SetCurrentParseNode(fnNodes);
            if (!CheckFunctionTables(m))
            {
                return false;
            }
        }
        // this will move us back to the beginning of the function declarations
        m.SetCurrentParseNode(varStmts);
        return true;
    }


    bool AsmJSCompiler::CheckFunction( AsmJsModuleCompiler &m, ParseNodeFnc * fncNode )
    {
        Assert( fncNode->nop == knopFncDecl );

        if( PHASE_TRACE1( Js::ByteCodePhase ) )
        {
            Output::Print( _u("  Checking Asm function: %s\n"), fncNode->funcInfo->name);
        }

        if( !CheckFunctionHead( m, fncNode, false ) )
        {
            return false;
        }

        AsmJsFunc* func = m.CreateNewFunctionEntry(fncNode);
        if (!func)
        {
            return m.Fail(fncNode, _u("      Error creating function entry"));
        }
        return true;
    }

    bool AsmJSCompiler::CheckFunctionsSequential( AsmJsModuleCompiler &m )
    {
        AsmJSParser& list = m.GetCurrentParserNode();
        Assert( list->nop == knopList );


        ParseNode* pnode = ParserWrapper::GetBinaryLeft(list);

        while (pnode->nop == knopFncDecl)
        {
            if( !CheckFunction( m, pnode->AsParseNodeFnc() ) )
            {
                return false;
            }

            if(ParserWrapper::GetBinaryRight(list)->nop == knopEndCode)
            {
                break;
            }
            list = ParserWrapper::GetBinaryRight(list);
            pnode = ParserWrapper::GetBinaryLeft(list);
        }

        m.SetCurrentParseNode( list );

        return true;
    }

    bool AsmJSCompiler::CheckFunctionTables(AsmJsModuleCompiler &m)
    {
        AsmJSParser& list = m.GetCurrentParserNode();
        Assert(list->nop == knopList);

        int32 funcPtrTableCount = 0;
        while (list->nop != knopEndCode)
        {
            ParseNode * varStmt = ParserWrapper::GetBinaryLeft(list);
            if (varStmt->nop != knopConstDecl && varStmt->nop != knopVarDecl)
            {
                break;
            }
            if (!varStmt->AsParseNodeVar()->pnodeInit || varStmt->AsParseNodeVar()->pnodeInit->nop != knopArray)
            {
                break;
            }
            const uint tableSize = varStmt->AsParseNodeVar()->pnodeInit->AsParseNodeArrLit()->count;
            if (!::Math::IsPow2(tableSize))
            {
                return m.FailName(varStmt, _u("Function table [%s] size must be a power of 2"), varStmt->name());
            }
            if (!m.AddFunctionTable(varStmt->name(), tableSize))
            {
                return m.FailName(varStmt, _u("Unable to create new function table %s"), varStmt->name());
            }

            AsmJsFunctionTable* ftable = (AsmJsFunctionTable*)m.LookupIdentifier(varStmt->name());
            Assert(ftable);
            ParseNode* pnode = varStmt->AsParseNodeVar()->pnodeInit->AsParseNodeArrLit()->pnode1;
            if (pnode->nop == knopList)
            {
                pnode = ParserWrapper::GetBinaryLeft(pnode);
            }
            if (!ParserWrapper::IsNameDeclaration(pnode))
            {
                return m.FailName(pnode, _u("Invalid element in function table %s"), varStmt->name());
            }
            ++funcPtrTableCount;
            list = ParserWrapper::GetBinaryRight(list);
        }

        m.SetFuncPtrTableCount(funcPtrTableCount);

        m.SetCurrentParseNode(list);
        return true;
    }

    bool AsmJSCompiler::CheckModuleReturn( AsmJsModuleCompiler& m )
    {
        ParseNode* endStmt = m.GetCurrentParserNode();

        if (endStmt->nop != knopList)
        {
            return m.Fail(endStmt, _u("Module must have a return"));
        }

        ParseNode* node = ParserWrapper::GetBinaryLeft( endStmt );
        ParseNode* endNode = ParserWrapper::GetBinaryRight( endStmt );

        if( node->nop != knopReturn || endNode->nop != knopEndCode )
        {
            return m.Fail( node, _u("Only expression after table functions must be a return") );
        }

        ParseNode* objNode = node->AsParseNodeReturn()->pnodeExpr;
        if ( !objNode )
        {
            return m.Fail( node, _u( "Module return must be an object or 1 function" ) );
        }

        if( objNode->nop != knopObject )
        {
            if( ParserWrapper::IsNameDeclaration( objNode ) )
            {
                PropertyName name = objNode->name();
                AsmJsSymbol* sym = m.LookupIdentifier( name );
                if( !sym )
                {
                    return m.FailName( node, _u("Symbol %s not recognized inside module"), name );
                }

                if (!AsmJsFunc::Is(sym))
                {
                    return m.FailName( node, _u("Symbol %s can only be a function of the module"), name );
                }

                AsmJsFunc* func = AsmJsFunc::FromSymbol(sym);
                if( !m.SetExportFunc( func ) )
                {
                    return m.FailName( node, _u("Error adding return Symbol %s"), name );
                }
                return true;
            }
            return m.Fail( node, _u("Module return must be an object or 1 function") );
        }

        ParseNode* objectElement = ParserWrapper::GetUnaryNode(objNode);
        if (!objectElement)
        {
            return m.Fail(node, _u("Return object must not be empty"));
        }
        while( objectElement )
        {
            ParseNode* member = nullptr;
            if( objectElement->nop == knopList )
            {
                member = ParserWrapper::GetBinaryLeft( objectElement );
                objectElement = ParserWrapper::GetBinaryRight( objectElement );
            }
            else if( objectElement->nop == knopMember )
            {
                member = objectElement;
                objectElement = nullptr;
            }
            else
            {
                return m.Fail( node, _u("Return object must only contain members") );
            }

            if( member )
            {
                if (!(member->Grfnop() & fnopBin))
                {
                    return m.Fail(node, _u("Return object member must be an assignment expression"));
                }

                ParseNode* field = ParserWrapper::GetBinaryLeft( member );
                ParseNode* value = ParserWrapper::GetBinaryRight( member );
                if( !ParserWrapper::IsNameDeclaration( field ) || !ParserWrapper::IsNameDeclaration( value ) )
                {
                    return m.Fail( node, _u("Return object member must be fields") );
                }

                AsmJsSymbol* sym = m.LookupIdentifier( value->name() );
                if( !sym )
                {
                    return m.FailName( node, _u("Symbol %s not recognized inside module"), value->name() );
                }

                if (!AsmJsFunc::Is(sym))
                {
                    return m.FailName(node, _u("Symbol %s can only be a function of the module"), value->name());
                }

                AsmJsFunc* func = AsmJsFunc::FromSymbol(sym);
                if( !m.AddExport( field->name(), func->GetFunctionIndex() ) )
                {
                    return m.FailName( node, _u("Error adding return Symbol %s"), value->name() );
                }
            }
        }

        return true;
    }

    bool AsmJSCompiler::CheckFuncPtrTables( AsmJsModuleCompiler &m )
    {
        ParseNode *list = m.GetCurrentParserNode();
        if (!list)
        {
            return true;
        }
        while (list->nop != knopEndCode)
        {
            ParseNode * varStmt = ParserWrapper::GetBinaryLeft(list);
            if (varStmt->nop != knopConstDecl && varStmt->nop != knopVarDecl)
            {
                break;
            }

            ParseNode* nodeInit = varStmt->AsParseNodeVar()->pnodeInit;
            if( !nodeInit || nodeInit->nop != knopArray )
            {
                return m.Fail( varStmt, _u("Invalid variable after function declaration") );
            }

            PropertyName tableName = varStmt->name();

            AsmJsSymbol* symFunctionTable = m.LookupIdentifier(tableName);
            if( !symFunctionTable)
            {
                // func table not used in functions disregard it
            }
            else
            {
                //Check name
                if(!AsmJsFunctionTable::Is(symFunctionTable))
                {
                    return m.FailName( varStmt, _u("Variable %s is already defined"), tableName );
                }

                AsmJsFunctionTable* table = AsmJsFunctionTable::FromSymbol(symFunctionTable);
                if( table->IsDefined() )
                {
                    return m.FailName( varStmt, _u("Multiple declaration of function table %s"), tableName );
                }

                // Check content of the array
                uint count = nodeInit->AsParseNodeArrLit()->count;
                if( table->GetSize() != count )
                {
                    return m.FailName( varStmt, _u("Invalid size of function table %s"), tableName );
                }

                // Set the content of the array in the table
                ParseNode* node = nodeInit->AsParseNodeArrLit()->pnode1;
                uint i = 0;
                while( node )
                {
                    ParseNode* funcNameNode = nullptr;
                    if( node->nop == knopList )
                    {
                        funcNameNode = ParserWrapper::GetBinaryLeft( node );
                        node = ParserWrapper::GetBinaryRight( node );
                    }
                    else
                    {
                        Assert( i + 1 == count );
                        funcNameNode = node;
                        node = nullptr;
                    }

                    if( ParserWrapper::IsNameDeclaration( funcNameNode ) )
                    {
                        AsmJsSymbol* sym = m.LookupIdentifier( funcNameNode->name() );
                        if (!AsmJsFunc::Is(sym))
                        {
                            return m.FailName( varStmt, _u("Element in function table %s is not a function"), tableName );
                        }
                        AsmJsFunc* func = AsmJsFunc::FromSymbol(sym);
                        AsmJsRetType retType;
                        if (!table->SupportsArgCall(func->GetArgCount(), func->GetArgTypeArray(), retType))
                        {
                            return m.FailName(funcNameNode, _u("Function signatures in table %s do not match"), tableName);
                        }
                        if (!table->CheckAndSetReturnType(func->GetReturnType()))
                        {
                            return m.FailName(funcNameNode, _u("Function return types in table %s do not match"), tableName);
                        }
                        table->SetModuleFunctionIndex( func->GetFunctionIndex(), i );
                        ++i;
                    }
                    else
                    {
                        return m.FailName(funcNameNode, _u("Element in function table %s is not a function name"), tableName);
                    }
                }

                table->Define();
            }

            list = ParserWrapper::GetBinaryRight(list);
        }

        if( !m.AreAllFuncTableDefined() )
        {
            return m.Fail(list, _u("Some function table were used but not defined"));
        }

        m.SetCurrentParseNode(list);
        return true;
    }

    bool AsmJSCompiler::CheckModule( ExclusiveContext *cx, AsmJSParser &parser, ParseNode *stmtList )
    {
        AsmJsModuleCompiler m( cx, parser );
        if( !m.Init() )
        {
            return false;
        }
        if( PropertyName moduleFunctionName = ParserWrapper::FunctionName( m.GetModuleFunctionNode() ) )
        {
            if( !CheckModuleLevelName( m, m.GetModuleFunctionNode(), moduleFunctionName ) )
            {
                return false;
            }
            m.InitModuleName( moduleFunctionName );

            if( PHASE_TRACE1( Js::ByteCodePhase ) )
            {
                Output::Print( _u("Asm.Js Module [%s] detected, trying to compile\n"), moduleFunctionName->Psz() );
            }
        }

        m.AccumulateCompileTime(AsmJsCompilation::Module);

        if( !CheckFunctionHead( m, m.GetModuleFunctionNode() ) )
        {
            goto AsmJsCompilationError;
        }

        if (!CheckModuleArguments(m, m.GetModuleFunctionNode()))
        {
            goto AsmJsCompilationError;
        }

        if (!CheckModuleGlobals(m))
        {
            goto AsmJsCompilationError;
        }

        m.AccumulateCompileTime(AsmJsCompilation::Module);

        if (!CheckFunctionsSequential(m))
        {
            goto AsmJsCompilationError;
        }

        m.AccumulateCompileTime();
        m.InitMemoryOffsets();

        if( !m.CompileAllFunctions() )
        {
            return false;
        }

        m.AccumulateCompileTime(AsmJsCompilation::ByteCode);

        if (!CheckFuncPtrTables(m))
        {
            m.RevertAllFunctions();
            return false;
        }

        m.AccumulateCompileTime();

        if (!CheckModuleReturn(m))
        {
            m.RevertAllFunctions();
            return false;
        }

        m.CommitFunctions();
        m.CommitModule();
        m.AccumulateCompileTime(AsmJsCompilation::Module);

        m.PrintCompileTrace();

        return true;

AsmJsCompilationError:
        ParseNode * moduleNode = m.GetModuleFunctionNode();
        if( moduleNode )
        {
            FunctionBody* body = moduleNode->AsParseNodeFnc()->funcInfo->GetParsedFunctionBody();
            body->ResetByteCodeGenState();
        }

        cx->byteCodeGenerator->Writer()->Reset();
        return false;
    }

    bool AsmJSCompiler::Compile(ExclusiveContext *cx, AsmJSParser parser, ParseNode *stmtList)
    {
        if (!CheckModule(cx, parser, stmtList))
        {
            OutputError(cx->scriptContext, _u("Asm.js compilation failed."));
            return false;
        }
        return true;
    }

    void AsmJSCompiler::OutputError(ScriptContext * scriptContext, const wchar * message, ...)
    {
        va_list argptr;
        va_start(argptr, message);
        VOutputMessage(scriptContext, DEIT_ASMJS_FAILED, message, argptr);
    }

    void AsmJSCompiler::OutputMessage(ScriptContext * scriptContext, const DEBUG_EVENT_INFO_TYPE messageType, const wchar * message, ...)
    {
        va_list argptr;
        va_start(argptr, message);
        VOutputMessage(scriptContext, messageType, message, argptr);
    }

    void AsmJSCompiler::VOutputMessage(ScriptContext * scriptContext, const DEBUG_EVENT_INFO_TYPE messageType, const wchar * message, va_list argptr)
    {
        char16 buf[2048];
        size_t size;

        size = _vsnwprintf_s(buf, _countof(buf), _TRUNCATE, message, argptr);
        if (size == -1)
        {
            size = _countof(buf) - 1;  // characters written, excluding the terminating null
        }
#ifdef ENABLE_SCRIPT_DEBUGGING
        scriptContext->RaiseMessageToDebugger(messageType, buf, scriptContext->GetUrl());
#endif
        if (PHASE_TRACE1(AsmjsPhase) || PHASE_TESTTRACE1(AsmjsPhase))
        {
            Output::PrintBuffer(buf, size);
            Output::Print(_u("\n"));
            Output::Flush();
        }
    }
}

#endif
