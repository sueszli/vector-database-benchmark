#include <easy/attributes.h>

#include <llvm/IR/Module.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/InstIterator.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/DebugInfo.h>

#include <llvm/IR/LegacyPassManager.h>

#include "llvm/InitializePasses.h"

#include <llvm/Transforms/IPO.h>
#include <llvm/Transforms/Utils/CtorUtils.h>
#include <llvm/Transforms/Utils/Cloning.h>

#include <llvm/Bitcode/BitcodeWriter.h>

#include <llvm/ADT/SetVector.h>

#define DEBUG_TYPE "easy-register-bitcode"
#include <llvm/Support/Debug.h>

#include <llvm/Support/CommandLine.h>
#include <llvm/Support/Regex.h>

#include <llvm/Support/raw_ostream.h>

#include <memory>

#include "MayAliasTracer.h"
#include "StaticPasses.h"
#include "Utils.h"

using namespace llvm;

static cl::opt<std::string> RegexString("easy-export",
                                        cl::desc("A regular expression to describe functions to expose at runtime."),
                                        cl::init(""));

namespace easy {
  struct RegisterBitcode : public ModulePass {
    static char ID;

    RegisterBitcode()
      : ModulePass(ID) {};

    bool runOnModule(Module &M) override {

      // execute the rest of the easy::jit passes
      legacy::PassManager Passes;
      Passes.add(easy::createRegisterLayoutPass());
      bool Changed = Passes.run(M);

      SmallVector<GlobalObject*, 8> ObjectsToJIT;

      collectObjectsToJIT(M, ObjectsToJIT);

      if(ObjectsToJIT.empty())
        return Changed;

      SmallVector<GlobalValue*, 8> LocalVariables;
      collectLocalGlobals(M, LocalVariables);
      nameGlobals(LocalVariables, "unnamed_local_global");

      auto Bitcode = embedBitcode(M, ObjectsToJIT);
      GlobalVariable* GlobalMapping = getGlobalMapping(M, LocalVariables);

      Function* RegisterBitcodeFun = declareRegisterBitcode(M, GlobalMapping);
      registerBitcode(M, ObjectsToJIT, Bitcode, GlobalMapping, RegisterBitcodeFun);

      return Changed;
    }

    private:

    static bool canExtractBitcode(GlobalObject &GO, std::string &Reason) {
      if(GO.isDeclaration()) {
        Reason = "Can't extract a declaration.";
        return false;
      }
      return true;
    }

    static auto compilerInterface(Module &M) {
      SmallVector<std::reference_wrapper<Function>, 4> Funs;
      std::copy_if(M.begin(), M.end(), std::back_inserter(Funs),
                   [](Function &F) {return F.getSection() == CI_SECTION;});
      return Funs;
    }

    static bool mayBeOverload(Function *FOverload, Function *F) {
      auto *FOverloadTy = FOverload->getFunctionType();
      auto *FTy = F->getFunctionType();
      if (FTy->getReturnType() != FOverloadTy->getReturnType())
        return false;
      if (FTy->getNumParams() != FOverloadTy->getNumParams())
        return false;
      if (FOverloadTy->isVarArg())
        return false;

      auto FParamIter = FTy->param_begin();
      auto FOverloadParamIter = FOverloadTy->param_begin();

      // TODO: check that first parameter type are compatible?
      if (!isa<PointerType>(*FOverloadParamIter))
        return false;

      for (++FParamIter, ++FOverloadParamIter; FParamIter != FTy->param_end();
           ++FParamIter, ++FOverloadParamIter) {
        if (*FParamIter != *FOverloadParamIter)
          return false;
      }

      // an overload must be registered in a virtual table
      for(User* U : F->users()) {
        auto* CE = dyn_cast<ConstantExpr>(U);
        if(!CE || !CE->isCast())
          continue;

        for(User* CEU : CE->users()) {
          if(auto* Init = dyn_cast<ConstantAggregate>(CEU)) {
            return true; // probably a vtable
          }
        }
      }

      return false;
    }

    void collectObjectsToJIT(Module &M, SmallVectorImpl<GlobalObject*> &ObjectsToJIT) {

      // get **all** functions passed as parameter to easy jit calls
      //   not only the target function, but also its parameters
      deduceObjectsToJIT(M);
      regexFunctionsToJIT(M);
      deduceVirtualMethodsToJIT(M);

      // get functions in section jit section
      for(GlobalObject &GO : M.global_objects()) {
        if(GO.getSection() != JIT_SECTION)
          continue;

        GO.setSection(""); // drop the identifier

        std::string Reason;
        if(!canExtractBitcode(GO, Reason)) {
          DEBUG(dbgs() << "Could not extract global '" << GO.getName() << "'. " << Reason << "\n");
          continue;
        }
        DEBUG(dbgs() << "Global '" << GO.getName() << "' marked for extraction.\n");

        ObjectsToJIT.push_back(&GO);
      }

      // also collect virtual functions in two steps

    }

    static bool isConstant(GlobalObject const& GO) {
      if(isa<Function const>(GO))
        return true;
      return cast<GlobalVariable>(GO).isConstant();
    }

    void deduceVirtualMethodsToJIT(Module &M) {
      // First collect all loaded types we could monitor stores but stores
      // could be done at call site, outside of ObjectsToJIT's scopes

      SmallPtrSet<Type*, 8> VirtualMethodTys;
      for(Function& F: M) {
        if(F.getSection() != JIT_SECTION)
          continue;
        for(auto& I : instructions(F)) {
          auto* LI = dyn_cast<LoadInst>(&I);
          if(!LI)
            continue;

          MDNode *Tag = I.getMetadata(LLVMContext::MD_tbaa);
          if(!Tag || !Tag->isTBAAVtableAccess())
            continue;

          VirtualMethodTys.insert(cast<PointerType>(cast<PointerType>(LI->getType())->getElementType())->getElementType());
        }
      }

      // Second look at functions that have a type compatible with the virtual one
      for(GlobalObject &GO : M.global_objects()) {
        GlobalVariable* GV = dyn_cast<GlobalVariable>(&GO);
        if(!GV || ! GV->hasInitializer())
          continue;

        ConstantStruct* CS = dyn_cast<ConstantStruct>(GV->getInitializer());
        if(!CS || CS->getNumOperands() != 1)
          continue;

        ConstantAggregate* CA = dyn_cast<ConstantAggregate>(CS->getOperand(0));
        if(!CA)
          continue;

        for(Use& U : CA->operands()) {
          Constant* C = dyn_cast<Constant>(U.get());
          if(!C)
            continue;
          if(auto* CE = dyn_cast<ConstantExpr>(C)) {
            if(CE->isCast()) {
              C = CE->getOperand(0);
            }
          }
          auto* F = dyn_cast<Function>(C);
          if(!F)
            continue;

          auto* FTy = F->getFunctionType();
          if(VirtualMethodTys.count(FTy)) {
            F->setSection(JIT_SECTION);
            // also look for overloads
            for(auto& FOverload: M) {
              if(&FOverload == F)
                continue;
              if(mayBeOverload(&FOverload, F))
                FOverload.setSection(JIT_SECTION);
            }
          }
        }
      }
    }

    void deduceObjectsToJIT(Module &M) {
      for(Function &EasyJitFun : compilerInterface(M)) {
        for(User* U : EasyJitFun.users()) {
          if(CallSite CS{U}) {
            for(Value* O : CS.args()) {
              O = O->stripPointerCastsNoFollowAliases();
              MayAliasTracer Tracer(O);
              for(GlobalObject& GO: M.global_objects()) {
                if(isConstant(GO) and Tracer.count(GO)) {
                  GO.setSection(JIT_SECTION);
                }
              }
            }
          }
        }
      }
    }

    static void regexFunctionsToJIT(Module &M) {
      if(RegexString.empty())
        return;
      llvm::Regex Match(RegexString);
      for(GlobalObject &GO : M.global_objects())
        if(Match.match(GO.getName()))
          GO.setSection(JIT_SECTION);
    }

    static void collectLocalGlobals(Module &M, SmallVectorImpl<GlobalValue*> &Globals) {
      for(GlobalVariable &GV : M.globals())
        if(GV.hasLocalLinkage())
          Globals.push_back(&GV);
    }

    static void nameGlobals(SmallVectorImpl<GlobalValue*> &Globals, Twine Name) {
      for(GlobalValue *GV : Globals)
        if(!GV->hasName())
          GV->setName(Name);
    }

    static GlobalVariable*
    getGlobalMapping(Module &M, SmallVectorImpl<GlobalValue*> &Globals) {
      LLVMContext &C = M.getContext();
      SmallVector<Constant*, 8> Entries;

      Type* PtrTy = Type::getInt8PtrTy(C);
      StructType *EntryTy = StructType::get(C, {PtrTy, PtrTy}, true);

      for(GlobalValue* GV : Globals) {
        GlobalVariable* Name = getStringGlobal(M, GV->getName());
        Constant* NameCast = ConstantExpr::getPointerCast(Name, PtrTy);
        Constant* GVCast = GV;
        if(GV->getType() != PtrTy)
          GVCast = ConstantExpr::getPointerCast(GV, PtrTy);
        Constant* Entry = ConstantStruct::get(EntryTy, {NameCast, GVCast});
        Entries.push_back(Entry);
      }
      Entries.push_back(Constant::getNullValue(EntryTy));

      Constant* Init = ConstantArray::get(ArrayType::get(EntryTy, Entries.size()), Entries);
      return new GlobalVariable(M, Init->getType(), true,
                                GlobalVariable::PrivateLinkage,
                                Init, "global_mapping");
    }

    static SmallVector<GlobalVariable*, 8>
    embedBitcode(Module &M, SmallVectorImpl<GlobalObject*> &Objs) {
      SmallVector<GlobalVariable*, 8> Bitcode(Objs.size());
      for(size_t i = 0, n = Objs.size(); i != n; ++i)
        Bitcode[i] = embedBitcode(M, *Objs[i]);
      return Bitcode;
    }

    static GlobalVariable* embedBitcode(Module &M, GlobalObject& GO) {
      std::unique_ptr<Module> Embed = CloneModule(&M);

      GlobalValue *FEmbed = Embed->getNamedValue(GO.getName());
      assert(FEmbed && "global value with that name exists");
      cleanModule(*FEmbed, *Embed);

      Twine ModuleName = GO.getName() + "_bitcode";
      Embed->setModuleIdentifier(ModuleName.str());

      return writeModuleToGlobal(M, *Embed, FEmbed->getName() + "_bitcode");
    }

    static std::string moduleToString(Module &M) {
      std::string s;
      raw_string_ostream so(s);
      WriteBitcodeToFile(&M, so);
      so.flush();
      return s;
    }

    static GlobalVariable* writeModuleToGlobal(Module &M, Module &Embed, Twine Name) {
      std::string Bitcode = moduleToString(Embed);
      Constant* BitcodeInit = ConstantDataArray::getString(M.getContext(), Bitcode, true);
      return new GlobalVariable(M, BitcodeInit->getType(), true,
                                GlobalVariable::PrivateLinkage,
                                BitcodeInit, Name);
    }

    static void cleanModule(GlobalValue &Entry, Module &M) {

      llvm::StripDebugInfo(M);

      bool ForFunction = isa<Function>(Entry);

      auto Referenced = getReferencedFromEntry(Entry);
      Referenced.push_back(&Entry);

      if(ForFunction) {
        Entry.setLinkage(GlobalValue::ExternalLinkage);
      }

      //clean the cloned module
      legacy::PassManager Passes;
      Passes.add(createGVExtractionPass(Referenced));
      Passes.add(createGlobalDCEPass());
      Passes.add(createStripDeadDebugInfoPass());
      Passes.add(createStripDeadPrototypesPass());
      Passes.run(M);

      if(ForFunction) {
        fixLinkages(Entry, M);
      }
    }

    static std::vector<GlobalValue*> getReferencedFromEntry(GlobalValue &Entry) {
      std::vector<GlobalValue*> Funs;

      SmallPtrSet<User*, 32> Visited;
      SmallVector<User*, 8> ToVisit;
      ToVisit.push_back(&Entry);

      while(!ToVisit.empty()) {
        User* U = ToVisit.pop_back_val();
        if(!Visited.insert(U).second)
          continue;
        if(Function* UF = dyn_cast<Function>(U)) {
          Funs.push_back(UF);

          for(Instruction &I : instructions(UF))
            for(Value* Op : I.operands())
              if(User* OpU = dyn_cast<User>(Op))
                ToVisit.push_back(OpU);
        }
        else if(GlobalVariable* GV = dyn_cast<GlobalVariable>(U)) {
          if(GV->hasInitializer()) {
            ToVisit.push_back(GV->getInitializer());
          }
        }

        for(Value* Op : U->operands())
          if(User* OpU = dyn_cast<User>(Op))
            ToVisit.push_back(OpU);
      }

      return Funs;
    }

    static void fixLinkages(GlobalValue &Entry, Module &M) {
      for(GlobalValue &GV : M.global_values()) {
        if(GV.getName().startswith("llvm."))
          continue;

        if(GlobalObject* GO = dyn_cast<GlobalObject>(&GV)) {
          GO->setComdat(nullptr);
        }

        if(auto* GVar = dyn_cast<GlobalVariable>(&GV)) {
          // gv becomes a declaration
          GVar->setInitializer(nullptr);
          GVar->setVisibility(GlobalValue::DefaultVisibility);
          GVar->setLinkage(GlobalValue::ExternalLinkage);
        } else if(auto* F = dyn_cast<Function>(&GV)) {
          // f becomes private
          F->removeFnAttr(Attribute::NoInline);
          if(F == &Entry)
            continue;

          if(!F->isDeclaration() &&
             (F->getVisibility() != GlobalValue::DefaultVisibility ||
              F->getLinkage() != GlobalValue::PrivateLinkage)) {
            F->setVisibility(GlobalValue::DefaultVisibility);
            F->setLinkage(GlobalValue::PrivateLinkage);
          }
        } else llvm::report_fatal_error("Easy::Jit [not yet implemented]: handle aliases, ifuncs.");
      }
    }

    Function* declareRegisterBitcode(Module &M, GlobalVariable *GlobalMapping) {
      StringRef Name = "easy_register";
      if(Function* F = M.getFunction(Name))
        return F;

      LLVMContext &C = M.getContext();
      DataLayout const &DL = M.getDataLayout();

      Type* Void = Type::getVoidTy(C);
      Type* I8Ptr = Type::getInt8PtrTy(C);
      Type* GMTy = GlobalMapping->getType();
      Type* SizeT = DL.getLargestLegalIntType(C);

      assert(SizeT);

      FunctionType* FTy =
          FunctionType::get(Void, {I8Ptr, I8Ptr, GMTy, I8Ptr, SizeT}, false);
      return Function::Create(FTy, Function::ExternalLinkage, Name, &M);
    }

    static void
    registerBitcode(Module &M, SmallVectorImpl<GlobalObject*> &Objs,
                    SmallVectorImpl<GlobalVariable*> &Bitcodes,
                    Value* GlobalMapping,
                    Function* RegisterBitcodeFun) {
      // Create static initializer with low priority to register everything
      Type* FPtr = RegisterBitcodeFun->getFunctionType()->getParamType(0);
      Type* StrPtr = RegisterBitcodeFun->getFunctionType()->getParamType(1);
      Type* BitcodePtr = RegisterBitcodeFun->getFunctionType()->getParamType(3);
      Type* SizeTy = RegisterBitcodeFun->getFunctionType()->getParamType(4);

      Function *Ctor = GetCtor(M, "register_bitcode");
      IRBuilder<> B(Ctor->getEntryBlock().getTerminator());

      for(size_t i = 0, n = Objs.size(); i != n; ++i) {
        GlobalVariable* Name = getStringGlobal(M, Objs[i]->getName());
        ArrayType* ArrTy = cast<ArrayType>(Bitcodes[i]->getInitializer()->getType());
        size_t Size = ArrTy->getNumElements()-1; /*-1 for the 0 terminator*/

        Value* Fun = B.CreatePointerCast(Objs[i], FPtr);
        Value* NameCast = B.CreatePointerCast(Name, StrPtr);
        Value* Bitcode = B.CreatePointerCast(Bitcodes[i], BitcodePtr);
        Value* BitcodeSize = ConstantInt::get(SizeTy, Size, false);

        // fun, name, gm, bitcode, bitcode size
        B.CreateCall(RegisterBitcodeFun,
                     {Fun, NameCast, GlobalMapping, Bitcode, BitcodeSize}, "");
      }
    }

    static GlobalVariable* getStringGlobal(Module& M, StringRef Name) {
      Constant* Init = ConstantDataArray::getString(M.getContext(), Name, true);
      return new GlobalVariable(M, Init->getType(), true,
                                GlobalVariable::PrivateLinkage,
                                Init, Name + "_name");
    }
  };

  char RegisterBitcode::ID = 0;
  static RegisterPass<RegisterBitcode> Register("easy-register-bitcode",
    "Parse the compilation unit and insert runtime library calls to register "
    "the bitcode associated to functions marked as \"jit\".",
                                                false, false);

  llvm::Pass* createRegisterBitcodePass() {
    return new RegisterBitcode();
  }
}
