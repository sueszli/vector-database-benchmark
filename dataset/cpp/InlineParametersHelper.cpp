#include "InlineParametersHelper.h"
#include <easy/runtime/BitcodeTracker.h>

#include <llvm/Linker/Linker.h>

#include <llvm/Support/raw_ostream.h>

using namespace llvm;
using namespace easy;

HighLevelLayout::HighLevelLayout(easy::Context const& C, llvm::Function &F) {
  StructReturn_ = nullptr;

  FunctionType* FTy = F.getFunctionType();
  if(F.arg_begin()->hasStructRetAttr())
    StructReturn_ = FTy->getParamType(0);

  Return_ = FTy->getReturnType();

  auto &BT = easy::BitcodeTracker::GetTracker();

  size_t ParamIdx = 0;
  size_t ArgIdx = 0;
  if(StructReturn_)
    ParamIdx++;

  for(easy::layout_id lid : C.getLayout()) {
    size_t N = BT.getLayoutInfo(lid).NumFields;

    Args_.emplace_back(ArgIdx, ParamIdx);
    HighLevelArg& Arg = Args_.back();

    size_t ArgEnd = (ParamIdx+N);

    bool SingleArg = (ParamIdx + 1) == ArgEnd;
    if(SingleArg) {
      Type* ParamTy = FTy->getParamType(ParamIdx);
      Arg.Types_.push_back(ParamTy);
      Arg.StructByPointer_ = ParamTy->isPointerTy();
      ++ParamIdx;
      ++ArgIdx;
    } else {
      for(; ParamIdx != ArgEnd; ++ParamIdx)
        Arg.Types_.push_back(FTy->getParamType(ParamIdx));
      ++ArgIdx;
    }
  }
}

llvm::SmallVector<llvm::Value*, 4>
easy::GetForwardArgs(easy::HighLevelLayout::HighLevelArg &ArgInF, easy::HighLevelLayout &FHLL,
               llvm::Function &Wrapper, easy::HighLevelLayout &WrapperHLL) {

  llvm::SmallVector<llvm::Value*, 4> Args;

  auto GetArg = [&Wrapper, &FHLL](size_t i) -> Argument*
    { return &*(Wrapper.arg_begin()+i+(FHLL.StructReturn_ ? 1 : 0)); };

  // find layout in the new wrapper
  size_t ArgPosition = ArgInF.Position_;
  auto &ArgInWrapper = *std::find_if(WrapperHLL.Args_.begin(), WrapperHLL.Args_.end(),
                                     [ArgPosition](HighLevelLayout::HighLevelArg &ArgInWrapper)
                                      { return ArgInWrapper.Position_ == ArgPosition; });

  for(size_t j = 0; j != ArgInF.Types_.size(); ++j) {
    Args.push_back(GetArg(ArgInWrapper.FirstParamIdx_ + j));
  }
  return Args;
}

Constant* easy::GetScalarArgument(ArgumentBase const& Arg, Type* T) {
  switch(Arg.kind()) {
    case easy::ArgumentBase::AK_Int: {
      auto const *Int = Arg.as<easy::IntArgument>();
      return ConstantInt::get(T, Int->get(), true);
    }
    case easy::ArgumentBase::AK_Float: {
      auto const *Float = Arg.as<easy::FloatArgument>();
      return ConstantFP::get(T, Float->get());
    }
    case easy::ArgumentBase::AK_Ptr: {
      auto const *Ptr = Arg.as<easy::PtrArgument>();
      uintptr_t Addr = (uintptr_t)Ptr->get();
      return ConstantExpr::getIntToPtr(
                ConstantInt::get(Type::getInt64Ty(T->getContext()), Addr, false),
                T);
    }
    default:
      return nullptr;
  }
}

llvm::Constant* easy::LinkPointerIfPossible(llvm::Module &M, easy::PtrArgument const &Ptr, Type* PtrTy) {
  auto &BT = easy::BitcodeTracker::GetTracker();
  void* PtrValue = const_cast<void*>(Ptr.get());
  if(BT.hasGlobalMapping(PtrValue)) {
    const char* LName = std::get<0>(BT.getNameAndGlobalMapping(PtrValue));
    std::unique_ptr<Module> LM = BT.getModuleWithContext(PtrValue, M.getContext());

    if(!Linker::linkModules(M, std::move(LM), Linker::OverrideFromSrc,
                            [](Module &, const StringSet<> &){}))
    {
      GlobalValue *GV = M.getNamedValue(LName);
      if(GlobalVariable* G = dyn_cast<GlobalVariable>(GV)) {
        GV->setLinkage(llvm::Function::PrivateLinkage);
        if(GV->getType() != PtrTy) {
          return ConstantExpr::getPointerCast(GV, PtrTy);
        }
        return GV;
      }
      else if(llvm::Function* F = dyn_cast<llvm::Function>(GV)) {
        F->setLinkage(llvm::Function::PrivateLinkage);
        return F;
      }
      assert(false && "wtf");
    }
  }
  return nullptr;
}

std::pair<llvm::Constant*, size_t> easy::GetConstantFromRaw(llvm::DataLayout const& DL,
                                                            llvm::Type* T, const uint8_t* Raw) {
  // pack in a I8 constant vector and cast
  Type* I8 = Type::getInt8Ty(T->getContext());
  size_t Size = DL.getTypeStoreSize(T); // TODO: not sure about this

  SmallVector<Constant*, sizeof(uint64_t)> Elements(Size, nullptr);
  for(size_t i = 0; i != Size; ++i) {
    Elements[i] = ConstantInt::get(I8, Raw[i]);
  }

  Constant* DataAsI8 = ConstantVector::get(Elements);
  Constant* DataAsT;
  if(T->isPointerTy()) {
    Type* TInt = DL.getIntPtrType(T->getContext());
    Constant* DataAsTSizedInt = ConstantExpr::getBitCast(DataAsI8, TInt);
    DataAsT = ConstantExpr::getIntToPtr(DataAsTSizedInt, T);
  } else {
    DataAsT = ConstantExpr::getBitCast(DataAsI8, T);
  }
  return {DataAsT, Size};
}

static
size_t StoreStructField(llvm::IRBuilder<> &B,
                        llvm::DataLayout const &DL,
                        Type* Ty,
                        uint8_t const* Raw,
                        AllocaInst* Alloc, SmallVectorImpl<Value*> &GEP) {

  StructType* STy = dyn_cast<StructType>(Ty);
  size_t RawOffset = 0;
  if(STy) {
    errs() << "struct " << *STy << "\n";
    size_t Fields = STy->getNumContainedTypes();
    for(size_t Field = 0; Field != Fields; ++Field) {
      GEP.push_back(B.getInt32(Field));
      size_t Size = StoreStructField(B, DL, STy->getElementType(Field), Raw+RawOffset, Alloc, GEP);
      RawOffset += Size;
      GEP.pop_back();
    }
  } else {
    Constant* FieldValue;
    std::tie(FieldValue, RawOffset) = easy::GetConstantFromRaw(DL, Ty, (uint8_t const*)Raw);

    Value* FieldPtr = B.CreateGEP(nullptr, Alloc, GEP, "field.gep");
    B.CreateStore(FieldValue, FieldPtr);
  }
  return RawOffset;
}

llvm::AllocaInst* easy::GetStructAlloc(llvm::IRBuilder<> &B,
                                       llvm::DataLayout const &DL,
                                       easy::StructArgument const &Struct,
                                       llvm::Type* StructPtrTy) {
  Type* StructTy = StructPtrTy->getContainedType(0);
  AllocaInst* Alloc = B.CreateAlloca(StructTy);

  SmallVector<Value*, 4> GEP = {B.getInt32(0)};

  // TODO: Data points to the data structure or holds the data structure itself ?
  // Check that size matches the .data()

  size_t Size = StoreStructField(B, DL, StructTy, (uint8_t const*)Struct.get().data(), Alloc, GEP);

  return Alloc;
}
