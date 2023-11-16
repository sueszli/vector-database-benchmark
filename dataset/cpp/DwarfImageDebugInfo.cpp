/*
 * Copyright 2009-2012, Ingo Weinhold, ingo_weinhold@gmx.de.
 * Copyright 2012-2018, Rene Gollent, rene@gollent.com.
 * Distributed under the terms of the MIT License.
 */


#include "DwarfImageDebugInfo.h"

#include <errno.h>
#include <stdio.h>
#include <unistd.h>

#include <algorithm>
#include <new>

#include <AutoDeleter.h>
#include <AutoLocker.h>

#include "Architecture.h"
#include "BasicFunctionDebugInfo.h"
#include "CLanguage.h"
#include "CompilationUnit.h"
#include "CppLanguage.h"
#include "CpuState.h"
#include "DebuggerInterface.h"
#include "DebugInfoEntries.h"
#include "Demangler.h"
#include "DisassembledCode.h"
#include "Dwarf.h"
#include "DwarfFile.h"
#include "DwarfFunctionDebugInfo.h"
#include "DwarfStackFrameDebugInfo.h"
#include "DwarfTargetInterface.h"
#include "DwarfTypeFactory.h"
#include "DwarfTypes.h"
#include "DwarfUtils.h"
#include "ElfFile.h"
#include "FileManager.h"
#include "FileSourceCode.h"
#include "FunctionID.h"
#include "FunctionInstance.h"
#include "GlobalTypeLookup.h"
#include "Image.h"
#include "ImageDebugInfo.h"
#include "InstructionInfo.h"
#include "LocatableFile.h"
#include "Register.h"
#include "RegisterMap.h"
#include "SourceFile.h"
#include "StackFrame.h"
#include "Statement.h"
#include "StringUtils.h"
#include "SymbolInfo.h"
#include "TargetAddressRangeList.h"
#include "Team.h"
#include "TeamFunctionSourceInformation.h"
#include "TeamMemory.h"
#include "Tracing.h"
#include "TypeLookupConstraints.h"
#include "UnsupportedLanguage.h"
#include "Variable.h"
#include "ValueLocation.h"


namespace {


// #pragma mark - HasTypePredicate


template<typename EntryType>
struct HasTypePredicate {
	inline bool operator()(EntryType* entry) const
	{
		return entry->GetType() != NULL;
	}
};

}


// #pragma mark - BasicTargetInterface


struct DwarfImageDebugInfo::BasicTargetInterface : DwarfTargetInterface {
	BasicTargetInterface(const Register* registers, int32 registerCount,
		RegisterMap* fromDwarfMap, Architecture* architecture,
		TeamMemory* teamMemory)
		:
		fRegisters(registers),
		fRegisterCount(registerCount),
		fFromDwarfMap(fromDwarfMap),
		fArchitecture(architecture),
		fTeamMemory(teamMemory)
	{
		fFromDwarfMap->AcquireReference();
	}

	~BasicTargetInterface()
	{
		fFromDwarfMap->ReleaseReference();
	}

	virtual uint32 CountRegisters() const
	{
		return fRegisterCount;
	}

	virtual uint32 RegisterValueType(uint32 index) const
	{
		const Register* reg = _RegisterAt(index);
		return reg != NULL ? reg->ValueType() : 0;
	}

	virtual bool GetRegisterValue(uint32 index, BVariant& _value) const
	{
		return false;
	}

	virtual bool SetRegisterValue(uint32 index, const BVariant& value)
	{
		return false;
	}

	virtual bool IsCalleePreservedRegister(uint32 index) const
	{
		const Register* reg = _RegisterAt(index);
		return reg != NULL && reg->IsCalleePreserved();
	}

	virtual status_t InitRegisterRules(CfaContext& context) const
	{
		return fArchitecture->InitRegisterRules(context);
	}

	virtual bool ReadMemory(target_addr_t address, void* buffer,
		size_t size) const
	{
		ssize_t bytesRead = fTeamMemory->ReadMemory(address, buffer, size);
		return bytesRead >= 0 && (size_t)bytesRead == size;
	}

	virtual bool ReadValueFromMemory(target_addr_t address,
		uint32 valueType, BVariant& _value) const
	{
		return fArchitecture->ReadValueFromMemory(address, valueType, _value)
			== B_OK;
	}

	virtual bool ReadValueFromMemory(target_addr_t addressSpace,
		target_addr_t address, uint32 valueType, BVariant& _value) const
	{
		return fArchitecture->ReadValueFromMemory(addressSpace, address,
			valueType, _value) == B_OK;
	}

protected:
	const Register* _RegisterAt(uint32 dwarfIndex) const
	{
		int32 index = fFromDwarfMap->MapRegisterIndex(dwarfIndex);
		return index >= 0 && index < fRegisterCount ? fRegisters + index : NULL;
	}

protected:
	const Register*	fRegisters;
	int32			fRegisterCount;
	RegisterMap*	fFromDwarfMap;
	Architecture*	fArchitecture;
	TeamMemory*		fTeamMemory;
};


// #pragma mark - UnwindTargetInterface


struct DwarfImageDebugInfo::UnwindTargetInterface : BasicTargetInterface {
	UnwindTargetInterface(const Register* registers, int32 registerCount,
		RegisterMap* fromDwarfMap, RegisterMap* toDwarfMap, CpuState* cpuState,
		Architecture* architecture, TeamMemory* teamMemory)
		:
		BasicTargetInterface(registers, registerCount, fromDwarfMap,
			architecture, teamMemory),
		fToDwarfMap(toDwarfMap),
		fCpuState(cpuState)
	{
		fToDwarfMap->AcquireReference();
		fCpuState->AcquireReference();
	}

	~UnwindTargetInterface()
	{
		fToDwarfMap->ReleaseReference();
		fCpuState->ReleaseReference();
	}

	virtual bool GetRegisterValue(uint32 index, BVariant& _value) const
	{
		const Register* reg = _RegisterAt(index);
		if (reg == NULL)
			return false;
		return fCpuState->GetRegisterValue(reg, _value);
	}

	virtual bool SetRegisterValue(uint32 index, const BVariant& value)
	{
		const Register* reg = _RegisterAt(index);
		if (reg == NULL)
			return false;
		return fCpuState->SetRegisterValue(reg, value);
	}

private:
	RegisterMap*	fToDwarfMap;
	CpuState*		fCpuState;
};


// #pragma mark - EntryListWrapper


/*!	Wraps a DebugInfoEntryList, which is a typedef and thus cannot appear in
	the header, since our policy disallows us to include DWARF headers there.
*/
struct DwarfImageDebugInfo::EntryListWrapper {
	const DebugInfoEntryList&	list;

	EntryListWrapper(const DebugInfoEntryList& list)
		:
		list(list)
	{
	}
};


// #pragma mark - DwarfImageDebugInfo::TypeNameKey


struct DwarfImageDebugInfo::TypeNameKey {
	BString			typeName;

	TypeNameKey(const BString& typeName)
		:
		typeName(typeName)
	{
	}

	uint32 HashValue() const
	{
		return StringUtils::HashValue(typeName);
	}

	bool operator==(const TypeNameKey& other) const
	{
		return typeName == other.typeName;
	}
};


// #pragma mark - DwarfImageDebugInfo::TypeNameEntry


struct DwarfImageDebugInfo::TypeNameEntry : TypeNameKey {
	TypeNameEntry* next;
	TypeEntryList types;

	TypeNameEntry(const BString& name)
		:
		TypeNameKey(name),
		types(10, true)
	{
	}

	~TypeNameEntry()
	{
	}

};


// #pragma mark - DwarfImageDebugInfo::TypeNameEntryHashDefinition


struct DwarfImageDebugInfo::TypeNameEntryHashDefinition {
	typedef TypeNameKey		KeyType;
	typedef	TypeNameEntry	ValueType;

	size_t HashKey(const TypeNameKey& key) const
	{
		return key.HashValue();
	}

	size_t Hash(const TypeNameEntry* value) const
	{
		return value->HashValue();
	}

	bool Compare(const TypeNameKey& key,
		const TypeNameEntry* value) const
	{
		return key == *value;
	}

	TypeNameEntry*& GetLink(TypeNameEntry* value) const
	{
		return value->next;
	}
};


// #pragma mark - DwarfImageDebugInfo::TypeEntryInfo


struct DwarfImageDebugInfo::TypeEntryInfo {
	DIEType* type;
	CompilationUnit* unit;

	TypeEntryInfo(DIEType* type, CompilationUnit* unit)
		:
		type(type),
		unit(unit)
	{
	}
};


// #pragma mark - DwarfImageDebugInfo


DwarfImageDebugInfo::DwarfImageDebugInfo(const ImageInfo& imageInfo,
	DebuggerInterface* interface, Architecture* architecture,
	FileManager* fileManager, GlobalTypeLookup* typeLookup,
	GlobalTypeCache* typeCache, TeamFunctionSourceInformation* sourceInfo,
	DwarfFile* file)
	:
	fLock("dwarf image debug info"),
	fImageInfo(imageInfo),
	fDebuggerInterface(interface),
	fArchitecture(architecture),
	fFileManager(fileManager),
	fTypeLookup(typeLookup),
	fTypeCache(typeCache),
	fSourceInfo(sourceInfo),
	fTypeNameTable(NULL),
	fFile(file),
	fTextSegment(NULL),
	fRelocationDelta(0),
	fTextSectionStart(0),
	fTextSectionEnd(0),
	fPLTSectionStart(0),
	fPLTSectionEnd(0)
{
	fDebuggerInterface->AcquireReference();
	fFile->AcquireReference();
	fTypeCache->AcquireReference();
}


DwarfImageDebugInfo::~DwarfImageDebugInfo()
{
	fDebuggerInterface->ReleaseReference();
	fFile->ReleaseReference();
	fTypeCache->ReleaseReference();

	TypeNameEntry* entry = fTypeNameTable->Clear(true);
	while (entry != NULL) {
		TypeNameEntry* next = entry->next;
		delete entry;
		entry = next;
	}
	delete fTypeNameTable;
}


status_t
DwarfImageDebugInfo::Init()
{
	status_t error = fLock.InitCheck();
	if (error != B_OK)
		return error;

	fTextSegment = fFile->GetElfFile()->TextSegment();
	if (fTextSegment == NULL)
		return B_ENTRY_NOT_FOUND;

	fRelocationDelta = fImageInfo.TextBase() - fTextSegment->LoadAddress();

	ElfSection* section = fFile->GetElfFile()->FindSection(".text");
	if (section != NULL) {
		fTextSectionStart = section->LoadAddress() + fRelocationDelta;
		fTextSectionEnd = fTextSectionStart + section->Size();
	}

	section = fFile->GetElfFile()->FindSection(".plt");
	if (section != NULL) {
		fPLTSectionStart = section->LoadAddress() + fRelocationDelta;
		fPLTSectionEnd = fPLTSectionStart + section->Size();
	}

	return _BuildTypeNameTable();
}


status_t
DwarfImageDebugInfo::GetFunctions(const BObjectList<SymbolInfo>& symbols,
	BObjectList<FunctionDebugInfo>& functions)
{
	TRACE_IMAGES("DwarfImageDebugInfo::GetFunctions()\n");
	TRACE_IMAGES("  %" B_PRId32 " compilation units\n",
		fFile->CountCompilationUnits());

	status_t error = B_OK;
	for (int32 i = 0; CompilationUnit* unit = fFile->CompilationUnitAt(i);
			i++) {
		DIECompileUnitBase* unitEntry = unit->UnitEntry();
//		printf("  %s:\n", unitEntry->Name());
//		printf("    address ranges:\n");
//		TargetAddressRangeList* rangeList = unitEntry->AddressRanges();
//		if (rangeList != NULL) {
//			int32 count = rangeList->CountRanges();
//			for (int32 i = 0; i < count; i++) {
//				TargetAddressRange range = rangeList->RangeAt(i);
//				printf("      %#llx - %#llx\n", range.Start(), range.End());
//			}
//		} else {
//			printf("      %#llx - %#llx\n", (target_addr_t)unitEntry->LowPC(),
//				(target_addr_t)unitEntry->HighPC());
//		}

//		printf("    functions:\n");
		for (DebugInfoEntryList::ConstIterator it
					= unitEntry->OtherChildren().GetIterator();
				DebugInfoEntry* entry = it.Next();) {
			if (entry->Tag() == DW_TAG_subprogram) {
				DIESubprogram* subprogramEntry
					= static_cast<DIESubprogram*>(entry);
				error = _AddFunction(subprogramEntry, unit, functions);
				if (error != B_OK)
					return error;
			}

			DIENamespace* nsEntry = dynamic_cast<DIENamespace*>(entry);
			if (nsEntry != NULL) {
				error = _RecursiveTraverseNamespaceForFunctions(nsEntry, unit,
					functions);
				if (error != B_OK)
					return error;
			}
		}
	}

	if (fFile->CountCompilationUnits() != 0)
		return B_OK;

	// if we had no compilation units, fall back to providing basic
	// debug infos with DWARF-supported call frame unwinding,
	// if available.
	if (fFile->HasFrameInformation()) {
		return SpecificImageDebugInfo::GetFunctionsFromSymbols(symbols,
			functions, fDebuggerInterface, fImageInfo, this);
	}

	return B_OK;
}


status_t
DwarfImageDebugInfo::GetType(GlobalTypeCache* cache, const BString& name,
	const TypeLookupConstraints& constraints, Type*& _type)
{
	TypeNameEntry* entry = fTypeNameTable->Lookup(name);
	if (entry == NULL)
		return B_ENTRY_NOT_FOUND;

	for (int32 i = 0; TypeEntryInfo* info = entry->types.ItemAt(i); i++) {
		DIEType* typeEntry = info->type;
		if (constraints.HasTypeKind()) {
			if (dwarf_tag_to_type_kind(typeEntry->Tag())
				!= constraints.TypeKind()) {
				continue;
			}

			if (!_EvaluateBaseTypeConstraints(typeEntry, constraints))
				continue;
		}

		if (constraints.HasSubtypeKind()
			&& dwarf_tag_to_subtype_kind(typeEntry->Tag())
				!= constraints.SubtypeKind()) {
			continue;
		}

		int32 registerCount = fArchitecture->CountRegisters();
		const Register* registers = fArchitecture->Registers();

		// get the DWARF <-> architecture register maps
		RegisterMap* toDwarfMap;
		RegisterMap* fromDwarfMap;
		status_t error = fArchitecture->GetDwarfRegisterMaps(&toDwarfMap,
			&fromDwarfMap);
		if (error != B_OK)
			return error;

		BReference<RegisterMap> toDwarfMapReference(toDwarfMap, true);
		BReference<RegisterMap> fromDwarfMapReference(fromDwarfMap, true);

		// create the target interface
		BasicTargetInterface* targetInterface
			= new(std::nothrow) BasicTargetInterface(registers, registerCount,
				fromDwarfMap, fArchitecture, fDebuggerInterface);
		if (targetInterface == NULL)
			return B_NO_MEMORY;

		BReference<BasicTargetInterface> targetInterfaceReference(
			targetInterface, true);

		DwarfTypeContext* typeContext = new(std::nothrow)
			DwarfTypeContext(fArchitecture, fImageInfo.ImageID(), fFile,
				info->unit, NULL, 0, 0, fRelocationDelta, targetInterface, NULL);
		if (typeContext == NULL)
			return B_NO_MEMORY;
		BReference<DwarfTypeContext> typeContextReference(typeContext, true);

		// create the type
		DwarfType* type;
		DwarfTypeFactory typeFactory(typeContext, fTypeLookup, cache);
		error = typeFactory.CreateType(typeEntry, type);
		if (error != B_OK)
			continue;

		_type = type;
		return B_OK;
	}

	return B_ENTRY_NOT_FOUND;
}


bool
DwarfImageDebugInfo::HasType(const BString& name,
	const TypeLookupConstraints& constraints) const
{
	TypeNameEntry* entry = fTypeNameTable->Lookup(name);
	if (entry == NULL)
		return false;

	for (int32 i = 0; TypeEntryInfo* info = entry->types.ItemAt(i); i++) {
		DIEType* typeEntry = info->type;
		if (constraints.HasTypeKind()) {
			if (dwarf_tag_to_type_kind(typeEntry->Tag())
				!= constraints.TypeKind()) {
				continue;
			}

			if (!_EvaluateBaseTypeConstraints(typeEntry, constraints))
				continue;
		}

		if (constraints.HasSubtypeKind()
			&& dwarf_tag_to_subtype_kind(typeEntry->Tag())
				!= constraints.SubtypeKind()) {
			continue;
		}

		return true;
	}

	return false;
}


AddressSectionType
DwarfImageDebugInfo::GetAddressSectionType(target_addr_t address)
{
	if (address >= fTextSectionStart && address < fTextSectionEnd)
		return ADDRESS_SECTION_TYPE_FUNCTION;

 	if (address >= fPLTSectionStart && address < fPLTSectionEnd)
		return ADDRESS_SECTION_TYPE_PLT;

	return ADDRESS_SECTION_TYPE_UNKNOWN;
}


status_t
DwarfImageDebugInfo::CreateFrame(Image* image,
	FunctionInstance* functionInstance, CpuState* cpuState,
	bool getFullFrameInfo, ReturnValueInfoList* returnValueInfos,
	StackFrame*& _frame, CpuState*& _previousCpuState)
{
	DwarfFunctionDebugInfo* function = dynamic_cast<DwarfFunctionDebugInfo*>(
		functionInstance->GetFunctionDebugInfo());

	FunctionID* functionID = functionInstance->GetFunctionID();
	BReference<FunctionID> functionIDReference;
	if (functionID != NULL)
		functionIDReference.SetTo(functionID, true);

	DIESubprogram* entry = function != NULL
		? function->SubprogramEntry() : NULL;

	TRACE_CFI("DwarfImageDebugInfo::CreateFrame(): subprogram DIE: %p, "
		"function: %s\n", entry,
		functionID->FunctionName().String());

	int32 registerCount = fArchitecture->CountRegisters();
	const Register* registers = fArchitecture->Registers();

	// get the DWARF <-> architecture register maps
	RegisterMap* toDwarfMap;
	RegisterMap* fromDwarfMap;
	status_t error = fArchitecture->GetDwarfRegisterMaps(&toDwarfMap,
		&fromDwarfMap);
	if (error != B_OK)
		return error;
	BReference<RegisterMap> toDwarfMapReference(toDwarfMap, true);
	BReference<RegisterMap> fromDwarfMapReference(fromDwarfMap, true);

	// create a clean CPU state for the previous frame
	CpuState* previousCpuState;
	error = fArchitecture->CreateCpuState(previousCpuState);
	if (error != B_OK)
		return error;
	BReference<CpuState> previousCpuStateReference(previousCpuState, true);

	// create the target interfaces
	UnwindTargetInterface* inputInterface
		= new(std::nothrow) UnwindTargetInterface(registers, registerCount,
			fromDwarfMap, toDwarfMap, cpuState, fArchitecture,
			fDebuggerInterface);
	if (inputInterface == NULL)
		return B_NO_MEMORY;
	BReference<UnwindTargetInterface> inputInterfaceReference(inputInterface,
		true);

	UnwindTargetInterface* outputInterface
		= new(std::nothrow) UnwindTargetInterface(registers, registerCount,
			fromDwarfMap, toDwarfMap, previousCpuState, fArchitecture,
			fDebuggerInterface);
	if (outputInterface == NULL)
		return B_NO_MEMORY;
	BReference<UnwindTargetInterface> outputInterfaceReference(outputInterface,
		true);

	// do the unwinding
	target_addr_t instructionPointer
		= cpuState->InstructionPointer() - fRelocationDelta;
	target_addr_t framePointer;
	CompilationUnit* unit = function != NULL ? function->GetCompilationUnit()
			: NULL;
	error = fFile->UnwindCallFrame(unit,
		fArchitecture->AddressSize(), fArchitecture->IsBigEndian(),
		entry, instructionPointer, inputInterface, outputInterface,
		framePointer);

	if (error != B_OK) {
		TRACE_CFI("Failed to unwind call frame: %s\n", strerror(error));
		return B_UNSUPPORTED;
	}

	TRACE_CFI_ONLY(
		TRACE_CFI("unwound registers:\n");
		for (int32 i = 0; i < registerCount; i++) {
			const Register* reg = registers + i;
			BVariant value;
			if (previousCpuState->GetRegisterValue(reg, value)) {
				TRACE_CFI("  %3s: %#" B_PRIx64 "\n", reg->Name(),
					value.ToUInt64());
			} else
				TRACE_CFI("  %3s: undefined\n", reg->Name());
		}
	)

	// create the stack frame debug info
	DIESubprogram* subprogramEntry = function != NULL ?
		function->SubprogramEntry() : NULL;
	DwarfStackFrameDebugInfo* stackFrameDebugInfo
		= new(std::nothrow) DwarfStackFrameDebugInfo(fArchitecture,
			fImageInfo.ImageID(), fFile, unit, subprogramEntry, fTypeLookup,
			fTypeCache, instructionPointer, framePointer, fRelocationDelta,
			inputInterface, fromDwarfMap);
	if (stackFrameDebugInfo == NULL)
		return B_NO_MEMORY;
	BReference<DwarfStackFrameDebugInfo> stackFrameDebugInfoReference(
		stackFrameDebugInfo, true);

	error = stackFrameDebugInfo->Init();
	if (error != B_OK)
		return error;

	// create the stack frame
	StackFrame* frame = new(std::nothrow) StackFrame(STACK_FRAME_TYPE_STANDARD,
		cpuState, framePointer, cpuState->InstructionPointer(),
		stackFrameDebugInfo);
	if (frame == NULL)
		return B_NO_MEMORY;
	BReference<StackFrame> frameReference(frame, true);

	error = frame->Init();
	if (error != B_OK)
		return error;

	frame->SetReturnAddress(previousCpuState->InstructionPointer());
		// Note, this is correct, since we actually retrieved the return
		// address. Our caller will fix the IP for us.

	// The subprogram entry may not be available since this may be a case
	// where .eh_frame was used to unwind the stack without other DWARF
	// info being available.
	if (subprogramEntry != NULL && getFullFrameInfo) {
		// create function parameter objects
		for (DebugInfoEntryList::ConstIterator it
			= subprogramEntry->Parameters().GetIterator();
			DebugInfoEntry* entry = it.Next();) {
			if (entry->Tag() != DW_TAG_formal_parameter)
				continue;

			BString parameterName;
			DwarfUtils::GetDIEName(entry, parameterName);
			if (parameterName.Length() == 0)
				continue;

			DIEFormalParameter* parameterEntry
				= dynamic_cast<DIEFormalParameter*>(entry);
			Variable* parameter;
			if (stackFrameDebugInfo->CreateParameter(functionID,
				parameterEntry, parameter) != B_OK) {
				continue;
			}
			BReference<Variable> parameterReference(parameter, true);

			if (!frame->AddParameter(parameter))
				return B_NO_MEMORY;
		}

		// create objects for the local variables
		_CreateLocalVariables(unit, frame, functionID, *stackFrameDebugInfo,
			instructionPointer, functionInstance->Address() - fRelocationDelta,
			subprogramEntry->Variables(), subprogramEntry->Blocks());

		if (returnValueInfos != NULL && !returnValueInfos->IsEmpty()) {
			_CreateReturnValues(returnValueInfos, image, frame,
				*stackFrameDebugInfo);
		}
	}

	_frame = frameReference.Detach();
	_previousCpuState = previousCpuStateReference.Detach();

	frame->SetPreviousCpuState(_previousCpuState);

	return B_OK;
}


status_t
DwarfImageDebugInfo::GetStatement(FunctionDebugInfo* _function,
	target_addr_t address, Statement*& _statement)
{
	TRACE_CODE("DwarfImageDebugInfo::GetStatement(function: %p, address: %#"
		B_PRIx64 ")\n", _function, address);

	DwarfFunctionDebugInfo* function
		= dynamic_cast<DwarfFunctionDebugInfo*>(_function);
	if (function == NULL) {
		TRACE_LINES("  -> no dwarf function\n");
		// fall back to assembly
		return fArchitecture->GetStatement(function, address, _statement);
	}

	AutoLocker<BLocker> locker(fLock);

	// check whether we have the source code
	CompilationUnit* unit = function->GetCompilationUnit();
	LocatableFile* file = function->SourceFile();
	if (file == NULL) {
		TRACE_CODE("  -> no source file\n");

		// no source code -- rather return the assembly statement
		return fArchitecture->GetStatement(function, address, _statement);
	}

	SourceCode* sourceCode = NULL;
	status_t error = fSourceInfo->GetActiveSourceCode(_function, sourceCode);
	BReference<SourceCode> sourceReference(sourceCode, true);
	if (error != B_OK || dynamic_cast<DisassembledCode*>(sourceCode) != NULL) {
		// either no source code or disassembly is currently active (i.e.
		// due to failing to locate the source file on disk or the user
		// deliberately switching to disassembly view).
		// return the assembly statement.
		return fArchitecture->GetStatement(function, address, _statement);
	}

	// get the index of the source file in the compilation unit for cheaper
	// comparison below
	int32 fileIndex = _GetSourceFileIndex(unit, file);

	// Get the statement by executing the line number program for the
	// compilation unit.
	LineNumberProgram& program = unit->GetLineNumberProgram();
	if (!program.IsValid()) {
		TRACE_CODE("  -> no line number program\n");
		return B_BAD_DATA;
	}

	// adjust address
	address -= fRelocationDelta;

	LineNumberProgram::State state;
	program.GetInitialState(state);

	target_addr_t statementAddress = 0;
	int32 statementLine = -1;
	int32 statementColumn = -1;
	while (program.GetNextRow(state)) {
		// skip statements of other files
		if (state.file != fileIndex)
			continue;

		if (statementAddress != 0
			&& (state.isStatement || state.isSequenceEnd)) {
			target_addr_t endAddress = state.address;
			if (address >= statementAddress && address < endAddress) {
				ContiguousStatement* statement = new(std::nothrow)
					ContiguousStatement(
						SourceLocation(statementLine, statementColumn),
						TargetAddressRange(fRelocationDelta + statementAddress,
							endAddress - statementAddress));
				if (statement == NULL)
					return B_NO_MEMORY;

				_statement = statement;
				return B_OK;
			}

			statementAddress = 0;
		}

		if (state.isStatement) {
			statementAddress = state.address;
			statementLine = state.line - 1;
			// discard column info until proper support is implemented
			// statementColumn = std::max(state.column - 1, (int32)0);
			statementColumn = 0;
		}
	}

	TRACE_CODE("  -> no line number program match\n");
	return B_ENTRY_NOT_FOUND;
}


status_t
DwarfImageDebugInfo::GetStatementAtSourceLocation(FunctionDebugInfo* _function,
	const SourceLocation& sourceLocation, Statement*& _statement)
{
	DwarfFunctionDebugInfo* function
		= dynamic_cast<DwarfFunctionDebugInfo*>(_function);
	if (function == NULL)
		return B_BAD_VALUE;

	target_addr_t functionStartAddress = function->Address() - fRelocationDelta;
	target_addr_t functionEndAddress = functionStartAddress + function->Size();

	TRACE_LINES2("DwarfImageDebugInfo::GetStatementAtSourceLocation(%p, "
		"(%" B_PRId32 ", %" B_PRId32 ")): function range: %#" B_PRIx64 " - %#"
		B_PRIx64 "\n", function, sourceLocation.Line(), sourceLocation.Column(),
		functionStartAddress, functionEndAddress);

	AutoLocker<BLocker> locker(fLock);

	// get the source file
	LocatableFile* file = function->SourceFile();
	if (file == NULL)
		return B_ENTRY_NOT_FOUND;

	CompilationUnit* unit = function->GetCompilationUnit();

	// get the index of the source file in the compilation unit for cheaper
	// comparison below
	int32 fileIndex = _GetSourceFileIndex(unit, file);

	// Get the statement by executing the line number program for the
	// compilation unit.
	LineNumberProgram& program = unit->GetLineNumberProgram();
	if (!program.IsValid())
		return B_BAD_DATA;

	LineNumberProgram::State state;
	program.GetInitialState(state);

	target_addr_t statementAddress = 0;
	int32 statementLine = -1;
	int32 statementColumn = -1;
	while (program.GetNextRow(state)) {
		bool isOurFile = state.file == fileIndex;

		if (statementAddress != 0
			&& (!isOurFile || state.isStatement || state.isSequenceEnd)) {
			target_addr_t endAddress = state.address;

			if (statementAddress < endAddress) {
				TRACE_LINES2("  statement: %#" B_PRIx64 " - %#" B_PRIx64
					", location: (%" B_PRId32 ", %" B_PRId32 ")\n",
					statementAddress, endAddress, statementLine,
				 	statementColumn);
			}

			if (statementAddress < endAddress
				&& statementAddress >= functionStartAddress
				&& statementAddress < functionEndAddress
				&& statementLine == (int32)sourceLocation.Line()
				&& statementColumn == (int32)sourceLocation.Column()) {
				TRACE_LINES2("  -> found statement!\n");

				ContiguousStatement* statement = new(std::nothrow)
					ContiguousStatement(
						SourceLocation(statementLine, statementColumn),
						TargetAddressRange(fRelocationDelta + statementAddress,
							endAddress - statementAddress));
				if (statement == NULL)
					return B_NO_MEMORY;

				_statement = statement;
				return B_OK;
			}

			statementAddress = 0;
		}

		// skip statements of other files
		if (!isOurFile)
			continue;

		if (state.isStatement) {
			statementAddress = state.address;
			statementLine = state.line - 1;
			// discard column info until proper support is implemented
			// statementColumn = std::max(state.column - 1, (int32)0);
			statementColumn = 0;
		}
	}

	return B_ENTRY_NOT_FOUND;
}


status_t
DwarfImageDebugInfo::GetSourceLanguage(FunctionDebugInfo* _function,
	SourceLanguage*& _language)
{
	DwarfFunctionDebugInfo* function
		= dynamic_cast<DwarfFunctionDebugInfo*>(_function);
	if (function == NULL)
		return B_BAD_VALUE;

	SourceLanguage* language;
	CompilationUnit* unit = function->GetCompilationUnit();
	switch (unit->UnitEntry()->Language()) {
		case DW_LANG_C89:
		case DW_LANG_C:
		case DW_LANG_C99:
			language = new(std::nothrow) CLanguage;
			break;
		case DW_LANG_C_plus_plus:
			language = new(std::nothrow) CppLanguage;
			break;
		case 0:
		default:
			language = new(std::nothrow) UnsupportedLanguage;
			break;
	}

	if (language == NULL)
		return B_NO_MEMORY;

	_language = language;
	return B_OK;
}


ssize_t
DwarfImageDebugInfo::ReadCode(target_addr_t address, void* buffer, size_t size)
{
	target_addr_t offset = address - fRelocationDelta
		- fTextSegment->LoadAddress() + fTextSegment->FileOffset();
	ssize_t bytesRead = pread(fFile->GetElfFile()->FD(), buffer, size, offset);
	return bytesRead >= 0 ? bytesRead : errno;
}


status_t
DwarfImageDebugInfo::AddSourceCodeInfo(LocatableFile* file,
	FileSourceCode* sourceCode)
{
	bool addedAny = false;
	for (int32 i = 0; CompilationUnit* unit = fFile->CompilationUnitAt(i);
			i++) {
		int32 fileIndex = _GetSourceFileIndex(unit, file);
		if (fileIndex < 0)
			continue;

		status_t error = _AddSourceCodeInfo(unit, sourceCode, fileIndex);
		if (error == B_NO_MEMORY)
			return error;
		addedAny |= error == B_OK;
	}

	return addedAny ? B_OK : B_ENTRY_NOT_FOUND;
}


status_t
DwarfImageDebugInfo::_AddSourceCodeInfo(CompilationUnit* unit,
	FileSourceCode* sourceCode, int32 fileIndex)
{
	// Get the statements by executing the line number program for the
	// compilation unit and filtering the rows for our source file.
	LineNumberProgram& program = unit->GetLineNumberProgram();
	if (!program.IsValid())
		return B_BAD_DATA;

	LineNumberProgram::State state;
	program.GetInitialState(state);

	target_addr_t statementAddress = 0;
	int32 statementLine = -1;
	int32 statementColumn = -1;
	while (program.GetNextRow(state)) {
		TRACE_LINES2("  %#" B_PRIx64 "  (%" B_PRId32 ", %" B_PRId32 ", %"
			B_PRId32 ")  %d\n", state.address, state.file, state.line,
			state.column, state.isStatement);

		bool isOurFile = state.file == fileIndex;

		if (statementAddress != 0
			&& (!isOurFile || state.isStatement || state.isSequenceEnd)) {
			target_addr_t endAddress = state.address;
			if (endAddress > statementAddress) {
				// add the statement
				status_t error = sourceCode->AddSourceLocation(
					SourceLocation(statementLine, statementColumn));
				if (error != B_OK)
					return error;

				TRACE_LINES2("  -> statement: %#" B_PRIx64 " - %#" B_PRIx64
					", source location: (%" B_PRId32 ", %" B_PRId32 ")\n",
					statementAddress, endAddress, statementLine,
				 	statementColumn);
			}

			statementAddress = 0;
		}

		// skip statements of other files
		if (!isOurFile)
			continue;

		if (state.isStatement) {
			statementAddress = state.address;
			statementLine = state.line - 1;
			// discard column info until proper support is implemented
			// statementColumn = std::max(state.column - 1, (int32)0);
			statementColumn = 0;
		}
	}

	return B_OK;
}


int32
DwarfImageDebugInfo::_GetSourceFileIndex(CompilationUnit* unit,
	LocatableFile* sourceFile) const
{
	// get the index of the source file in the compilation unit for cheaper
	// comparison below
	const char* directory;
	for (int32 i = 0; const char* fileName = unit->FileAt(i, &directory); i++) {
		LocatableFile* file = fFileManager->GetSourceFile(directory, fileName);
		if (file != NULL) {
			BReference<LocatableFile> fileReference(file, true);
			if (file == sourceFile) {
				return i + 1;
					// indices are one-based
			}
		}
	}

	return -1;
}


status_t
DwarfImageDebugInfo::_CreateLocalVariables(CompilationUnit* unit,
	StackFrame* frame, FunctionID* functionID,
	DwarfStackFrameDebugInfo& factory, target_addr_t instructionPointer,
	target_addr_t lowPC, const EntryListWrapper& variableEntries,
	const EntryListWrapper& blockEntries)
{
	TRACE_LOCALS("DwarfImageDebugInfo::_CreateLocalVariables(): ip: %#" B_PRIx64
		", low PC: %#" B_PRIx64 "\n", instructionPointer, lowPC);

	// iterate through the variables and add the ones in scope
	for (DebugInfoEntryList::ConstIterator it
			= variableEntries.list.GetIterator();
		DIEVariable* variableEntry = dynamic_cast<DIEVariable*>(it.Next());) {

		TRACE_LOCALS("  variableEntry %p, scope start: %" B_PRIu64 "\n",
			variableEntry, variableEntry->StartScope());

		// check the variable's scope
		if (instructionPointer < lowPC + variableEntry->StartScope())
			continue;

		// add the variable
		Variable* variable;
		if (factory.CreateLocalVariable(functionID, variableEntry, variable)
				!= B_OK) {
			continue;
		}
		BReference<Variable> variableReference(variable, true);

		if (!frame->AddLocalVariable(variable))
			return B_NO_MEMORY;
	}

	// iterate through the blocks and find the one we're currently in (if any)
	for (DebugInfoEntryList::ConstIterator it = blockEntries.list.GetIterator();
		DIELexicalBlock* block = dynamic_cast<DIELexicalBlock*>(it.Next());) {

		TRACE_LOCALS("  lexical block: %p\n", block);

		// check whether the block has low/high PC attributes
		if (block->LowPC() != 0) {
			TRACE_LOCALS("    has lowPC\n");

			// yep, compare with the instruction pointer
			if (instructionPointer < block->LowPC()
				|| instructionPointer >= block->HighPC()) {
				continue;
			}
		} else {
			TRACE_LOCALS("    no lowPC\n");

			// check the address ranges instead
			TargetAddressRangeList* rangeList = fFile->ResolveRangeList(unit,
				block->AddressRangesOffset());
			if (rangeList == NULL) {
				TRACE_LOCALS("    failed to get ranges\n");
				continue;
			}
			BReference<TargetAddressRangeList> rangeListReference(rangeList,
				true);

			if (!rangeList->Contains(instructionPointer)) {
				TRACE_LOCALS("    ranges don't contain IP\n");
				continue;
			}
		}

		// found a block -- recurse
		return _CreateLocalVariables(unit, frame, functionID, factory,
			instructionPointer, lowPC, block->Variables(), block->Blocks());
	}

	return B_OK;
}


status_t
DwarfImageDebugInfo::_CreateReturnValues(ReturnValueInfoList* returnValueInfos,
	Image* image, StackFrame* frame, DwarfStackFrameDebugInfo& factory)
{
	for (int32 i = 0; i < returnValueInfos->CountItems(); i++) {
		Image* targetImage = image;
		ReturnValueInfo* valueInfo = returnValueInfos->ItemAt(i);
		target_addr_t subroutineAddress = valueInfo->SubroutineAddress();
		CpuState* subroutineState = valueInfo->State();
		if (!targetImage->ContainsAddress(subroutineAddress)) {
			// our current image doesn't contain the target function,
			// locate the one which does.
			targetImage = image->GetTeam()->ImageByAddress(subroutineAddress);
			if (targetImage == NULL) {
				// nothing we can do, try the next entry (if any)
				continue;
			}
		}

		status_t result = B_OK;
		ImageDebugInfo* imageInfo = targetImage->GetImageDebugInfo();
		if (imageInfo == NULL) {
			// the subroutine may have resolved to a different image
			// that doesn't have debug information available.
			continue;
		}

		FunctionInstance* targetFunction;
		if (imageInfo->GetAddressSectionType(subroutineAddress)
				== ADDRESS_SECTION_TYPE_PLT) {
			result = fArchitecture->ResolvePICFunctionAddress(
				subroutineAddress, subroutineState, subroutineAddress);
			if (result != B_OK)
				continue;
			if (!targetImage->ContainsAddress(subroutineAddress)) {
				// the PLT entry doesn't necessarily point to a function
				// in the same image; as such we may need to try to
				// resolve the target address again.
				targetImage = image->GetTeam()->ImageByAddress(
					subroutineAddress);
				if (targetImage == NULL)
					continue;
				imageInfo = targetImage->GetImageDebugInfo();
				if (imageInfo == NULL) {
					// As above, since the indirection here may have
					// landed us in an entirely different image, there is
					// no guarantee that debug info is available,
					// depending on which image it was.
					continue;
				}

			}
		}

		targetFunction = imageInfo->FunctionAtAddress(subroutineAddress);
		if (targetFunction != NULL) {
			DwarfFunctionDebugInfo* targetInfo =
				dynamic_cast<DwarfFunctionDebugInfo*>(
					targetFunction->GetFunctionDebugInfo());
			if (targetInfo != NULL) {
				DIESubprogram* subProgram = targetInfo->SubprogramEntry();
				DIEType* returnType = subProgram->ReturnType();
				if (returnType == NULL) {
					// check if we have a specification, and if so, if that has
					// a return type
					subProgram = dynamic_cast<DIESubprogram*>(
						subProgram->Specification());
					if (subProgram != NULL)
						returnType = subProgram->ReturnType();

					// function doesn't return a value, we're done.
					if (returnType == NULL)
						return B_OK;
				}

				uint32 byteSize = 0;
				if (returnType->ByteSize() == NULL) {
					if (dynamic_cast<DIEAddressingType*>(returnType) != NULL)
						byteSize = fArchitecture->AddressSize();
				} else
					byteSize = returnType->ByteSize()->constant;

				// if we were unable to determine a size for the type,
				// simply default to the architecture's register width.
				if (byteSize == 0)
					byteSize = fArchitecture->AddressSize();

				ValueLocation* location;
				result = fArchitecture->GetReturnAddressLocation(frame,
					byteSize, location);
				if (result != B_OK)
					return result;

				BReference<ValueLocation> locationReference(location, true);
				Variable* variable = NULL;
				BReference<FunctionID> idReference(
					targetFunction->GetFunctionID(), true);
				result = factory.CreateReturnValue(idReference, returnType,
					location, subroutineState, variable);
				if (result != B_OK)
					return result;

				BReference<Variable> variableReference(variable, true);
				if (!frame->AddLocalVariable(variable))
					return B_NO_MEMORY;
			}
		}
	}

	return B_OK;
}


bool
DwarfImageDebugInfo::_EvaluateBaseTypeConstraints(DIEType* type,
	const TypeLookupConstraints& constraints) const
{
	if (constraints.HasBaseTypeName()) {
		BString baseEntryName;
		DIEType* baseTypeOwnerEntry = NULL;

		switch (constraints.TypeKind()) {
			case TYPE_ADDRESS:
			{
				DIEAddressingType* addressType =
					dynamic_cast<DIEAddressingType*>(type);
				if (addressType != NULL) {
					baseTypeOwnerEntry = DwarfUtils::GetDIEByPredicate(
						addressType, HasTypePredicate<DIEAddressingType>());
				}
				break;
			}
			case TYPE_ARRAY:
			{
				DIEArrayType* arrayType =
					dynamic_cast<DIEArrayType*>(type);
				if (arrayType != NULL) {
					baseTypeOwnerEntry = DwarfUtils::GetDIEByPredicate(
						arrayType, HasTypePredicate<DIEArrayType>());
				}
				break;
			}
			default:
				break;
		}

		if (baseTypeOwnerEntry != NULL) {
			DwarfUtils::GetFullyQualifiedDIEName(baseTypeOwnerEntry,
				baseEntryName);

			if (baseEntryName != constraints.BaseTypeName())
				return false;
		}
	}

	return true;
}


status_t
DwarfImageDebugInfo::_RecursiveTraverseNamespaceForFunctions(
	DIENamespace* nsEntry, CompilationUnit* unit,
	BObjectList<FunctionDebugInfo>& functions)
{
	status_t error = B_OK;
	for (DebugInfoEntryList::ConstIterator it
				= nsEntry->Children().GetIterator();
			DebugInfoEntry* entry = it.Next();) {
		if (entry->Tag() == DW_TAG_subprogram) {
			DIESubprogram* subprogramEntry
				= static_cast<DIESubprogram*>(entry);
			error = _AddFunction(subprogramEntry, unit, functions);
			if (error != B_OK)
				return error;
		}

		DIENamespace* nsEntry = dynamic_cast<DIENamespace*>(entry);
		if (nsEntry != NULL) {
			error = _RecursiveTraverseNamespaceForFunctions(nsEntry, unit,
				functions);
			if (error != B_OK)
				return error;
			continue;
		}

		DIEClassBaseType* classEntry = dynamic_cast<DIEClassBaseType*>(entry);
		if (classEntry != NULL) {
			for (DebugInfoEntryList::ConstIterator it
						= classEntry->MemberFunctions().GetIterator();
					DebugInfoEntry* memberEntry = it.Next();) {
				error = _AddFunction(static_cast<DIESubprogram*>(memberEntry),
					unit, functions);
				if (error != B_OK)
					return error;
			}
		}
	}

	return B_OK;
}


status_t
DwarfImageDebugInfo::_AddFunction(DIESubprogram* subprogramEntry,
	CompilationUnit* unit, BObjectList<FunctionDebugInfo>& functions)
{
	// ignore declarations and inlined functions
	if (subprogramEntry->IsDeclaration()
		|| subprogramEntry->Inline() == DW_INL_inlined
		|| subprogramEntry->Inline() == DW_INL_declared_inlined) {
		return B_OK;
	}

	// get the name
	BString name;
	DwarfUtils::GetFullyQualifiedDIEName(subprogramEntry, name);
	if (name.Length() == 0)
		return B_OK;

	// get the address ranges
	TargetAddressRangeList* rangeList = fFile->ResolveRangeList(unit,
		subprogramEntry->AddressRangesOffset());
	if (rangeList == NULL) {
		target_addr_t lowPC = subprogramEntry->LowPC();
		target_addr_t highPC = subprogramEntry->HighPC();
		if (highPC <= lowPC)
			return B_OK;

		rangeList = new(std::nothrow) TargetAddressRangeList(
			TargetAddressRange(lowPC, highPC - lowPC));
		if (rangeList == NULL)
			return B_NO_MEMORY;
				// TODO: Clean up already added functions!
	}
	BReference<TargetAddressRangeList> rangeListReference(rangeList,
		true);

	// get the source location
	const char* directoryPath = NULL;
	const char* fileName = NULL;
	int32 line = -1;
	int32 column = -1;
	DwarfUtils::GetDeclarationLocation(fFile, subprogramEntry,
		directoryPath, fileName, line, column);

	LocatableFile* file = NULL;
	if (fileName != NULL) {
		file = fFileManager->GetSourceFile(directoryPath,
			fileName);
	}
	BReference<LocatableFile> fileReference(file, true);

	// create and add the functions
	DwarfFunctionDebugInfo* function
		= new(std::nothrow) DwarfFunctionDebugInfo(this, unit,
			subprogramEntry, rangeList, name, file,
			SourceLocation(line, std::max(column, (int32)0)));
	if (function == NULL || !functions.AddItem(function)) {
		delete function;
		return B_NO_MEMORY;
			// TODO: Clean up already added functions!
	}

	return B_OK;
}


status_t
DwarfImageDebugInfo::_BuildTypeNameTable()
{
	fTypeNameTable = new(std::nothrow) TypeNameTable;
	if (fTypeNameTable == NULL)
		return B_NO_MEMORY;

	status_t error = fTypeNameTable->Init();
	if (error != B_OK)
		return error;

	// iterate through all compilation units
	for (int32 i = 0; CompilationUnit* unit = fFile->CompilationUnitAt(i);
		i++) {
		// iterate through all types of the compilation unit
		for (DebugInfoEntryList::ConstIterator it
				= unit->UnitEntry()->Types().GetIterator();
			DIEType* typeEntry = dynamic_cast<DIEType*>(it.Next());) {

			if (_RecursiveAddTypeNames(typeEntry, unit) != B_OK)
				return B_NO_MEMORY;
		}

		for (DebugInfoEntryList::ConstIterator it
			= unit->UnitEntry()->OtherChildren().GetIterator();
			DebugInfoEntry* child = it.Next();) {
			DIENamespace* namespaceEntry = dynamic_cast<DIENamespace*>(child);
			if (namespaceEntry == NULL)
				continue;

			if (_RecursiveTraverseNamespaceForTypes(namespaceEntry, unit)
					!= B_OK) {
				return B_NO_MEMORY;
			}
		}
	}

	return B_OK;
}


status_t
DwarfImageDebugInfo::_RecursiveAddTypeNames(DIEType* type, CompilationUnit* unit)
{
	if (type->IsDeclaration())
		return B_OK;

	BString typeEntryName;
	DwarfUtils::GetFullyQualifiedDIEName(type, typeEntryName);

	status_t error = B_OK;
	TypeNameEntry* entry = fTypeNameTable->Lookup(typeEntryName);
	if (entry == NULL) {
		entry = new(std::nothrow) TypeNameEntry(typeEntryName);
		if (entry == NULL)
			return B_NO_MEMORY;

		error = fTypeNameTable->Insert(entry);
		if (error != B_OK)
			return error;
	}

	TypeEntryInfo* info = new(std::nothrow) TypeEntryInfo(type,	unit);
	if (info == NULL)
		return B_NO_MEMORY;

	if (!entry->types.AddItem(info)) {
		delete info;
		return B_NO_MEMORY;
	}

	DIEClassBaseType* classType = dynamic_cast<DIEClassBaseType*>(type);
	if (classType == NULL)
		return B_OK;

	for (DebugInfoEntryList::ConstIterator it
			= classType->InnerTypes().GetIterator();
		DIEType* innerType = dynamic_cast<DIEType*>(it.Next());) {
		error = _RecursiveAddTypeNames(innerType, unit);
		if (error != B_OK)
			return error;
	}

	return B_OK;
}


status_t
DwarfImageDebugInfo::_RecursiveTraverseNamespaceForTypes(DIENamespace* nsEntry,
	CompilationUnit* unit)
{
	for (DebugInfoEntryList::ConstIterator it
				= nsEntry->Children().GetIterator();
			DebugInfoEntry* child = it.Next();) {

		if (child->IsType()) {
			DIEType* type = dynamic_cast<DIEType*>(child);
			if (_RecursiveAddTypeNames(type, unit) != B_OK)
				return B_NO_MEMORY;
		} else {
			DIENamespace* nameSpace = dynamic_cast<DIENamespace*>(child);
			if (nameSpace == NULL)
				continue;

			status_t error = _RecursiveTraverseNamespaceForTypes(nameSpace,
				unit);
			if (error != B_OK)
				return error;
			continue;
		}
	}

	return B_OK;
}
