#pragma once

#include <stack>

#include "Scene/GameObject.hpp" // For Terminal::MAX_OUTPUT_COUNT
#include "Variant.hpp"
#include "VirtualMachine/Backend/FunctionBindings.hpp"
#include "VirtualMachine/Backend/IR.hpp"

namespace flex
{
	struct DiagnosticContainer;
	class FunctionBindings;

	namespace IR
	{
		struct Assignment;
		struct Value;
		struct IntermediateRepresentation;
		enum class OperatorType;
	}

	namespace AST
	{
		struct AST;
	}

	namespace VM
	{
		struct VariantWrapper
		{
			enum class Type
			{
				REGISTER,
				CONSTANT,
				TERMINAL_OUTPUT,
				NONE
			};

			VariantWrapper() :
				type(Type::NONE),
				variant(g_EmptyVariant)
			{}

			VariantWrapper(Type type, const Variant& variant) :
				type(type),
				variant(variant)
			{}

			Type type;
			Variant variant;

			Variant& Get(VirtualMachine* vm);
			Variant& GetW(VirtualMachine* vm);
			bool Valid() const;
			std::string ToString() const;
		};

		enum class OpCode
		{
			MOV,
			ADD,
			SUB,
			MUL,
			DIV,
			MOD,
			INV,
			AND,
			OR,
			XOR,
			ITF, // int-to-float cast
			FTI, // float-to-int cast
			CALL,
			PUSH,
			POP,
			CMP,
			JMP,
			JZ,
			JNZ,
			JEQ,
			JNE,
			JLT,
			JLE,
			JGT,
			JGE,
			YIELD,
			RETURN,
			HALT,

			_NONE
		};

		static const char* s_OpCodeStrings[] =
		{
			"mov",
			"add",
			"sub",
			"mul",
			"div",
			"inv",
			"mod",
			"and",
			"or ",
			"xor",
			"itf",
			"fti",
			"cal",
			"psh",
			"pop",
			"cmp",
			"jmp",
			"jz",
			"jnz",
			"jeq",
			"jne",
			"jlt",
			"jle",
			"jgt",
			"jge",
			"yld",
			"rtn",
			"hlt",

			"NONE"
		};

		static_assert(ARRAY_LENGTH(s_OpCodeStrings) == ((size_t)OpCode::_NONE + 1), "Length of s_OpCodeStrings must match length of OpCode enum");

		const char* OpCodeToString(OpCode opCode);

		OpCode BinaryOperatorTypeToOpCode(IR::BinaryOperatorType opType);
		OpCode BinaryOperatorTypeToInverseOpCode(IR::BinaryOperatorType opType);
		OpCode InverseOpCode(OpCode opCode);

		// TODO: Delete:
		//OpCode IROperatorTypeToOpCode(IR::OperatorType irOperatorType);
		//OpCode BinaryOperatorTypeToOpCode(AST::BinaryOperatorType operatorType);
		//OpCode BinaryOpToJumpCode(AST::BinaryOperatorType operatorType);

		//template<typename Ret, typename... Ts>
		//struct FuncPtr
		//{
		//	typedef Ret(*PtrType)(Ts...);

		//	FuncPtr(PtrType ptr) : ptr(ptr) {}

		//	Ret operator()()
		//	{
		//		return ptr();
		//	}

		//	PtrType ptr;
		//};

		struct Instruction
		{
			Instruction(OpCode opCode) :
				opCode(opCode),
				val0(VariantWrapper::Type::CONSTANT, g_EmptyVariant),
				val1(VariantWrapper::Type::CONSTANT, g_EmptyVariant),
				val2(VariantWrapper::Type::CONSTANT, g_EmptyVariant)
			{}

			Instruction(OpCode opCode, const VariantWrapper& val0) :
				opCode(opCode),
				val0(val0),
				val1(VariantWrapper::Type::CONSTANT, g_EmptyVariant),
				val2(VariantWrapper::Type::CONSTANT, g_EmptyVariant)
			{}

			Instruction(OpCode opCode, const VariantWrapper& val0, const VariantWrapper& val1) :
				opCode(opCode),
				val0(val0),
				val1(val1),
				val2(VariantWrapper::Type::CONSTANT, g_EmptyVariant)
			{}

			Instruction(OpCode opCode, const VariantWrapper& val0, const VariantWrapper& val1, const VariantWrapper& val2) :
				opCode(opCode),
				val0(val0),
				val1(val1),
				val2(val2)
			{}

			OpCode opCode;
			VariantWrapper val0;
			VariantWrapper val1;
			VariantWrapper val2;
		};

		struct IntermediateFuncAddress
		{
			i32 uid;
			i32 ip;
		};

		struct State;

		struct InstructionBlock
		{
			void PushBack(const Instruction& inst, Span origin, State* state);

			std::vector<Instruction> instructions;
			std::vector<Span> instructionOrigins;

			i32 startOffset = -1;
		};

		struct State
		{
			State();
			void Clear();
			void Destroy();
			InstructionBlock& CurrentInstructionBlock();
			InstructionBlock& PushInstructionBlock();

			std::map<std::string, IR::Assignment> varUsages;
			std::map<std::string, i32> funcNameToBlockIndexTable;
			std::map<std::string, i32> varRegisterMap;
			std::map<std::string, IR::Value::Type> tmpVarTypes;
			std::vector<InstructionBlock> instructionBlocks;

			DiagnosticContainer* diagnosticContainer = nullptr;
		};

		class VirtualMachine
		{
		public:
			VirtualMachine();
			~VirtualMachine();

			// _ Syntax rules _
			// Vars must be declared (with type) and given an initial value prior to usage
			// Functions are global and can be called prior to definition

			// _ Calling convention _
			// Caller pushes registers onto stack
			// Caller pushes return address onto stack
			// Caller pushes args in reverse order onto stack
			// Called runs
			//   Called pops args first to last
			//   Called returns (with optional value)
			//   Resume address is popped off stack
			//   Return value is pushed onto stack
			// Caller resumes
			// Caller pops return value off stack (if non-void func)
			// Caller pops registers off stack

			void GenerateFromSource(const char* source);
			void GenerateFromIR(IR::IntermediateRepresentation* ir, FunctionBindings* functionBindings);
			void GenerateFromInstStream(const std::vector<Instruction>& inInstructions);

			void Execute(bool bSingleStep = false);

			DiagnosticContainer* GetDiagnosticContainer();

			bool IsExecuting() const;
			i32 InstructionIndex() const;
			i32 CurrentLineNumber() const;
			void ClearRuntimeState();

			bool ZeroFlagSet() const;
			bool SignFlagSet() const;

			bool IsCompiled() const;

			static const i32 REGISTER_COUNT = 32;
			static const i32 MAX_STACK_HEIGHT = 2048;
			static const u32 MEMORY_POOL_SIZE = 32768;

			static VariantWrapper g_ZeroIntVariantWrapper;
			static VariantWrapper g_ZeroFloatVariantWrapper;
			static VariantWrapper g_OneIntVariantWrapper;
			static VariantWrapper g_OneFloatVariantWrapper;

			struct RunningState
			{
				void Clear()
				{
					instructionIdx = -1;
					terminated = false;
					zf = 0;
					sf = 0;
				}

				i32 instructionIdx = -1;
				bool terminated = false;
				// Flags bitfield
				u32 zf : 1, sf : 1;
			};

			std::vector<Instruction> instructions;
			std::vector<Span> instructionOrigins;

			std::array<Variant, REGISTER_COUNT> registers;
			std::array<Variant, Terminal::MAX_OUTPUT_COUNT> terminalOutputs;
			std::stack<Variant> stack;

			u32* memory = nullptr;

			State* state = nullptr;
			DiagnosticContainer* diagnosticContainer = nullptr;

			std::string astStr;
			std::string irStr;
			std::string unpatchedInstructionStr;
			std::string instructionStr;

			static bool IsTerminalOutputVar(const std::string& varName);
			// Returns -1, or terminal output var index if valid name
			static i32 GetTerminalOutputVar(const std::string& varName);

			RunningState m_RunningState;

		private:
			IR::Value::Type FindIRType(IR::State* irState, IR::Value* irValue);

			void AllocateMemory();
			void ZeroOutRegisters();
			void ZeroOutTerminalOutputs();
			void ClearStack();
			void HandleComparison(VariantWrapper& regVal, IR::IntermediateRepresentation* ir, IR::BinaryValue* binaryValue);
			void HandleComparison(IR::IntermediateRepresentation* ir, IR::Value* condition, i32 ifTrueBlockIndex, i32 ifFalseBlockIndex, bool bInvCondition);

			bool IsExternal(FuncAddress funcAddress);
			FuncAddress GetExternalFuncAddress(const std::string& functionName);
			i32 TranslateLocalFuncAddress(FuncAddress localFuncAddress);
			bool DispatchExternalCall(FuncAddress funcAddress);

			VariantWrapper GetValueWrapperFromIRValue(IR::State* irState, IR::Value* value);
			i32 CombineInstructionIndex(i32 instructionBlockIndex, i32 instructionIndex);
			void SplitInstructionIndex(i32 combined, i32& outInstructionBlockIndex, i32& outInstructionIndex);
			i32 GenerateCallInstruction(IR::State* irState, IR::FunctionCallValue* funcCallValue);

			AST::AST* m_AST = nullptr;
			IR::IntermediateRepresentation* m_IR = nullptr;
			FunctionBindings* m_FunctionBindings = nullptr;

			bool m_bCompiled = false;

		};
	}// namespace VM
} // namespace flex
