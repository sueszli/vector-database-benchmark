// SPDX-License-Identifier: MPL-2.0
// Copyright © 2020 Skyline Team and Contributors (https://github.com/skyline-emu/)

#include "macro_state.h"
#include "soc/gm20b/engines/engine.h"
#include "macro_interpreter.h"

namespace skyline::soc::gm20b::engine {
    MacroInterpreter::MacroInterpreter(span<u32> macroCode) : macroCode{macroCode} {}

    void MacroInterpreter::Execute(size_t offset, span<u32> args, MacroEngineBase *targetEngine) {
        // Reset the interpreter state
        engine = targetEngine;
        opcode = reinterpret_cast<Opcode *>(&macroCode[offset]);
        registers = {};
        argument = args.data();
        methodAddress.raw = 0;
        carryFlag = false;

        // The first argument is stored in register 1
        registers[1] = *argument++;

        while (Step());
    }

    __attribute__((always_inline)) bool MacroInterpreter::Step(Opcode *delayedOpcode) {
        switch (opcode->operation) {
            case Opcode::Operation::AluRegister: {
                u32 result{HandleAlu(opcode->aluOperation, registers[opcode->srcA], registers[opcode->srcB])};

                HandleAssignment(opcode->assignmentOperation, opcode->dest, result);
                break;
            }

            case Opcode::Operation::AddImmediate:
                HandleAssignment(opcode->assignmentOperation, opcode->dest,
                                 static_cast<u32>(static_cast<i32>(registers[opcode->srcA]) + opcode->immediate));
                break;

            case Opcode::Operation::BitfieldReplace: {
                u32 src{registers[opcode->srcB]};
                u32 dest{registers[opcode->srcA]};

                // Extract the source region
                src = (src >> opcode->bitfield.srcBit) & opcode->bitfield.GetMask();

                // Mask out the bits that we will replace
                dest &= ~(opcode->bitfield.GetMask() << opcode->bitfield.destBit);

                // Replace the bitfield region in the destination with the region from the source
                dest |= src << opcode->bitfield.destBit;

                HandleAssignment(opcode->assignmentOperation, opcode->dest, dest);
                break;
            }

            case Opcode::Operation::BitfieldExtractShiftLeftImmediate: {
                u32 src{registers[opcode->srcB]};
                u32 dest{registers[opcode->srcA]};

                u32 result{((src >> dest) & opcode->bitfield.GetMask()) << opcode->bitfield.destBit};

                HandleAssignment(opcode->assignmentOperation, opcode->dest, result);
                break;
            }

            case Opcode::Operation::BitfieldExtractShiftLeftRegister: {
                u32 src{registers[opcode->srcB]};
                u32 dest{registers[opcode->srcA]};

                u32 result{((src >> opcode->bitfield.srcBit) & opcode->bitfield.GetMask()) << dest};

                HandleAssignment(opcode->assignmentOperation, opcode->dest, result);
                break;
            }

            case Opcode::Operation::ReadImmediate: {
                u32 result{engine->ReadMethodFromMacro(static_cast<u32>(static_cast<i32>(registers[opcode->srcA]) + opcode->immediate))};
                HandleAssignment(opcode->assignmentOperation, opcode->dest, result);
                break;
            }

            case Opcode::Operation::Branch: {
                if (delayedOpcode != nullptr)
                    throw exception("Cannot branch while inside a delay slot");

                u32 value{registers[opcode->srcA]};
                bool branch{(opcode->branchCondition == Opcode::BranchCondition::Zero) ? (value == 0) : (value != 0)};

                if (branch) {
                    if (opcode->noDelay) {
                        opcode += opcode->immediate;
                        return true;
                    } else {
                        Opcode *targetOpcode{opcode + opcode->immediate};

                        // Step into delay slot
                        opcode++;
                        return Step(targetOpcode);
                    }
                }
                break;
            }

            default:
                throw exception("Unknown MME opcode encountered: 0x{:X}", static_cast<u8>(opcode->operation));
        }

        if (opcode->exit && (delayedOpcode == nullptr)) {
            // Exit has a delay slot
            opcode++;
            Step(opcode);
            return false;
        }

        if (delayedOpcode != nullptr)
            opcode = delayedOpcode;
        else
            opcode++;

        return true;
    }

    __attribute__((always_inline)) u32 MacroInterpreter::HandleAlu(Opcode::AluOperation operation, u32 srcA, u32 srcB) {
        switch (operation) {
            case Opcode::AluOperation::Add: {
                u64 result{static_cast<u64>(srcA) + srcB};

                carryFlag = result >> 32;
                return static_cast<u32>(result);
            }
            case Opcode::AluOperation::AddWithCarry: {
                u64 result{static_cast<u64>(srcA) + srcB + carryFlag};

                carryFlag = result >> 32;
                return static_cast<u32>(result);
            }
            case Opcode::AluOperation::Subtract: {
                u64 result{static_cast<u64>(srcA) - srcB};

                carryFlag = result & 0xFFFFFFFF;
                return static_cast<u32>(result);
            }
            case Opcode::AluOperation::SubtractWithBorrow: {
                u64 result{static_cast<u64>(srcA) - srcB - !carryFlag};

                carryFlag = result & 0xFFFFFFFF;
                return static_cast<u32>(result);
            }
            case Opcode::AluOperation::BitwiseXor:
                return srcA ^ srcB;
            case Opcode::AluOperation::BitwiseOr:
                return srcA | srcB;
            case Opcode::AluOperation::BitwiseAnd:
                return srcA & srcB;
            case Opcode::AluOperation::BitwiseAndNot:
                return srcA & ~srcB;
            case Opcode::AluOperation::BitwiseNand:
                return ~(srcA & srcB);
        }
    }

    __attribute__((always_inline)) void MacroInterpreter::HandleAssignment(Opcode::AssignmentOperation operation, u8 reg, u32 result) {
        switch (operation) {
            case Opcode::AssignmentOperation::IgnoreAndFetch:
                WriteRegister(reg, *argument++);
                break;
            case Opcode::AssignmentOperation::Move:
                WriteRegister(reg, result);
                break;
            case Opcode::AssignmentOperation::MoveAndSetMethod:
                WriteRegister(reg, result);
                methodAddress.raw = result;
                break;
            case Opcode::AssignmentOperation::FetchAndSend:
                WriteRegister(reg, *argument++);
                Send(result);
                break;
            case Opcode::AssignmentOperation::MoveAndSend:
                WriteRegister(reg, result);
                Send(result);
                break;
            case Opcode::AssignmentOperation::FetchAndSetMethod:
                WriteRegister(reg, *argument++);
                methodAddress.raw = result;
                break;
            case Opcode::AssignmentOperation::MoveAndSetMethodThenFetchAndSend:
                WriteRegister(reg, result);
                methodAddress.raw = result;
                Send(*argument++);
                break;
            case Opcode::AssignmentOperation::MoveAndSetMethodThenSendHigh:
                WriteRegister(reg, result);
                methodAddress.raw = result;
                Send(methodAddress.increment);
                break;
        }
    }

    __attribute__((always_inline)) void MacroInterpreter::Send(u32 pArgument) {
        engine->CallMethodFromMacro(methodAddress.address, pArgument);
        methodAddress.address += methodAddress.increment;
    }

    __attribute__((always_inline)) void MacroInterpreter::WriteRegister(u8 reg, u32 value) {
        // Register 0 should always be zero so block writes to it
        if (reg == 0) [[unlikely]]
            return;

        registers[reg] = value;
    }
}
