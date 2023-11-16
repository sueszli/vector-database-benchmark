// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

#include "compile_tensor_function.h"
#include "tensor_function.h"

#include <vespa/vespalib/util/classname.h>

namespace vespalib::eval {

namespace {

using vespalib::getClassName;
using State = InterpretedFunction::State;
using Instruction = InterpretedFunction::Instruction;

void op_skip(State &state, uint64_t param) {
    state.program_offset += param;
}

void op_skip_if_false(State &state, uint64_t param) {
    ++state.if_cnt;
    if (!state.peek(0).as_bool()) {
        state.program_offset += param;
    }
    state.stack.pop_back();
}

struct Frame {
    const TensorFunction &node;
    std::vector<TensorFunction::Child::CREF> children;
    size_t child_idx;
    Frame(const TensorFunction &node_in) : node(node_in), children(), child_idx(0) { node.push_children(children); }
    bool has_next_child() const { return (child_idx < children.size()); }
    const TensorFunction &next_child() { return children[child_idx++].get().get(); }
};

struct ProgramCompiler {
    const ValueBuilderFactory &factory;
    Stash &stash;
    std::vector<Frame> stack;
    std::vector<Instruction> prog;
    CTFMetaData *meta;
    ProgramCompiler(const ValueBuilderFactory &factory_in, Stash &stash_in, CTFMetaData *meta_in)
      : factory(factory_in), stash(stash_in), stack(), prog(), meta(meta_in) {}
    ~ProgramCompiler();

    void maybe_add_meta(const TensorFunction &node, const Instruction &instr) {
        if (meta != nullptr) {
            meta->steps.emplace_back(getClassName(node), instr.resolve_symbol());
        }
    }

    void append(const std::vector<Instruction> &other_prog) {
        prog.insert(prog.end(), other_prog.begin(), other_prog.end());
    }

    void open(const TensorFunction &node) {
        if (auto if_node = as<tensor_function::If>(node)) {
            append(compile_tensor_function(factory, if_node->cond(), stash, meta));
            maybe_add_meta(node, Instruction(op_skip_if_false));
            auto true_prog = compile_tensor_function(factory, if_node->true_child(), stash, meta);
            maybe_add_meta(node, Instruction(op_skip));
            auto false_prog = compile_tensor_function(factory, if_node->false_child(), stash, meta);
            true_prog.emplace_back(op_skip, false_prog.size());
            prog.emplace_back(op_skip_if_false, true_prog.size());
            append(true_prog);
            append(false_prog);
        } else {
            stack.emplace_back(node);
        }
    }

    void close(const TensorFunction &node) {
        prog.push_back(node.compile_self(factory, stash));
        maybe_add_meta(node, prog.back());
    }

    std::vector<Instruction> compile(const TensorFunction &function) {
        open(function);
        while (!stack.empty()) {
            if (stack.back().has_next_child()) {
                open(stack.back().next_child());
            } else {
                close(stack.back().node);
                stack.pop_back();
            }
        }
        return std::move(prog);
    }
};
ProgramCompiler::~ProgramCompiler() = default;

} // namespace vespalib::eval::<unnamed>

CTFMetaData::~CTFMetaData() = default;

std::vector<Instruction> compile_tensor_function(const ValueBuilderFactory &factory, const TensorFunction &function, Stash &stash, CTFMetaData *meta) {
    ProgramCompiler compiler(factory, stash, meta);
    return compiler.compile(function);
}

} // namespace vespalib::eval
