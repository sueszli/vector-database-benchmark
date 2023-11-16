/* Loads and run ARM code snippets on the target */

#include "emu.h"
#include "cpu.h"
#include "armsnippets.h"
#include "mem.h"
#include "translate.h"
#include "armsnippets.h"
#include "debug.h"

#include <string.h>
#include <stdio.h>
#include <stdint.h>

#include "armcode_bin.h"

struct arm_state armloader_orig_arm_state;

static void armloader_restore_state(void) {
    memcpy(&arm, &armloader_orig_arm_state, sizeof(arm));
}

static void (*armloader_cb_ptr)(struct arm_state*);

void armloader_cb(void) {
    struct arm_state after_snippet_exec_arm_state;
    memcpy(&after_snippet_exec_arm_state, &arm, sizeof(arm));
    armloader_restore_state();
    if(armloader_cb_ptr)
        armloader_cb_ptr(&after_snippet_exec_arm_state);
}

/* Load the snippet and jump to it.
 * snippets are defined in armsnippets.S.
 * params can be null if not required.
 * params may contain pointers to data which should be copied to device space.
 * Each param will be copied to the ARM stack, and its address written in rX, starting from r0.
 * params_num must be less or equal than 12.
 * callback() will be called once the snippet has finished its execution. Can be NULL.
 * returns 0 if success.
 */
bool armloader_load_snippet(enum SNIPPETS snippet, struct armloader_load_params params[],  uint32_t params_num, void (*callback)(struct arm_state*)) {
    uint32_t i;
    uint32_t code_size = snippets_bin_len;
    void *code_ptr;
    uint32_t orig_pc;

    if(code_size % 4)
        code_size += 4 - (code_size % 4); // word-aligned

    memcpy(&armloader_orig_arm_state, &arm, sizeof(arm));

    if(!virt_mem_ptr(arm.reg[13] /* sp */, 4)) {
        emuprintf("sp points to an invalid address\n");
        armloader_restore_state();
        return false;
    }
    arm.reg[13] -= code_size;
    code_ptr = virt_mem_ptr(arm.reg[13], code_size);
    if(!code_ptr) {
        emuprintf("not enough stack space to run snippet\n");
        armloader_restore_state();
        return false;
    }
    memcpy(code_ptr, snippets_bin, code_size);

    orig_pc = arm.reg[15];
    arm.reg[14] = arm.reg[15]; // return address
    arm.reg[15] = arm.reg[13] + *(uint32_t *)code_ptr; // load_snippet
    arm.reg[12] = snippet;

    for(i = 0; i < params_num; i++) {
        if(params[i].t == ARMLOADER_PARAM_VAL)
            arm.reg[i] = params[i].v;
        else {
            void *param_ptr;
            uint32_t size = params[i].p.size;
            if (size % 4)
                size += 4 - size % 4; // word-aligned
            arm.reg[13] -= size;
            arm.reg[i] = arm.reg[13];
            param_ptr = virt_mem_ptr(arm.reg[13], 4);
            if (!param_ptr) {
                emuprintf("not enough stack space for snippet parameters\n");
                armloader_restore_state();
                return false;
            }
            memcpy(param_ptr, params[i].p.ptr, params[i].p.size);
        }
    }
    armloader_cb_ptr = callback;
    uint32_t *flags = &RAM_FLAGS(virt_mem_ptr(orig_pc, 4));
    if (*flags & RF_CODE_TRANSLATED) flush_translations();
    *flags |= RF_ARMLOADER_CB;

    // for debugging
    /*flags = &RAM_FLAGS(virt_mem_ptr(arm.reg[15], 4));
    if (*flags & RF_CODE_TRANSLATED) flush_translations();
    *flags |= RF_EXEC_BREAKPOINT;*/

    return true;
}
