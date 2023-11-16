#include "global.h"
#include "scrcmd.h"
#include "message_format.h"
#include "math_util.h"
#include "unk_020286F8.h"
#include "unk_020377F0.h"

extern void* FieldSysGetAttrAddr(struct FieldSystem*, u8 idx);

extern BOOL sub_020612EC(struct FieldSystem*);
extern BOOL sub_020612F8(struct FieldSystem*);

BOOL ScrCmd_UnionGroup(struct ScriptContext* ctx) //021D
{
    MessageFormat **messageFormat = FieldSysGetAttrAddr(ctx->fieldSystem, SCRIPTENV_MESSAGE_FORMAT);
    struct UnkSaveStruct_020286F8* unk_sav_ptr = sub_0202881C(ctx->fieldSystem->saveData);
    struct SaveData* save = ctx->fieldSystem->saveData;

    u16 option = ScriptReadHalfword(ctx);
    switch (option)
    {
    case 0: { //check if group ID exists
        u16 unk_var = ScriptGetVar(ctx);
        u16* ret_ptr = ScriptGetVarPointer(ctx);

        *ret_ptr = (u16)sub_02028828(unk_sav_ptr, unk_var);
        return FALSE;
    }
    case 1: { //check if group ID is accessible
        u16 unk_var = ScriptGetVar(ctx);
        u16* ret_ptr = ScriptGetVarPointer(ctx);

        *ret_ptr = (u16)sub_02028840(unk_sav_ptr, unk_var);
        return FALSE;
    }
    case 2: { //writes group ID to string buffer
        u16 unk_var = ScriptGetVar(ctx);
        u16 idx = ScriptGetVar(ctx);

        BufferGroupName(*messageFormat, save, unk_var, idx, 0);
        break;
    }
    case 3: { //writes group leader name to string buffer
        u16 unk_var = ScriptGetVar(ctx);
        u16 idx = ScriptGetVar(ctx);

        BufferGroupName(*messageFormat, save, unk_var, idx, 1);
        break;
    }
    case 4: { //opens keyboard, 2 if group id exists, 1 if cancel, 0 otherwise
        u16* unk_str_ptr = sub_020287A8(unk_sav_ptr, 0, 0);
        u16* ret_ptr = ScriptGetVarPointer(ctx);

        CreateNamingScreen(ctx->taskManager, NAMINGSCREEN_GROUP, 0, PLAYER_NAME_LENGTH, 0, unk_str_ptr, ret_ptr); //should be GROUP_NAME_LENGTH?
        return TRUE;
    }
    case 5: { //enter in group id (whatever this means, needs more investigation)
        u16 src_idx = ScriptGetVar(ctx);
        BOOL unk_bool = sub_02028828(unk_sav_ptr, 1);

        sub_02028700(unk_sav_ptr, src_idx, 1);
        if (unk_bool != FALSE)
        {
            sub_020612F8(ctx->fieldSystem);
        }

        return FALSE;
    }
    case 6: { //create a group
        struct String* player_name = String_New(64, HEAP_ID_32);
        PlayerProfile* player = Save_PlayerData_GetProfileAddr(ctx->fieldSystem->saveData);

        PlayerName_FlatToString(player, player_name);
        sub_020287C0(unk_sav_ptr, 0, 1, player_name);
        sub_020287EC(unk_sav_ptr, 0, PlayerProfile_GetTrainerGender(player));
        sub_02028810(unk_sav_ptr, 0, 2);
        sub_02028788(unk_sav_ptr, 0, MTRandom());

        String_Delete(player_name);

        sub_02028700(unk_sav_ptr, 0, 1);
        sub_020612EC(ctx->fieldSystem);

        break;
    }
    }

    return FALSE;
}
