#include "global.h"
#include "mail_message.h"
#include "message_format.h"
#include "msgdata.h"
#include "msgdata/msg.naix"
#include "string_control_code.h"
#include "constants/easy_chat.h"

struct UnkStruct_020ED556
{
    u8 unk_0;
    u8 unk_1;
    s16 unk_2;
    u16 unk_4;
    s16 unk_6;
    u16 unk_8;
};

extern u16 GetECWordIndexByPair(s16 bank, u16 num);

static const u16 sMessageBanks[] = {
    NARC_msg_narc_0397_bin,
    NARC_msg_narc_0399_bin,
    NARC_msg_narc_0395_bin,
    NARC_msg_narc_0396_bin,
    NARC_msg_narc_0398_bin
};

const struct UnkStruct_020ED556 UNK_020ED556[] = {
    { 0, 0, 0x184,  7, -1, 0 },
    { 1, 0, 0x184, 33, -1, 0 },
    { 2, 0, 0x188, 10, -1, 0 },
    { 1, 4, 0x184,  1, -1, 0 }
};

void MailMsg_Init(struct MailMessage * mailMsg)
{
    s32 i;
    mailMsg->msg_bank = MAILMSG_BANK_NONE;
    for (i = 0; i < 2; i++)
    {
        mailMsg->fields[i] = EC_WORD_NULL;
    }
}

void MailMsg_Init_WithBank(struct MailMessage * mailMsg, u16 bank)
{
    s32 i;
    mailMsg->msg_bank = bank;
    mailMsg->msg_no = 0;
    for (i = 0; i < MAILMSG_FIELDS_MAX; i++)
    {
        mailMsg->fields[i] = EC_WORD_NULL;
    }
}

void MailMsg_Init_Default(struct MailMessage * mailMsg)
{
    MailMsg_Init_WithBank(mailMsg, 4);
    mailMsg->msg_no = 5;
}

void MailMsg_Init_FromTemplate(struct MailMessage * mailMsg, u32 a1)
{
    GF_ASSERT(a1 < 4);
    if (a1 < 4)
    {
        MailMsg_Init_WithBank(mailMsg, UNK_020ED556[a1].unk_0);
        mailMsg->msg_no = UNK_020ED556[a1].unk_1;
        if (UNK_020ED556[a1].unk_2 != -1)
            mailMsg->fields[0] = GetECWordIndexByPair(UNK_020ED556[a1].unk_2, UNK_020ED556[a1].unk_4);
        if (UNK_020ED556[a1].unk_6 != -1)
            mailMsg->fields[1] = GetECWordIndexByPair(UNK_020ED556[a1].unk_6, UNK_020ED556[a1].unk_8);
    }
}

struct String * MailMsg_GetExpandedString(struct MailMessage * mailMsg, HeapID heapId)
{
    s32 i;
    MessageFormat * messageFormat = MessageFormat_New(heapId);
    struct MsgData * msgData;
    struct String * ret;
    for (i = 0; i < MAILMSG_FIELDS_MAX; i++)
    {
        if (mailMsg->fields[i] == EC_WORD_NULL)
            break;
        BufferECWord(messageFormat, (u32)i, mailMsg->fields[i]);
    }
    msgData = NewMsgDataFromNarc(MSGDATA_LOAD_LAZY, NARC_MSGDATA_MSG, sMessageBanks[mailMsg->msg_bank], heapId);
    ret = ReadMsgData_ExpandPlaceholders(messageFormat, msgData, mailMsg->msg_no, heapId);
    DestroyMsgData(msgData);
    MessageFormat_Delete(messageFormat);
    return ret;
}

struct String * MailMsg_GetRawString(struct MailMessage * mailMsg, HeapID heapId)
{
    return ReadMsgData_NewNarc_NewString(NARC_MSGDATA_MSG, sMessageBanks[mailMsg->msg_bank], mailMsg->msg_no, heapId);
}

BOOL MailMsg_IsInit(struct MailMessage * mailMsg)
{
    return mailMsg->msg_bank != MAILMSG_BANK_NONE;
}

BOOL MailMsg_AllFieldsAreInit(struct MailMessage * mailMsg)
{
    s32 i;
    u32 n = MailMsg_NumFields(mailMsg->msg_bank, mailMsg->msg_no);
    for (i = 0; i < n; i++)
    {
        if (mailMsg->fields[i] == EC_WORD_NULL)
            return FALSE;
    }
    return TRUE;
}

u32 MailMsg_NumFields(u16 bank, u16 num)
{
    struct String * str;
    const u16 * cstr;
    u32 count;
    GF_ASSERT(bank < NELEMS(sMessageBanks));
    GF_ASSERT(num < MailMsg_NumMsgsInBank(bank));
    str = ReadMsgData_NewNarc_NewString(NARC_MSGDATA_MSG, sMessageBanks[bank], num, HEAP_ID_DEFAULT);
    cstr = String_c_str(str);
    count = 0;
    while (*cstr != EOS)
    {
        if (*cstr == EXT_CTRL_CODE_BEGIN)
        {
            if (MsgArray_ControlCodeIsStrVar(cstr))
                count++;
            cstr = MsgArray_SkipControlCode(cstr);
        }
        else
            cstr++;
    }
    String_Delete(str);
    return count;
}

u16 MailMsg_GetFieldI(struct MailMessage * mailMsg, u32 a1)
{
    return mailMsg->fields[a1];
}

u16 MailMsg_GetMsgBank(struct MailMessage * mailMsg)
{
    return mailMsg->msg_bank;
}

u16 MailMsg_GetMsgNo(struct MailMessage * mailMsg)
{
    return mailMsg->msg_no;
}

BOOL MailMsg_Compare(const struct MailMessage *mailMsg, const struct MailMessage *a1)
{
    s32 i;
    if (mailMsg->msg_bank != a1->msg_bank || mailMsg->msg_no != a1->msg_no)
        return FALSE;
    for (i = 0; i < MAILMSG_FIELDS_MAX; i++)
    {
        if (mailMsg->fields[i] != a1->fields[i])
            return FALSE;
    }
    return TRUE;
}

void MailMsg_Copy(struct MailMessage * mailMsg, const struct MailMessage * a1)
{
    *mailMsg = *a1;
}

u32 MailMsg_NumMsgsInBank(u16 bank)
{
    return (u32)((bank < NELEMS(sMessageBanks)) ? 20 : 0);
}

void MailMsg_SetMsgBankAndNum(struct MailMessage * mailMsg, u16 bank, u16 num)
{
    GF_ASSERT(bank < NELEMS(sMessageBanks));
    mailMsg->msg_bank = bank;
    mailMsg->msg_no = num;
}

void MailMsg_SetFieldI(struct MailMessage * mailMsg, u32 idx, u16 word)
{
    GF_ASSERT(idx < MAILMSG_FIELDS_MAX);
    mailMsg->fields[idx] = word;
}

void MailMsg_SetTrailingFieldsEmpty(struct MailMessage * mailMsg)
{
    u32 n;
    for (n = MailMsg_NumFields(mailMsg->msg_bank, mailMsg->msg_no); n < MAILMSG_FIELDS_MAX; n++)
    {
        mailMsg->fields[n] = EC_WORD_NULL;
    }
}
