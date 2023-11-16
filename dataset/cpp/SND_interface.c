#include "SND_interface.h"
#include "SND_command.h"
#include "nitro/types.h"
#include "code32.h"

static void PushCommand_impl(s32 id, u32 par1, u32 par2, u32 par3, u32 par4);
#define PushCmd1(id, a)          PushCommand_impl(id, (u32)(a), 0, 0, 0)
#define PushCmd2(id, a, b)       PushCommand_impl(id, (u32)(a), (u32)(b), 0, 0)
#define PushCmd3(id, a, b, c)    PushCommand_impl(id, (u32)(a), (u32)(b), (u32)(c), 0)
#define PushCmd4(id, a, b, c, d) PushCommand_impl(id, (u32)(a), (u32)(b), (u32)(c), (u32)(d))

// TODO fill in "random" constants with macros

// void SND_StartSeq(s32 player, const void *seqBasePtr, u32 seqOffset, struct SNDBankData *bankData) { }

void SND_StopSeq(s32 player) {
    PushCmd1(SND_CMD_STOP_SEQ, player);
}

void SND_PrepareSeq(s32 player, const void *seqBasePtr, u32 seqOffset, struct SNDBankData *bankData) {
    PushCmd4(SND_CMD_PREPARE_SEQ, player, seqBasePtr, seqOffset, bankData);
}

void SND_StartPreparedSeq(s32 player) {
    PushCmd1(SND_CMD_START_PREPARED_SEQ, player);
}

void SND_PauseSeq(s32 player, BOOL flag) {
    PushCmd2(SND_CMD_PAUSE_SEQ, player, flag);
}

// void SND_SetPlayerTempoRatio(s32 player, s32 ratio) { }

void SND_SetPlayerVolume(s32 player, s32 volume) {
    SNDi_SetPlayerParam(player, 6, (u32)volume, 2);
}

void SND_SetPlayerChannelPriority(s32 player, s32 prio) {
    SNDi_SetPlayerParam(player, 4, (u32)prio, 1);
}

// void SND_SetPlayerLocalVariable(s32 player, s32 varNo, s16 var) { }

// void SND_SetPlayerGlobalVariable(s32 varNo, s16 var) { }

// void SND_SetTrackVolume(s32 player, u32 trackBitMask, s32 volume) { }

void SND_SetTrackPitch(s32 player, u32 trackBitMask, s32 pitch) {
    SNDi_SetTrackParam(player, trackBitMask, 12, (u32)pitch, 2);
}

void SND_SetTrackPan(s32 player, u32 trackBitMask, s32 pan) {
    SNDi_SetTrackParam(player, trackBitMask, 9, (u32)pan, 1);
}

void SND_SetTrackAllocatableChannel(s32 player, u32 trackBitMask, u32 chnBitMask) {
    PushCmd3(SND_CMD_ALLOCATABLE_CHANNEL, player, trackBitMask, chnBitMask);
}

void SND_StartTimer(u32 chnBitMask, u32 capBitMask, u32 alarmBitMask, u32 flags) {
    PushCmd4(SND_CMD_START_TIMER, chnBitMask, capBitMask, alarmBitMask, flags);
}

void SND_StopTimer(u32 chnBitMask, u32 capBitMask, u32 alarmBitMask, u32 flags) {
    s32 i = 0;
    u32 tmpMask = alarmBitMask;

    while (i < SND_ALARM_COUNT && tmpMask != 0) {
        if (tmpMask & 1)
            SNDi_IncAlarmId((u32)i);
        i++;
        tmpMask >>= 1;
    }

    PushCmd4(SND_CMD_STOP_TIMER, chnBitMask, capBitMask, alarmBitMask, flags);
}

void SND_SetupCapture(s32 capture, s32 format, void *bufferPtr, u32 length, BOOL loopFlag, s32 in, s32 out) {
    PushCmd3(SND_CMD_SETUP_CAPTURE, bufferPtr, length,
            (capture << 31) | (format << 30) | (loopFlag << 29) | (in << 28) | (out << 27));
}

void SND_SetupAlarm(s32 alarm, u32 tick, u32 period, SNDAlarmCallback cb, void *userData) {
    PushCmd4(SND_CMD_SETUP_ALARM, alarm, tick, period, SNDi_SetAlarmHandler((u32)alarm, cb, userData));
}

// void SND_SetTrackMute(s32 player, u32 trackBitMask, BOOL flag) { }

// void SND_StopUnlockedChannel(u32 chnBitMask, u32 flags) { }

void SND_LockChannel(u32 chnBitMask, u32 flags) {
    PushCmd2(SND_CMD_LOCK_CHANNEL, chnBitMask, flags);
}

void SND_UnlockChannel(u32 chnBitMask, u32 flags) {
    PushCmd2(SND_CMD_UNLOCK_CHANNEL, chnBitMask, flags);
}

void SND_SetChannelTimer(u32 chnBitMask, s32 timer) {
    PushCmd2(SND_CMD_CHANNEL_TIMER, chnBitMask, timer);
}

void SND_SetChannelVolume(u32 chnBitMask, s32 volume, s32 chnDataShift) {
    PushCmd3(SND_CMD_CHANNEL_VOLUME, chnBitMask, volume, chnDataShift);
}

void SND_SetChannelPan(u32 chnBitMask, s32 pan) {
    PushCmd2(SND_CMD_CHANNEL_PAN, chnBitMask, pan);
}

void SND_SetupChannelPcm(s32 chn, s32 waveFormat, const void *dataAddr, s32 loopMode, s32 loopStart, s32 dataLen, s32 volume, s32 chnDataShift, s32 timer, s32 pan) {
    PushCmd4(SND_CMD_SETUP_CHANNEL_PCM,
            chn | (timer << 16),
            dataAddr,
            (volume << 24) | (chnDataShift << 22) | dataLen,
            (loopMode << 26) | (waveFormat << 24) | (pan << 16) | loopStart);
}

// void SND_SetupChannelPsg(s32 chn, s32 sndDuty, s32 volume, s32 chnDataShift, s32 timer, s32 pan) { }

// void SND_SetupChannelNoise(s32 chn, s32 volume, s32 chnDataShift, s32 timer, s32 pan) { }

void SND_InvalidateSeqData(const void *start, const void *end) {
    PushCmd2(SND_CMD_INVALIDATE_SEQ, start, end);
}

void SND_InvalidateBankData(const void *start, const void *end) {
    PushCmd2(SND_CMD_INVALIDATE_BANK, start, end);
}

void SND_InvalidateWaveData(const void *start, const void *end) {
    PushCmd2(SND_CMD_INVALIDATE_WAVE, start, end);
}

// void SND_SetMasterVolume(s32 volume) { }

void SND_SetOutputSelector(s32 left, s32 right, s32 channel1, s32 channel3) {
    PushCmd4(SND_CMD_OUTPUT_SELECTOR, left, right, channel1, channel3);
}

void SND_SetMasterPan(s32 pan) {
    PushCmd1(SND_CMD_MASTER_PAN, pan);
}

void SND_ResetMasterPan(void) {
    PushCmd1(SND_CMD_MASTER_PAN, -1);
}

// void SND_ReadDriverInfo(struct SNDDriverInfo *info) { }

void SNDi_SetPlayerParam(s32 player, u32 offset, u32 data, s32 size) {
    PushCmd4(SND_CMD_PLAYER_PARAM, player, offset, data, size);
}

void SNDi_SetTrackParam(s32 player, u32 trackBitMask, u32 offset, u32 data, s32 size) {
    PushCmd4(SND_CMD_TRACK_PARAM, player | (size << 24), trackBitMask, offset, data);
}

// void SNDi_SetSurroundDecay(s32 decay) { }

// void SNDi_SkipSeq(s32 player, u32 tick) { }

static void PushCommand_impl(s32 id, u32 par1, u32 par2, u32 par3, u32 par4) {
    struct SNDCommand *cmd = SND_AllocCommand(SND_CMD_FLAG_BLOCK);
    if (cmd == NULL)
        return;

    cmd->id = id;
    cmd->arg[0] = par1;
    cmd->arg[1] = par2;
    cmd->arg[2] = par3;
    cmd->arg[3] = par4;
    SND_PushCommand(cmd);
}
