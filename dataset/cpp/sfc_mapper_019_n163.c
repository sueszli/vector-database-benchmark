﻿#include "sfc_cpu.h"
#include "sfc_famicom.h"
#include "sfc_mapper_helper.h"
#include <assert.h>
#include <string.h>

#ifndef NDEBUG
#include <stdio.h>
#endif

// ------------------------------- MAPPER 019 - N129/N163

#define SFC_N163_IT1
#define SFC_N163_IT2
//#define SFC_N163_IT1 * 2
//#define SFC_N163_IT2 / 2

typedef struct {
    // IRQ 计数器
    uint16_t    irq_counter;
    // IRQ 使能
    uint8_t     irq_enable;
    // $E800 相关标志位(反转)
    uint8_t     flags_e800_rev;
} sfc_mapper13_t;


/// <summary>
/// SFCs the N163 internal chip.
/// </summary>
/// <param name="famicom">The famicom.</param>
/// <returns></returns>
static inline uint8_t* sfc_n163_internal_chip(sfc_famicom_t* famicom) {
    return famicom->expansion_ram32 + 8 * 1024;
}


/// <summary>
/// SFCs the mapper.
/// </summary>
/// <param name="famicom">The famicom.</param>
/// <returns></returns>
static inline sfc_mapper13_t* sfc_mapper(sfc_famicom_t* famicom) {
    return (sfc_mapper13_t*)famicom->mapper_buffer.mapper13;
}

#define MAPPER sfc_mapper13_t* const mapper = sfc_mapper(famicom);



/// <summary>
/// SFCs the N163 disable sound.
/// </summary>
/// <param name="famicom">The famicom.</param>
/// <param name="v">The v.</param>
static void sfc_n163_disable_sound(sfc_famicom_t* famicom, uint8_t v) {
    if (v)
        famicom->rom_info.extra_sound &= ~SFC_NSF_EX_VCR7;
    else
        famicom->rom_info.extra_sound |= SFC_NSF_EX_VCR7;
}


/// <summary>
/// StepFC: N163 采样
/// </summary>
/// <param name="famicom">The famicom.</param>
/// <param name="addr">The addr.</param>
/// <returns></returns>
static inline int8_t sfc_n163_sample(sfc_famicom_t* famicom, uint8_t addr) {
    const uint8_t data = sfc_n163_internal_chip(famicom)[addr >> 1];
    return (data >> ((addr & 1) << 2)) & 0xf;
}


/// <summary>
/// SFCs the N163 load prgrom.
/// </summary>
/// <param name="famicom">The famicom.</param>
/// <param name="id">The identifier.</param>
/// <param name="value">The value.</param>
static void sfc_n163_load_prgrom(sfc_famicom_t* famicom, int id, uint8_t value) {
    const uint32_t count_prgrom8kb = famicom->rom_info.size_prgrom >> 13;
    sfc_load_prgrom_8k(famicom, id, (value & 0x3f) % count_prgrom8kb);
}


/// <summary>
/// SFCs the N163 load chrbank.
/// </summary>
/// <param name="famicom">The famicom.</param>
/// <param name="id">The identifier.</param>
/// <param name="value">The value.</param>
/// <param name="flag">The flag.</param>
static void sfc_n163_load_chrbank(sfc_famicom_t* famicom, uint16_t id, uint8_t value, uint8_t flag) {
    // 支持切换CIRAM
    if (flag && (value & 0xe0) == 0xe0)
        famicom->ppu.banks[id] = famicom->video_memory + (((uint16_t)value & 1) << 10);
    // CHR-ROM
    else
        sfc_load_chrrom_1k(famicom, id, value);
}



// IRQ - 中断请求 - 确认
extern inline void sfc_operation_IRQ_acknowledge(sfc_famicom_t* famicom);
// 尝试触发
extern inline void sfc_operation_IRQ_try(sfc_famicom_t* famicom);


/// <summary>
/// Calculates the index.
/// </summary>
/// <param name="addr">The addr.</param>
/// <returns></returns>
static inline enum sfc_channel_index calc_index(uint8_t addr) {
    if (addr < 0x40) return SFC_N163_N163;
    return SFC_N163_Wavefrom0 + (addr >> 3) & 7;
}

/// <summary>
/// SFCs the mapper 15 write low.
/// </summary>
/// <param name="famicom">The famicom.</param>
/// <param name="address">The address.</param>
/// <param name="value">The value.</param>
void sfc_mapper_13_write_low(sfc_famicom_t* famicom, uint16_t address, uint8_t value) {
    MAPPER;
    // $4020-$5FFF
    switch ((address >> 11) & 3)
    {
        uint8_t tmp;
    case 1:
        famicom->interfaces.audio_change(
            famicom->argument, 
            famicom->cpu_cycle_count, 
            calc_index(famicom->apu.n163.n163_addr)
        );
        // Data Port ($4800-$4FFF)
        sfc_n163_internal_chip(famicom)[famicom->apu.n163.n163_addr] = value;
        famicom->apu.n163.n163_addr += famicom->apu.n163.n163_inc;
        famicom->apu.n163.n163_addr &= 0x7f;
        // 计算数量
        tmp = sfc_n163_internal_chip(famicom)[0x7f];
        famicom->apu.n163.n163_count = ((tmp >> 4) & 7) + 1;
        famicom->apu.n163.n163_lowest_id = 8 - famicom->apu.n163.n163_count;
        break;
    case 2:
        // IRQ Counter (low) ($5000-$57FF)
        mapper->irq_counter
            = (mapper->irq_counter & (uint16_t)0xff00)
            | (uint16_t)value
            ;
        sfc_operation_IRQ_acknowledge(famicom);
        break;
    case 3:
        // IRQ Counter (high) / IRQ Enable ($5800-$5FFF)
        mapper->irq_counter
            = (mapper->irq_counter & (uint16_t)0x00ff)
            | ((uint16_t)(value & 0x7f) << 8)
            ;
        mapper->irq_enable = value & 0x80;
        sfc_operation_IRQ_acknowledge(famicom);
        break;
    }
}


/// <summary>
/// SFCs the mapper 15 write high.
/// </summary>
/// <param name="famicom">The famicom.</param>
/// <param name="address">The address.</param>
/// <param name="value">The value.</param>
void sfc_mapper_13_write_high(sfc_famicom_t* famicom, uint16_t address, uint8_t value) {
    MAPPER;
    uint16_t addr0;
    // 取D11-D14
    switch (addr0 = (address >> 11) & 0xf)
    {
    case 0x0:
    case 0x1:
    case 0x2:
    case 0x3:
        // CHR and NT Select ($8000-$DFFF)
        sfc_n163_load_chrbank(famicom, addr0, value, mapper->flags_e800_rev & 0x40);
        break;
    case 0x4:
    case 0x5:
    case 0x6:
    case 0x7:
        // CHR and NT Select ($8000-$DFFF)
        sfc_n163_load_chrbank(famicom, addr0, value, mapper->flags_e800_rev & 0x80);
        break;
    case 0x8:
    case 0x9:
    case 0xa:
    case 0xb:
        // CHR and NT Select ($8000-$DFFF)
        sfc_n163_load_chrbank(famicom, addr0, value, 1);
        // 镜像
        famicom->ppu.banks[addr0 + 4] = famicom->ppu.banks[addr0];
        break;
    case 0xc:
        // PRG Select 1 ($E000-$E7FF)
        sfc_n163_disable_sound(famicom, value & 0x40);
        sfc_n163_load_prgrom(famicom, 0, value);
        break;
    case 0xd:
        // PRG Select 2 / CHR-RAM Enable ($E800-$EFFF)
        mapper->flags_e800_rev = ~value & 0xc0;
        sfc_n163_load_prgrom(famicom, 1, value);
        break;
    case 0xe:
        // PRG Select 3 ($F000-$F7FF)
        sfc_n163_load_prgrom(famicom, 2, value);
        break;
    case 0xf:
        // Write Protect for External RAM AND Chip RAM Address Port ($F800-$FFFF)

        // Address Port ($F800-$FFFF)
        famicom->apu.n163.n163_addr = value & 0x7f;
        famicom->apu.n163.n163_inc = (value >> 7) & 1;
        break;
    }
}

/// <summary>
/// SFCs the mapper 05 hsync.
/// </summary>
/// <param name="famicom">The famicom.</param>
/// <param name="value">The value.</param>
static void sfc_mapper_13_hsync(sfc_famicom_t* famicom, uint16_t value) {
    MAPPER;
    if (mapper->irq_enable && !(mapper->irq_counter & 0x8000)) {
        // 每根扫描线可以视为113.66周期
        const uint16_t count = (value % 3) ? 114 : 113;
        mapper->irq_counter += count;
        if (mapper->irq_counter & 0x8000) {
            sfc_operation_IRQ_try(famicom);
            //printf("N163: IRQ triggered@scanline: %d\n", value);
        }
    }
}


/// <summary>
/// SFCs the mapper 15 read low.
/// </summary>
/// <param name="famicom">The famicom.</param>
/// <param name="address">The address.</param>
/// <returns></returns>
static uint8_t sfc_mapper_13_read_low(sfc_famicom_t* famicom, uint16_t address) {
    MAPPER;
    // $4020-$5FFF
    switch ((address >> 11) & 3)
    {
        uint8_t value;
    case 1:
        // Data Port ($4800-$4FFF)
        value = sfc_n163_internal_chip(famicom)[famicom->apu.n163.n163_addr];
        famicom->apu.n163.n163_addr += famicom->apu.n163.n163_inc;
        famicom->apu.n163.n163_addr &= 0x7f;
        return value;
    case 2:
        // IRQ Counter (low) ($5000-$57FF)
        return mapper->irq_counter & 0xff;
    case 3:
        // IRQ Counter (high) / IRQ Enable ($5800-$5FFF)
        return mapper->irq_enable | (mapper->irq_counter >> 8);
    }
    return 0;
}


/// <summary>
/// SFCs the mapper 15 reset.
/// </summary>
/// <param name="famicom">The famicom.</param>
/// <returns></returns>
static sfc_ecode sfc_mapper_13_reset(sfc_famicom_t* famicom) {
    MAPPER;
    // 支持N163
    famicom->rom_info.extra_sound = SFC_NSF_EX_N163;
    // 载入最后的BANK
    const uint32_t count_prgrom8kb = famicom->rom_info.size_prgrom >> 13;
    //sfc_load_prgrom_8k(famicom, 0, count_prgrom8kb - 4);
    //sfc_load_prgrom_8k(famicom, 1, count_prgrom8kb - 3);
    //sfc_load_prgrom_8k(famicom, 2, count_prgrom8kb - 2);
    sfc_load_prgrom_8k(famicom, 3, count_prgrom8kb - 1);
    famicom->apu.n163.n163_current = 0;
    famicom->apu.n163.n163_count = 1;
    famicom->apu.n163.n163_lowest_id = 7;
    // TODO: Submapper处理副权重
    famicom->apu.n163.subweight_div16 = 6 * 16;

    for (int i = 0; i != 8; ++i)
        sfc_load_chrrom_1k(famicom, i, i);

    return SFC_ERROR_OK;
}


// 默认写入
extern void sfc_mapper_wrts_defualt(const sfc_famicom_t* famicom);
// 默认读取
extern void sfc_mapper_rrfs_defualt(sfc_famicom_t* famicom);


/// <summary>
/// SFCs the mapper 15 write ram.
/// </summary>
/// <param name="famicom">The famicom.</param>
static void sfc_mapper_13_write_ram(const sfc_famicom_t* famicom) {
    // 128字节N163内置RAM
    famicom->interfaces.sl_write_stream(
        famicom->argument,
        sfc_n163_internal_chip((sfc_famicom_t*)famicom),
        128
    );
    sfc_mapper_wrts_defualt(famicom);
}


/// <summary>
/// SFCs the mapper 15 read ram.
/// </summary>
/// <param name="famicom">The famicom.</param>
static void sfc_mapper_13_read_ram(sfc_famicom_t* famicom) {
    // 128字节N163内置RAM
    famicom->interfaces.sl_read_stream(
        famicom->argument,
        sfc_n163_internal_chip(famicom),
        128
    );
    sfc_mapper_rrfs_defualt(famicom);
}

/// <summary>
/// SFCs the load mapper 13.
/// </summary>
/// <param name="famicom">The famicom.</param>
/// <returns></returns>
extern inline sfc_ecode sfc_load_mapper_13(sfc_famicom_t* famicom) {
    enum { MAPPER_13_SIZE_IMPL = sizeof(sfc_mapper13_t) };
    static_assert(SFC_MAPPER_13_SIZE == MAPPER_13_SIZE_IMPL, "SAME");
    // 初始化回调接口
    famicom->mapper.reset = sfc_mapper_13_reset;
    famicom->mapper.hsync = sfc_mapper_13_hsync;
    famicom->mapper.read_low = sfc_mapper_13_read_low;
    famicom->mapper.write_low = sfc_mapper_13_write_low;
    famicom->mapper.write_high = sfc_mapper_13_write_high;
    famicom->mapper.write_ram_to_stream = sfc_mapper_13_write_ram;
    famicom->mapper.read_ram_from_stream = sfc_mapper_13_read_ram;
    // 初始化数据
    MAPPER;
    memset(mapper, 0, MAPPER_13_SIZE_IMPL);
    // 可能超过8KiB的SRAM
    if (famicom->rom_info.save_ram_flags)
        famicom->rom_info.save_ram_flags |= SFC_ROMINFO_SRAM_M128_Of8;
    return SFC_ERROR_OK;
}





// ----------------------------------------------------------------------------
//                                N163
// ----------------------------------------------------------------------------

#include "sfc_play.h"

enum {
    //SFC_N163_LowFrequency = 0,
    //SFC_N163_LowPhase,
    //SFC_N163_MidFrequency,
    //SFC_N163_MidPhase,
    //SFC_N163_HighFrequencyWaveLength,
    //SFC_N163_HighPhase,
    //SFC_N163_WaveAddress,
    //SFC_N163_Volume,

    SFC_N163_78 = 0,
    SFC_N163_79,
    SFC_N163_7A,
    SFC_N163_7B,
    SFC_N163_7C,
    SFC_N163_7D,
    SFC_N163_7E,
    SFC_N163_7F,
};

typedef struct { uint32_t freq; uint16_t len; uint8_t vol; uint8_t addr; } sfc_n163info_t;

/// <summary>
/// SFCs the N163 get information.
/// </summary>
/// <param name="famicom">The famicom.</param>
/// <param name="ch">The ch.</param>
extern sfc_n163info_t sfc_n163_get_info(sfc_famicom_t* famicom, uint8_t ch) {
    // 基础地址
    uint8_t* const ram = sfc_n163_internal_chip(famicom);
    // 把声道映射到地址
    uint8_t* const chn = ram + ((ch << 3) | 0x40);
    const uint32_t freq
        = ((uint32_t)(chn[SFC_N163_7C] & 0x03) << 16)
        | (uint32_t)(chn[SFC_N163_7A] << 8)
        | (uint32_t)(chn[SFC_N163_78])
        ;
    const uint16_t length = 256 - (chn[SFC_N163_7C] & 0xFC);
    const uint8_t volume = chn[SFC_N163_7F] & 0x0F;
    const uint8_t offset = chn[SFC_N163_7E];

    const sfc_n163info_t rv = { freq, length, volume, offset };
    return rv;
}


/// <summary>
/// SFCs the N163 update wavetable.
/// </summary>
/// <param name="famicom">The famicom.</param>
/// <param name="out">The out.</param>
/// <param name="ch">The ch.</param>
extern void sfc_n163_update_wavetable(sfc_famicom_t* famicom, float* out, uint16_t len) {
    const uint8_t * const ram = sfc_n163_internal_chip(famicom);
    const uint16_t count = len >> 1;
    for (uint16_t i = 0; i != count; ++i) {
        const uint8_t data = ram[i];
        out[0] = (float)(data & 0xf);
        out[1] = (float)((data >> 4) & 0xf);
        out += 2;
    }
}

/// <summary>
/// SFCs the N163 update.
/// </summary>
/// <param name="famicom">The famicom.</param>
/// <param name="ch">The ch.</param>
void sfc_n163_update_ch(sfc_famicom_t* famicom, uint8_t ch) {
    assert(ch < 8 && "out of range");
    /*
     https://wiki.nesdev.com/w/index.php/Namco_163_audio
     * w[$80] = the 163's internal memory
     * sample(x) = (w[x/2] >> ((x&1)*4)) & $0F
     * phase = (w[$7D] << 16) + (w[$7B] << 8) + w[$79]
     * freq = ((w[$7C] & $03) << 16) + (w[$7A] << 8) + w[$78]
     * length = 256 - (w[$7C] & $FC)
     * offset = w[$7E]
     * volume = w[$7F] & $0F

     phase = (phase + freq) % (length << 16)
     output = (sample(((phase >> 16) + offset) & $FF) - 8) * volume
    */

    // 基础地址
    uint8_t* const ram = sfc_n163_internal_chip(famicom);
    // 把声道映射到地址
    uint8_t* const chn = ram + ((ch << 3) | 0x40);
    // 直接复制
    uint32_t phase
        = ((uint32_t)chn[SFC_N163_7D] << 16)
        | ((uint32_t)chn[SFC_N163_7B] << 8)
        | ((uint32_t)chn[SFC_N163_79])
        ;
    const uint32_t freq
        = ((uint32_t)(chn[SFC_N163_7C] & 0x03) << 16)
        | (uint32_t)(chn[SFC_N163_7A] << 8)
        | (uint32_t)(chn[SFC_N163_78])
        ;
    const uint32_t length = 256 - (chn[SFC_N163_7C] & 0xFC);
    const uint8_t offset = chn[SFC_N163_7E];
    const int8_t volume = chn[SFC_N163_7F] & 0x0F;

    phase = (phase + freq SFC_N163_IT2) % (length << 16);

    const int8_t sample = sfc_n163_sample(famicom, ((phase >> 16) + offset) & 0xFF);
    const int8_t output = (sample - 8) * volume;
    famicom->apu.n163.ch_output[ch] = output;
    // 载入历史
    famicom->apu.n163.history[famicom->apu.n163.history_index].output = output;
    famicom->apu.n163.history[famicom->apu.n163.history_index].index = ch;
    famicom->apu.n163.history_index++;
    famicom->apu.n163.history_index &= 7;
    // 写回相位
    chn[SFC_N163_7D] = (phase >> 16) & 0xff;
    chn[SFC_N163_7B] = (phase >> 8) & 0xff;
    chn[SFC_N163_79] = (phase >> 0) & 0xff;
}

//void debug_putaudio(float x);
//static int16_t last = 0;
//static uint8_t counter = 0;

/// <summary>
/// SFCs the N163 update.
/// </summary>
/// <param name="famicom">The famicom.</param>
extern void sfc_n163_update(sfc_famicom_t* famicom) {
    uint8_t* const index = &famicom->apu.n163.n163_current;
    --*index;
    *index &= 7;
    if (*index < famicom->apu.n163.n163_lowest_id) *index = 7;
    sfc_n163_update_ch(famicom, *index);
    //const int8_t data = famicom->apu.n163.ch_output[*index];
    //++counter;
    //if (counter & 1) last = data;
    //else {
    //    last += data;
    //    debug_putaudio((float)last / 256.f);
    //}
}

#ifdef SFC_SMF_DEFINED

/// <summary>
/// SFCs the N163 per sample.
/// </summary>
/// <param name="famicom">The famicom.</param>
/// <param name="ctx">The CTX.</param>
/// <param name="cps">The CPS.</param>
/// <param name="mode">The mode.</param>
/// <returns></returns>
float sfc_n163_per_sample(sfc_famicom_t * famicom, sfc_n163_ctx_t * ctx, float cps, uint8_t mode) {
    ctx->clock += cps SFC_N163_IT1;
    while (ctx->clock >= 15.f) {
        ctx->clock -= 15.f;
        sfc_n163_update(famicom);
    }

    int16_t output = 0;
    const uint8_t start_index = famicom->apu.n163.history_index - mode;
    for (uint8_t i = 0; i != mode; ++i) {
        const uint8_t now_index = (start_index + i) & 7;
        output += famicom->apu.n163.history[now_index].output;
    }
    return (float)output / (float)mode;
}


/// <summary>
/// SFCs the MMC5 samplemode begin.
/// </summary>
/// <param name="famicom">The famicom.</param>
/// <param name="ctx">The CTX.</param>
void sfc_n163_samplemode_begin(sfc_famicom_t* famicom, sfc_n163_ctx_t* ctx) {
    ctx->clock = (float)famicom->apu.n163.n163_clock;
    ctx->subweight = (float)famicom->apu.n163.subweight_div16 / 16.f;
}

/// <summary>
/// SFCs the N163 samplemode end.
/// </summary>
/// <param name="famicom">The famicom.</param>
/// <param name="ctx">The CTX.</param>
void sfc_n163_samplemode_end(sfc_famicom_t* famicom, sfc_n163_ctx_t* ctx) {
    famicom->apu.n163.n163_clock = (uint8_t)ctx->clock;
}

#endif


/// <summary>
/// StepFC: N163 整型采样模式 - 采样
/// </summary>
/// <param name="famicom">The famicom.</param>
/// <param name="ctx">The CTX.</param>
/// <param name="chw">The CHW.</param>
/// <param name="cps">The CPS.</param>
void sfc_n163_smi_sample(sfc_famicom_t* famicom, sfc_n163_smi_ctx_t* ctx, const float chw[], sfc_fixed_t cps, uint8_t mode) {
    const uint8_t clock = sfc_fixed_add(&ctx->clock, cps SFC_N163_IT1);
    famicom->apu.n163.n163_clock += clock;
    // 每15周期更新一次
    while (famicom->apu.n163.n163_clock >= 15) {
        famicom->apu.n163.n163_clock -= 15;
        sfc_n163_update(famicom);
    }
    // 输出声道
    float output = 0;
    const uint8_t start_index = famicom->apu.n163.history_index - mode;
    for (uint8_t i = 0; i != mode; ++i) {
        const uint8_t now_index = (start_index + i) & 7;
        const int8_t chn_out = famicom->apu.n163.history[now_index].output;
        const uint8_t chn_id = famicom->apu.n163.history[now_index].index;
        output += chw[chn_id] * (float)chn_out;
    }
    ctx->output = output / (float)mode;
}

/// <summary>
/// StepFC: N163 整型采样模式 - 更新副权重
/// </summary>
/// <param name="famicom">The famicom.</param>
/// <param name="ctx">The CTX.</param>
void sfc_n163_smi_update_subweight(const sfc_famicom_t* famicom, sfc_n163_smi_ctx_t* ctx) {
    ctx->subweight = (float)famicom->apu.n163.subweight_div16 / 16.f;
}