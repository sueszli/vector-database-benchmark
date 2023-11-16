#if defined(RASPBERRY_PI)

#include <math.h>

#include "compat.h"

#include "tables.h"

/* Convert (little-endian) SC16 values with a range of -2048..+2047 to unsigned 16-bit magnitudes */

void STARCH_IMPL(magnitude_sc16q11, exact_u32) (const sc16_t *in, uint16_t *out, unsigned len)
{
    const sc16_t * restrict in_align = STARCH_ALIGNED(in);
    uint16_t * restrict out_align = STARCH_ALIGNED(out);

    while (len--) {
        uint32_t I = abs((int16_t) le16toh(in_align[0].I));
        uint32_t Q = abs((int16_t) le16toh(in_align[0].Q));

        uint32_t magsq = I * I + Q * Q;
        float mag = sqrtf(magsq) * 32;
        if (mag > 65535.0)
            mag = 65535.0;
        out_align[0] = (uint16_t)mag;

        out_align += 1;
        in_align += 1;
    }
}

void STARCH_IMPL(magnitude_sc16q11, exact_float) (const sc16_t *in, uint16_t *out, unsigned len)
{
    const sc16_t * restrict in_align = STARCH_ALIGNED(in);
    uint16_t * restrict out_align = STARCH_ALIGNED(out);

    while (len--) {
        float I = abs((int16_t) le16toh(in_align[0].I)) * 32;
        float Q = abs((int16_t) le16toh(in_align[0].Q)) * 32;

        float magsq = I * I + Q * Q;
        float mag = sqrtf(magsq);
        if (mag > 65535.0)
            mag = 65535.0;
        out_align[0] = (uint16_t)mag;

        out_align += 1;
        in_align += 1;
    }
}

void STARCH_IMPL(magnitude_sc16q11, 11bit_table) (const sc16_t *in, uint16_t *out, unsigned len)
{
    const uint16_t * restrict table = get_sc16q11_mag_11bit_table();
    const sc16_t * restrict in_align = STARCH_ALIGNED(in);
    uint16_t * restrict out_align = STARCH_ALIGNED(out);

    while (len--) {
        uint16_t I = abs((int16_t)le16toh(in_align[0].I));
        if (I >= 2048)
            I = 2047;
        uint16_t Q = abs((int16_t)le16toh(in_align[0].Q));
        if (Q >= 2048)
            Q = 2047;
        out_align[0] = table[(Q << 11) | I];

        in_align += 1;
        out_align += 1;
    }
}

void STARCH_IMPL(magnitude_sc16q11, 12bit_table) (const sc16_t *in, uint16_t *out, unsigned len)
{
    const uint16_t * restrict table = get_sc16q11_mag_12bit_table();
    const sc16_t * restrict in_align = STARCH_ALIGNED(in);
    uint16_t * restrict out_align = STARCH_ALIGNED(out);

    while (len--) {
        unsigned index = ((in_align[0].I & 4095) << 12) | (in_align[0].Q & 4095);
        out_align[0] = table[index];

        in_align += 1;
        out_align += 1;
    }
}

#ifdef STARCH_FEATURE_NEON

#include <arm_neon.h>

void STARCH_IMPL_REQUIRES(magnitude_sc16q11, neon_vrsqrte, STARCH_FEATURE_NEON) (const sc16_t *in, uint16_t *out, unsigned len)
{
    const int16_t * restrict in_align = (const int16_t *) STARCH_ALIGNED(in);
    uint16_t * restrict out_align = STARCH_ALIGNED(out);

    /* This uses NEON's floating-point reciprocal square root estimate instruction (vrsqrte).
     * The estimate is accurate to about 9 bits of mantissa, which is good enough for our purposes.
     */

    unsigned len4 = len >> 2;
    while (len4--) {
        int16x4x2_t iq = vld2_s16(in_align);
        int16x4_t i16 = iq.val[0]; /* Q11 */
        int16x4_t q16 = iq.val[1]; /* Q11 */

        uint32x4_t isq = vreinterpretq_u32_s32(vmull_s16(i16, i16)); /* Q22, unsigned */
        uint32x4_t qsq = vreinterpretq_u32_s32(vmull_s16(q16, q16)); /* Q22, unsigned */
        uint32x4_t magsq = vqaddq_u32(isq, qsq);                     /* Q22, unsigned */

        float32x4_t magsq_f32 = vcvtq_n_f32_u32(magsq, 22);
        float32x4_t mag_f32 = vmulq_f32(magsq_f32, vrsqrteq_f32(magsq_f32));  /* sqrt(x) = x * (1/sqrt(x)) */
        uint16x4_t mag_u16 = vqmovn_u32(vcvtq_n_u32_f32(mag_f32, 16));

        vst1_u16(out_align, mag_u16);

        in_align += 8;
        out_align += 4;
    }

    unsigned len1 = len & 3;
    while (len1--) {
        int16x4x2_t iq = vld2_dup_s16(in_align);
        int16x4_t i16 = iq.val[0];
        int16x4_t q16 = iq.val[1];

        uint32x4_t isq = vreinterpretq_u32_s32(vmull_s16(i16, i16));
        uint32x4_t qsq = vreinterpretq_u32_s32(vmull_s16(q16, q16));
        uint32x4_t magsq = vqaddq_u32(isq, qsq);

        float32x4_t magsq_f32 = vcvtq_n_f32_u32(magsq, 22);
        float32x4_t mag_f32 = vmulq_f32(magsq_f32, vrsqrteq_f32(magsq_f32));
        uint16x4_t mag_u16 = vqmovn_u32(vcvtq_n_u32_f32(mag_f32, 16));

        vst1_lane_u16(out_align, mag_u16, 0);

        in_align += 2;
        out_align += 1;
    }
}

#endif /* STARCH_FEATURE_NEON */

#endif /* RASPBERRY_PI */
