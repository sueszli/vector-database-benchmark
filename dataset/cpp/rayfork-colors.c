#include "rayfork-colors.h"

#pragma region pixel format

rf_public const char* rf_pixel_format_string(rf_pixel_format format)
{
    switch (format)
    {
        case rf_pixel_format_grayscale: return "RF_UNCOMPRESSED_GRAYSCALE";
        case rf_pixel_format_gray_alpha: return "RF_UNCOMPRESSED_GRAY_ALPHA";
        case rf_pixel_format_r5g6b5: return "RF_UNCOMPRESSED_R5G6B5";
        case rf_pixel_format_r8g8b8: return "RF_UNCOMPRESSED_R8G8B8";
        case rf_pixel_format_r5g5b5a1: return "RF_UNCOMPRESSED_R5G5B5A1";
        case rf_pixel_format_r4g4b4a4: return "RF_UNCOMPRESSED_R4G4B4A4";
        case rf_pixel_format_r8g8b8a8: return "RF_UNCOMPRESSED_R8G8B8A8";
        case rf_pixel_format_r32: return "RF_UNCOMPRESSED_R32";
        case rf_pixel_format_r32g32b32: return "RF_UNCOMPRESSED_R32G32B32";
        case rf_pixel_format_r32g32b32a32: return "RF_UNCOMPRESSED_R32G32B32A32";
        case rf_pixel_format_dxt1_rgb: return "RF_COMPRESSED_DXT1_RGB";
        case rf_pixel_format_dxt1_rgba: return "RF_COMPRESSED_DXT1_RGBA";
        case rf_pixel_format_dxt3_rgba: return "RF_COMPRESSED_DXT3_RGBA";
        case rf_pixel_format_dxt5_rgba: return "RF_COMPRESSED_DXT5_RGBA";
        case rf_pixel_format_etc1_rgb: return "RF_COMPRESSED_ETC1_RGB";
        case rf_pixel_format_etc2_rgb: return "RF_COMPRESSED_ETC2_RGB";
        case rf_pixel_format_etc2_eac_rgba: return "RF_COMPRESSED_ETC2_EAC_RGBA";
        case rf_pixel_format_pvrt_rgb: return "RF_COMPRESSED_PVRT_RGB";
        case rf_pixel_format_prvt_rgba: return "RF_COMPRESSED_PVRT_RGBA";
        case rf_pixel_format_astc_4x4_rgba: return "RF_COMPRESSED_ASTC_4x4_RGBA";
        case rf_pixel_format_astc_8x8_rgba: return "RF_COMPRESSED_ASTC_8x8_RGBA";
        default: return NULL;
    }
}

rf_public rf_bool rf_is_uncompressed_format(rf_pixel_format format)
{
    return format >= rf_pixel_format_grayscale && format <= rf_pixel_format_r32g32b32a32;
}

rf_public rf_bool rf_is_compressed_format(rf_pixel_format format)
{
    return format >= rf_pixel_format_dxt1_rgb && format <= rf_pixel_format_astc_8x8_rgba;
}

rf_public int rf_bits_per_pixel(rf_pixel_format format)
{
    switch (format)
    {
        case rf_pixel_format_grayscale: return 8; // 8 bit per pixel (no alpha)
        case rf_pixel_format_gray_alpha: return 8 * 2; // 8 * 2 bpp (2 channels)
        case rf_pixel_format_r5g6b5: return 16; // 16 bpp
        case rf_pixel_format_r8g8b8: return 24; // 24 bpp
        case rf_pixel_format_r5g5b5a1: return 16; // 16 bpp (1 bit alpha)
        case rf_pixel_format_r4g4b4a4: return 16; // 16 bpp (4 bit alpha)
        case rf_pixel_format_r8g8b8a8: return 32; // 32 bpp
        case rf_pixel_format_r32: return 32; // 32 bpp (1 channel - float)
        case rf_pixel_format_r32g32b32: return 32 * 3; // 32 * 3 bpp (3 channels - float)
        case rf_pixel_format_r32g32b32a32: return 32 * 4; // 32 * 4 bpp (4 channels - float)
        case rf_pixel_format_dxt1_rgb: return 4; // 4 bpp (no alpha)
        case rf_pixel_format_dxt1_rgba: return 4; // 4 bpp (1 bit alpha)
        case rf_pixel_format_dxt3_rgba: return 8; // 8 bpp
        case rf_pixel_format_dxt5_rgba: return 8; // 8 bpp
        case rf_pixel_format_etc1_rgb: return 4; // 4 bpp
        case rf_pixel_format_etc2_rgb: return 4; // 4 bpp
        case rf_pixel_format_etc2_eac_rgba: return 8; // 8 bpp
        case rf_pixel_format_pvrt_rgb: return 4; // 4 bpp
        case rf_pixel_format_prvt_rgba: return 4; // 4 bpp
        case rf_pixel_format_astc_4x4_rgba: return 8; // 8 bpp
        case rf_pixel_format_astc_8x8_rgba: return 2; // 2 bpp
        default: return 0;
    }
}

rf_public int rf_bytes_per_pixel(rf_uncompressed_pixel_format format)
{
    switch (format)
    {
        case rf_pixel_format_grayscale: return 1;
        case rf_pixel_format_gray_alpha: return 2;
        case rf_pixel_format_r5g5b5a1: return 2;
        case rf_pixel_format_r5g6b5: return 2;
        case rf_pixel_format_r4g4b4a4: return 2;
        case rf_pixel_format_r8g8b8a8: return 4;
        case rf_pixel_format_r8g8b8: return 3;
        case rf_pixel_format_r32: return 4;
        case rf_pixel_format_r32g32b32: return 12;
        case rf_pixel_format_r32g32b32a32: return 16;
        default: return 0;
    }
}

rf_public int rf_pixel_buffer_size(int width, int height, rf_pixel_format format)
{
    return width * height * rf_bits_per_pixel(format) / 8;
}

rf_public rf_bool rf_format_pixels_to_normalized(const void* src, rf_int src_size, rf_uncompressed_pixel_format src_format, rf_vec4* dst, rf_int dst_size)
{
    rf_bool success = 0;

    rf_int src_bpp = rf_bytes_per_pixel(src_format);
    rf_int src_pixel_count = src_size / src_bpp;
    rf_int dst_pixel_count = dst_size / sizeof(rf_vec4);

    if (dst_pixel_count >= src_pixel_count)
    {
        if (src_format == rf_pixel_format_r32g32b32a32)
        {
            success = 1;
            memcpy(dst, src, src_size);
        }
        else
        {
            success = 1;

            #define RF_FOR_EACH_PIXEL for (rf_int dst_iter = 0, src_iter = 0; src_iter < src_size && dst_iter < dst_size; dst_iter++, src_iter += src_bpp)
            switch (src_format)
            {
                case rf_pixel_format_grayscale:
                    RF_FOR_EACH_PIXEL
                    {
                        float value = ((unsigned char*)src)[src_iter] / 255.0f;

                        dst[dst_iter].x = value;
                        dst[dst_iter].y = value;
                        dst[dst_iter].z = value;
                        dst[dst_iter].w = 1.0f;
                    }
                    break;

                case rf_pixel_format_gray_alpha:
                    RF_FOR_EACH_PIXEL
                    {
                        float value0 = (float)((unsigned char*)src)[src_iter + 0] / 255.0f;
                        float value1 = (float)((unsigned char*)src)[src_iter + 1] / 255.0f;

                        dst[dst_iter].x = value0;
                        dst[dst_iter].y = value0;
                        dst[dst_iter].z = value0;
                        dst[dst_iter].w = value1;
                    }
                    break;

                case rf_pixel_format_r5g5b5a1:
                    RF_FOR_EACH_PIXEL
                    {
                        unsigned short pixel = ((unsigned short*) src)[src_iter];

                        dst[dst_iter].x = (float)((pixel & 0b1111100000000000) >> 11) * (1.0f/31);
                        dst[dst_iter].y = (float)((pixel & 0b0000011111000000) >>  6) * (1.0f/31);
                        dst[dst_iter].z = (float)((pixel & 0b0000000000111110) >>  1) * (1.0f/31);
                        dst[dst_iter].w = ((pixel & 0b0000000000000001) == 0) ? 0.0f : 1.0f;
                    }
                    break;

                case rf_pixel_format_r5g6b5:
                    RF_FOR_EACH_PIXEL
                    {
                        unsigned short pixel = ((unsigned short*)src)[src_iter];

                        dst[dst_iter].x = (float)((pixel & 0b1111100000000000) >> 11) * (1.0f / 31);
                        dst[dst_iter].y = (float)((pixel & 0b0000011111100000) >>  5) * (1.0f / 63);
                        dst[dst_iter].z = (float) (pixel & 0b0000000000011111)        * (1.0f / 31);
                        dst[dst_iter].w = 1.0f;
                    }
                    break;

                case rf_pixel_format_r4g4b4a4:
                    RF_FOR_EACH_PIXEL
                    {
                        unsigned short pixel = ((unsigned short*)src)[src_iter];

                        dst[dst_iter].x = (float)((pixel & 0b1111000000000000) >> 12) * (1.0f / 15);
                        dst[dst_iter].y = (float)((pixel & 0b0000111100000000) >> 8)  * (1.0f / 15);
                        dst[dst_iter].z = (float)((pixel & 0b0000000011110000) >> 4)  * (1.0f / 15);
                        dst[dst_iter].w = (float) (pixel & 0b0000000000001111)        * (1.0f / 15);
                    }
                    break;

                case rf_pixel_format_r8g8b8a8:
                    RF_FOR_EACH_PIXEL
                    {
                        dst[dst_iter].x = (float)((unsigned char*)src)[src_iter + 0] / 255.0f;
                        dst[dst_iter].y = (float)((unsigned char*)src)[src_iter + 1] / 255.0f;
                        dst[dst_iter].z = (float)((unsigned char*)src)[src_iter + 2] / 255.0f;
                        dst[dst_iter].w = (float)((unsigned char*)src)[src_iter + 3] / 255.0f;
                    }
                    break;

                case rf_pixel_format_r8g8b8:
                    RF_FOR_EACH_PIXEL
                    {
                        dst[dst_iter].x = (float)((unsigned char*)src)[src_iter + 0] / 255.0f;
                        dst[dst_iter].y = (float)((unsigned char*)src)[src_iter + 1] / 255.0f;
                        dst[dst_iter].z = (float)((unsigned char*)src)[src_iter + 2] / 255.0f;
                        dst[dst_iter].w = 1.0f;
                    }
                    break;

                case rf_pixel_format_r32:
                    RF_FOR_EACH_PIXEL
                    {
                        dst[dst_iter].x = ((float*)src)[src_iter];
                        dst[dst_iter].y = 0.0f;
                        dst[dst_iter].z = 0.0f;
                        dst[dst_iter].w = 1.0f;
                    }
                    break;

                case rf_pixel_format_r32g32b32:
                    RF_FOR_EACH_PIXEL
                    {
                        dst[dst_iter].x = ((float*)src)[src_iter + 0];
                        dst[dst_iter].y = ((float*)src)[src_iter + 1];
                        dst[dst_iter].z = ((float*)src)[src_iter + 2];
                        dst[dst_iter].w = 1.0f;
                    }
                    break;

                case rf_pixel_format_r32g32b32a32:
                    RF_FOR_EACH_PIXEL
                    {
                        dst[dst_iter].x = ((float*)src)[src_iter + 0];
                        dst[dst_iter].y = ((float*)src)[src_iter + 1];
                        dst[dst_iter].z = ((float*)src)[src_iter + 2];
                        dst[dst_iter].w = ((float*)src)[src_iter + 3];
                    }
                    break;

                default: break;
            }
            #undef RF_FOR_EACH_PIXEL
        }
    }
    else rf_log_error(rf_bad_buffer_size, "Buffer is size %d but function expected a size of at least %d.", dst_size, src_pixel_count * sizeof(rf_vec4));

    return success;
}

rf_public rf_bool rf_format_pixels_to_rgba32(const void* src, rf_int src_size, rf_uncompressed_pixel_format src_format, rf_color* dst, rf_int dst_size)
{
    rf_bool success = 0;

    rf_int src_bpp = rf_bytes_per_pixel(src_format);
    rf_int src_pixel_count = src_size / src_bpp;
    rf_int dst_pixel_count = dst_size / sizeof(rf_color);

    if (dst_pixel_count >= src_pixel_count)
    {
        if (src_format == rf_pixel_format_r8g8b8a8)
        {
            success = 1;
            memcpy(dst, src, src_size);
        }
        else
        {
            success = 1;
            #define RF_FOR_EACH_PIXEL for (rf_int dst_iter = 0, src_iter = 0; src_iter < src_size && dst_iter < dst_size; dst_iter++, src_iter += src_bpp)
            switch (src_format)
            {
                case rf_pixel_format_grayscale:
                    RF_FOR_EACH_PIXEL
                    {
                        unsigned char value = ((unsigned char*) src)[src_iter];
                        dst[dst_iter].r = value;
                        dst[dst_iter].g = value;
                        dst[dst_iter].b = value;
                        dst[dst_iter].a = 255;
                    }
                    break;

                case rf_pixel_format_gray_alpha:
                    RF_FOR_EACH_PIXEL
                    {
                        unsigned char value0 = ((unsigned char*) src)[src_iter + 0];
                        unsigned char value1 = ((unsigned char*) src)[src_iter + 1];

                        dst[dst_iter].r = value0;
                        dst[dst_iter].g = value0;
                        dst[dst_iter].b = value0;
                        dst[dst_iter].a = value1;
                    }
                    break;

                case rf_pixel_format_r5g5b5a1:
                    RF_FOR_EACH_PIXEL
                    {
                        unsigned short pixel = ((unsigned short*) src)[src_iter];

                        dst[dst_iter].r = (unsigned char)((float)((pixel & 0b1111100000000000) >> 11) * (255 / 31));
                        dst[dst_iter].g = (unsigned char)((float)((pixel & 0b0000011111000000) >>  6) * (255 / 31));
                        dst[dst_iter].b = (unsigned char)((float)((pixel & 0b0000000000111110) >>  1) * (255 / 31));
                        dst[dst_iter].a = (unsigned char)        ((pixel & 0b0000000000000001)        *  255);
                    }
                    break;

                case rf_pixel_format_r5g6b5:
                    RF_FOR_EACH_PIXEL
                    {
                        unsigned short pixel = ((unsigned short*) src)[src_iter];

                        dst[dst_iter].r = (unsigned char)((float)((pixel & 0b1111100000000000) >> 11)* (255 / 31));
                        dst[dst_iter].g = (unsigned char)((float)((pixel & 0b0000011111100000) >>  5)* (255 / 63));
                        dst[dst_iter].b = (unsigned char)((float) (pixel & 0b0000000000011111)       * (255 / 31));
                        dst[dst_iter].a = 255;
                    }
                    break;

                case rf_pixel_format_r4g4b4a4:
                    RF_FOR_EACH_PIXEL
                    {
                        unsigned short pixel = ((unsigned short*) src)[src_iter];

                        dst[dst_iter].r = (unsigned char)((float)((pixel & 0b1111000000000000) >> 12) * (255 / 15));
                        dst[dst_iter].g = (unsigned char)((float)((pixel & 0b0000111100000000) >> 8)  * (255 / 15));
                        dst[dst_iter].b = (unsigned char)((float)((pixel & 0b0000000011110000) >> 4)  * (255 / 15));
                        dst[dst_iter].a = (unsigned char)((float) (pixel & 0b0000000000001111)        * (255 / 15));
                    }
                    break;

                case rf_pixel_format_r8g8b8a8:
                    RF_FOR_EACH_PIXEL
                    {
                        dst[dst_iter].r = ((unsigned char*) src)[src_iter + 0];
                        dst[dst_iter].g = ((unsigned char*) src)[src_iter + 1];
                        dst[dst_iter].b = ((unsigned char*) src)[src_iter + 2];
                        dst[dst_iter].a = ((unsigned char*) src)[src_iter + 3];
                    }
                    break;

                case rf_pixel_format_r8g8b8:
                    RF_FOR_EACH_PIXEL
                    {
                        dst[dst_iter].r = (unsigned char)((unsigned char*) src)[src_iter + 0];
                        dst[dst_iter].g = (unsigned char)((unsigned char*) src)[src_iter + 1];
                        dst[dst_iter].b = (unsigned char)((unsigned char*) src)[src_iter + 2];
                        dst[dst_iter].a = 255;
                    }
                    break;

                case rf_pixel_format_r32:
                    RF_FOR_EACH_PIXEL
                    {
                        dst[dst_iter].r = (unsigned char)(((float*) src)[src_iter + 0] * 255.0f);
                        dst[dst_iter].g = 0;
                        dst[dst_iter].b = 0;
                        dst[dst_iter].a = 255;
                    }
                    break;

                case rf_pixel_format_r32g32b32:
                    RF_FOR_EACH_PIXEL
                    {
                        dst[dst_iter].r = (unsigned char)(((float*) src)[src_iter + 0] * 255.0f);
                        dst[dst_iter].g = (unsigned char)(((float*) src)[src_iter + 1] * 255.0f);
                        dst[dst_iter].b = (unsigned char)(((float*) src)[src_iter + 2] * 255.0f);
                        dst[dst_iter].a = 255;
                    }
                    break;

                case rf_pixel_format_r32g32b32a32:
                    RF_FOR_EACH_PIXEL
                    {
                        dst[dst_iter].r = (unsigned char)(((float*) src)[src_iter + 0] * 255.0f);
                        dst[dst_iter].g = (unsigned char)(((float*) src)[src_iter + 1] * 255.0f);
                        dst[dst_iter].b = (unsigned char)(((float*) src)[src_iter + 2] * 255.0f);
                        dst[dst_iter].a = (unsigned char)(((float*) src)[src_iter + 3] * 255.0f);
                    }
                    break;

                default: break;
            }
            #undef RF_FOR_EACH_PIXEL
        }
    }
    else rf_log_error(rf_bad_buffer_size, "Buffer is size %d but function expected a size of at least %d", dst_size, src_pixel_count * sizeof(rf_color));

    return success;
}

rf_public rf_bool rf_format_pixels(const void* src, rf_int src_size, rf_uncompressed_pixel_format src_format, void* dst, rf_int dst_size, rf_uncompressed_pixel_format dst_format)
{
    rf_bool success = 0;

    if (rf_is_uncompressed_format(src_format) && dst_format == rf_pixel_format_r32g32b32a32)
    {
        success = rf_format_pixels_to_normalized(src, src_size, src_format, dst, dst_size);
    }
    else if (rf_is_uncompressed_format(src_format) && dst_format == rf_pixel_format_r8g8b8a8)
    {
        success = rf_format_pixels_to_rgba32(src, src_size, src_format, dst, dst_size);
    }
    else if (rf_is_uncompressed_format(src_format) && rf_is_uncompressed_format(dst_format))
    {
        rf_int src_bpp = rf_bytes_per_pixel(src_format);
        rf_int dst_bpp = rf_bytes_per_pixel(dst_format);

        rf_int src_pixel_count = src_size / src_bpp;
        rf_int dst_pixel_count = dst_size / dst_bpp;

        if (dst_pixel_count >= src_pixel_count)
        {
            success = 1;

            //Loop over both src and dst
            #define RF_FOR_EACH_PIXEL for (rf_int src_iter = 0, dst_iter = 0; src_iter < src_size && dst_iter < dst_size; src_iter += src_bpp, dst_iter += dst_bpp)
            #define RF_COMPUTE_NORMALIZED_PIXEL() rf_format_one_pixel_to_normalized(((unsigned char*) src) + src_iter, src_format);
            if (src_format == dst_format)
            {
                memcpy(dst, src, src_size);
            }
            else
            {
                switch (dst_format)
                {
                    case rf_pixel_format_grayscale:
                        RF_FOR_EACH_PIXEL
                        {
                            rf_vec4 normalized = RF_COMPUTE_NORMALIZED_PIXEL();
                            ((unsigned char*)dst)[dst_iter] = (unsigned char)((normalized.x * 0.299f + normalized.y * 0.587f + normalized.z * 0.114f) * 255.0f);
                        }
                        break;

                    case rf_pixel_format_gray_alpha:
                        RF_FOR_EACH_PIXEL
                        {
                            rf_vec4 normalized = RF_COMPUTE_NORMALIZED_PIXEL();

                            ((unsigned char*)dst)[dst_iter    ] = (unsigned char)((normalized.x * 0.299f + (float)normalized.y * 0.587f + (float)normalized.z * 0.114f) * 255.0f);
                            ((unsigned char*)dst)[dst_iter + 1] = (unsigned char) (normalized.w * 255.0f);
                        }
                        break;

                    case rf_pixel_format_r5g6b5:
                        RF_FOR_EACH_PIXEL
                        {
                            rf_vec4 normalized = RF_COMPUTE_NORMALIZED_PIXEL();

                            unsigned char r = (unsigned char)(round(normalized.x * 31.0f));
                            unsigned char g = (unsigned char)(round(normalized.y * 63.0f));
                            unsigned char b = (unsigned char)(round(normalized.z * 31.0f));

                            ((unsigned short*)dst)[dst_iter] = (unsigned short)r << 11 | (unsigned short)g << 5 | (unsigned short)b;
                        }
                        break;

                    case rf_pixel_format_r8g8b8:
                        RF_FOR_EACH_PIXEL
                        {
                            rf_vec4 normalized = RF_COMPUTE_NORMALIZED_PIXEL();

                            ((unsigned char*)dst)[dst_iter    ] = (unsigned char)(normalized.x * 255.0f);
                            ((unsigned char*)dst)[dst_iter + 1] = (unsigned char)(normalized.y * 255.0f);
                            ((unsigned char*)dst)[dst_iter + 2] = (unsigned char)(normalized.z * 255.0f);
                        }
                        break;

                    case rf_pixel_format_r5g5b5a1:
                        RF_FOR_EACH_PIXEL
                        {
                            rf_vec4 normalized = RF_COMPUTE_NORMALIZED_PIXEL();

                            int ALPHA_THRESHOLD = 50;
                            unsigned char r = (unsigned char)(round(normalized.x * 31.0f));
                            unsigned char g = (unsigned char)(round(normalized.y * 31.0f));
                            unsigned char b = (unsigned char)(round(normalized.z * 31.0f));
                            unsigned char a = (normalized.w > ((float)ALPHA_THRESHOLD / 255.0f)) ? 1 : 0;

                            ((unsigned short*)dst)[dst_iter] = (unsigned short)r << 11 | (unsigned short)g << 6 | (unsigned short)b << 1 | (unsigned short)a;
                        }
                        break;

                    case rf_pixel_format_r4g4b4a4:
                        RF_FOR_EACH_PIXEL
                        {
                            rf_vec4 normalized = RF_COMPUTE_NORMALIZED_PIXEL();

                            unsigned char r = (unsigned char)(round(normalized.x * 15.0f));
                            unsigned char g = (unsigned char)(round(normalized.y * 15.0f));
                            unsigned char b = (unsigned char)(round(normalized.z * 15.0f));
                            unsigned char a = (unsigned char)(round(normalized.w * 15.0f));

                            ((unsigned short*)dst)[dst_iter] = (unsigned short)r << 12 | (unsigned short)g << 8 | (unsigned short)b << 4 | (unsigned short)a;
                        }
                        break;

                    case rf_pixel_format_r32:
                        RF_FOR_EACH_PIXEL
                        {
                            rf_vec4 normalized = RF_COMPUTE_NORMALIZED_PIXEL();

                            ((float*)dst)[dst_iter] = (float)(normalized.x * 0.299f + normalized.y * 0.587f + normalized.z * 0.114f);
                        }
                        break;

                    case rf_pixel_format_r32g32b32:
                        RF_FOR_EACH_PIXEL
                        {
                            rf_vec4 normalized = RF_COMPUTE_NORMALIZED_PIXEL();

                            ((float*)dst)[dst_iter    ] = normalized.x;
                            ((float*)dst)[dst_iter + 1] = normalized.y;
                            ((float*)dst)[dst_iter + 2] = normalized.z;
                        }
                        break;

                    default: break;
                }
            }
            #undef RF_FOR_EACH_PIXEL
            #undef RF_COMPUTE_NORMALIZED_PIXEL
        }
        else rf_log_error(rf_bad_buffer_size, "Buffer is size %d but function expected a size of at least %d.", dst_size, src_pixel_count * dst_bpp);
    }
    else rf_log_error(rf_bad_argument, "Function expected uncompressed pixel formats. Source format: %d, Destination format: %d.", src_format, dst_format);

    return success;
}

rf_public rf_vec4 rf_format_one_pixel_to_normalized(const void* src, rf_uncompressed_pixel_format src_format)
{
    rf_vec4 result = {0};

    switch (src_format)
    {
        case rf_pixel_format_grayscale:
        {
            float value = ((unsigned char*)src)[0] / 255.0f;

            result.x = value;
            result.y = value;
            result.z = value;
            result.w = 1.0f;
        }
        break;

        case rf_pixel_format_gray_alpha:
        {
            float value0 = (float)((unsigned char*)src)[0] / 255.0f;
            float value1 = (float)((unsigned char*)src)[1] / 255.0f;

            result.x = value0;
            result.y = value0;
            result.z = value0;
            result.w = value1;
        }
        break;

        case rf_pixel_format_r5g5b5a1:
        {
            unsigned short pixel = ((unsigned short*) src)[0];

            result.x = (float)((pixel & 0b1111100000000000) >> 11) * (1.0f/31);
            result.y = (float)((pixel & 0b0000011111000000) >>  6) * (1.0f/31);
            result.z = (float)((pixel & 0b0000000000111110) >>  1) * (1.0f/31);
            result.w = ((pixel & 0b0000000000000001) == 0) ? 0.0f : 1.0f;
        }
        break;

        case rf_pixel_format_r5g6b5:
        {
            unsigned short pixel = ((unsigned short*)src)[0];

            result.x = (float)((pixel & 0b1111100000000000) >> 11) * (1.0f / 31);
            result.y = (float)((pixel & 0b0000011111100000) >>  5) * (1.0f / 63);
            result.z = (float) (pixel & 0b0000000000011111)        * (1.0f / 31);
            result.w = 1.0f;
        }
        break;

        case rf_pixel_format_r4g4b4a4:
        {
            unsigned short pixel = ((unsigned short*)src)[0];

            result.x = (float)((pixel & 0b1111000000000000) >> 12) * (1.0f / 15);
            result.y = (float)((pixel & 0b0000111100000000) >> 8)  * (1.0f / 15);
            result.z = (float)((pixel & 0b0000000011110000) >> 4)  * (1.0f / 15);
            result.w = (float) (pixel & 0b0000000000001111)        * (1.0f / 15);
        }
        break;

        case rf_pixel_format_r8g8b8a8:
        {
            result.x = (float)((unsigned char*)src)[0] / 255.0f;
            result.y = (float)((unsigned char*)src)[1] / 255.0f;
            result.z = (float)((unsigned char*)src)[2] / 255.0f;
            result.w = (float)((unsigned char*)src)[3] / 255.0f;
        }
        break;

        case rf_pixel_format_r8g8b8:
        {
            result.x = (float)((unsigned char*)src)[0] / 255.0f;
            result.y = (float)((unsigned char*)src)[1] / 255.0f;
            result.z = (float)((unsigned char*)src)[2] / 255.0f;
            result.w = 1.0f;
        }
        break;

        case rf_pixel_format_r32:
        {
            result.x = ((float*)src)[0];
            result.y = 0.0f;
            result.z = 0.0f;
            result.w = 1.0f;
        }
        break;

        case rf_pixel_format_r32g32b32:
        {
            result.x = ((float*)src)[0];
            result.y = ((float*)src)[1];
            result.z = ((float*)src)[2];
            result.w = 1.0f;
        }
        break;

        case rf_pixel_format_r32g32b32a32:
        {
            result.x = ((float*)src)[0];
            result.y = ((float*)src)[1];
            result.z = ((float*)src)[2];
            result.w = ((float*)src)[3];
        }
        break;

        default: break;
    }

    return result;
}

rf_public rf_color rf_format_one_pixel_to_rgba32(const void* src, rf_uncompressed_pixel_format src_format)
{
    rf_color result = {0};

    switch (src_format)
    {
        case rf_pixel_format_grayscale:
        {
            unsigned char value = ((unsigned char*) src)[0];
            result.r = value;
            result.g = value;
            result.b = value;
            result.a = 255;
        }
        break;

        case rf_pixel_format_gray_alpha:
        {
            unsigned char value0 = ((unsigned char*) src)[0];
            unsigned char value1 = ((unsigned char*) src)[1];

            result.r = value0;
            result.g = value0;
            result.b = value0;
            result.a = value1;
        }
        break;

        case rf_pixel_format_r5g5b5a1:
        {
            unsigned short pixel = ((unsigned short*) src)[0];

            result.r = (unsigned char)((float)((pixel & 0b1111100000000000) >> 11) * (255 / 31));
            result.g = (unsigned char)((float)((pixel & 0b0000011111000000) >>  6) * (255 / 31));
            result.b = (unsigned char)((float)((pixel & 0b0000000000111110) >>  1) * (255 / 31));
            result.a = (unsigned char)        ((pixel & 0b0000000000000001)        *  255);
        }
        break;

        case rf_pixel_format_r5g6b5:
        {
            unsigned short pixel = ((unsigned short*) src)[0];

            result.r = (unsigned char)((float)((pixel & 0b1111100000000000) >> 11)* (255 / 31));
            result.g = (unsigned char)((float)((pixel & 0b0000011111100000) >>  5)* (255 / 63));
            result.b = (unsigned char)((float) (pixel & 0b0000000000011111)       * (255 / 31));
            result.a = 255;
        }
        break;

        case rf_pixel_format_r4g4b4a4:
        {
            unsigned short pixel = ((unsigned short*) src)[0];

            result.r = (unsigned char)((float)((pixel & 0b1111000000000000) >> 12) * (255 / 15));
            result.g = (unsigned char)((float)((pixel & 0b0000111100000000) >> 8)  * (255 / 15));
            result.b = (unsigned char)((float)((pixel & 0b0000000011110000) >> 4)  * (255 / 15));
            result.a = (unsigned char)((float) (pixel & 0b0000000000001111)        * (255 / 15));
        }
        break;

        case rf_pixel_format_r8g8b8a8:
        {
            result.r = ((unsigned char*) src)[0];
            result.g = ((unsigned char*) src)[1];
            result.b = ((unsigned char*) src)[2];
            result.a = ((unsigned char*) src)[3];
        }
        break;

        case rf_pixel_format_r8g8b8:
        {
            result.r = (unsigned char)((unsigned char*) src)[0];
            result.g = (unsigned char)((unsigned char*) src)[1];
            result.b = (unsigned char)((unsigned char*) src)[2];
            result.a = 255;
        }
        break;

        case rf_pixel_format_r32:
        {
            result.r = (unsigned char)(((float*) src)[0] * 255.0f);
            result.g = 0;
            result.b = 0;
            result.a = 255;
        }
        break;

        case rf_pixel_format_r32g32b32:
        {
            result.r = (unsigned char)(((float*) src)[0] * 255.0f);
            result.g = (unsigned char)(((float*) src)[1] * 255.0f);
            result.b = (unsigned char)(((float*) src)[2] * 255.0f);
            result.a = 255;
        }
        break;

        case rf_pixel_format_r32g32b32a32:
        {
            result.r = (unsigned char)(((float*) src)[0] * 255.0f);
            result.g = (unsigned char)(((float*) src)[1] * 255.0f);
            result.b = (unsigned char)(((float*) src)[2] * 255.0f);
            result.a = (unsigned char)(((float*) src)[3] * 255.0f);
        }
        break;

        default: break;
    }

    return result;
}

rf_public void rf_format_one_pixel(const void* src, rf_uncompressed_pixel_format src_format, void* dst, rf_uncompressed_pixel_format dst_format)
{
    if (src_format == dst_format && rf_is_uncompressed_format(src_format) && rf_is_uncompressed_format(dst_format))
    {
        memcpy(dst, src, rf_bytes_per_pixel(src_format));
    }
    else if (rf_is_uncompressed_format(src_format) && dst_format == rf_pixel_format_r32g32b32a32)
    {
        *((rf_vec4*)dst) = rf_format_one_pixel_to_normalized(src, src_format);
    }
    else if (rf_is_uncompressed_format(src_format) && dst_format == rf_pixel_format_r8g8b8a8)
    {
        *((rf_color*)dst) = rf_format_one_pixel_to_rgba32(src, src_format);
    }
    else if (rf_is_uncompressed_format(src_format) && rf_is_uncompressed_format(dst_format))
    {
        switch (dst_format)
        {
            case rf_pixel_format_grayscale:
            {
                rf_vec4 normalized = rf_format_one_pixel_to_normalized(src, src_format);
                ((unsigned char*)dst)[0] = (unsigned char)((normalized.x * 0.299f + normalized.y * 0.587f + normalized.z * 0.114f) * 255.0f);
            }
            break;

            case rf_pixel_format_gray_alpha:
            {
                rf_vec4 normalized = rf_format_one_pixel_to_normalized(src, src_format);
                ((unsigned char*)dst)[0    ] = (unsigned char)((normalized.x * 0.299f + (float)normalized.y * 0.587f + (float)normalized.z * 0.114f) * 255.0f);
                ((unsigned char*)dst)[0 + 1] = (unsigned char) (normalized.w * 255.0f);
            }
            break;

            case rf_pixel_format_r5g6b5:
            {
                rf_vec4 normalized = rf_format_one_pixel_to_normalized(src, src_format);
                unsigned char r = (unsigned char)(round(normalized.x * 31.0f));
                unsigned char g = (unsigned char)(round(normalized.y * 63.0f));
                unsigned char b = (unsigned char)(round(normalized.z * 31.0f));

                ((unsigned short*)dst)[0] = (unsigned short)r << 11 | (unsigned short)g << 5 | (unsigned short)b;
            }
            break;

            case rf_pixel_format_r8g8b8:
            {
                rf_vec4 normalized = rf_format_one_pixel_to_normalized(src, src_format);
                ((unsigned char*)dst)[0    ] = (unsigned char)(normalized.x * 255.0f);
                ((unsigned char*)dst)[0 + 1] = (unsigned char)(normalized.y * 255.0f);
                ((unsigned char*)dst)[0 + 2] = (unsigned char)(normalized.z * 255.0f);
            }
            break;

            case rf_pixel_format_r5g5b5a1:
            {
                rf_vec4 normalized = rf_format_one_pixel_to_normalized(src, src_format);
                int ALPHA_THRESHOLD = 50;
                unsigned char r = (unsigned char)(round(normalized.x * 31.0f));
                unsigned char g = (unsigned char)(round(normalized.y * 31.0f));
                unsigned char b = (unsigned char)(round(normalized.z * 31.0f));
                unsigned char a = (normalized.w > ((float)ALPHA_THRESHOLD / 255.0f)) ? 1 : 0;

                ((unsigned short*)dst)[0] = (unsigned short)r << 11 | (unsigned short)g << 6 | (unsigned short)b << 1 | (unsigned short)a;
            }
            break;

            case rf_pixel_format_r4g4b4a4:
            {
                rf_vec4 normalized = rf_format_one_pixel_to_normalized(src, src_format);
                unsigned char r = (unsigned char)(round(normalized.x * 15.0f));
                unsigned char g = (unsigned char)(round(normalized.y * 15.0f));
                unsigned char b = (unsigned char)(round(normalized.z * 15.0f));
                unsigned char a = (unsigned char)(round(normalized.w * 15.0f));

                ((unsigned short*)dst)[0] = (unsigned short)r << 12 | (unsigned short)g << 8 | (unsigned short)b << 4 | (unsigned short)a;
            }
            break;

            case rf_pixel_format_r8g8b8a8:
            {
                rf_vec4 normalized = rf_format_one_pixel_to_normalized(src, src_format);
                ((unsigned char*)dst)[0    ] = (unsigned char)(normalized.x * 255.0f);
                ((unsigned char*)dst)[0 + 1] = (unsigned char)(normalized.y * 255.0f);
                ((unsigned char*)dst)[0 + 2] = (unsigned char)(normalized.z * 255.0f);
                ((unsigned char*)dst)[0 + 3] = (unsigned char)(normalized.w * 255.0f);
            }
            break;

            case rf_pixel_format_r32:
            {
                rf_vec4 normalized = rf_format_one_pixel_to_normalized(src, src_format);
                ((float*)dst)[0] = (float)(normalized.x * 0.299f + normalized.y * 0.587f + normalized.z * 0.114f);
            }
            break;

            case rf_pixel_format_r32g32b32:
            {
                rf_vec4 normalized = rf_format_one_pixel_to_normalized(src, src_format);
                ((float*)dst)[0    ] = normalized.x;
                ((float*)dst)[0 + 1] = normalized.y;
                ((float*)dst)[0 + 2] = normalized.z;
            }
            break;

            case rf_pixel_format_r32g32b32a32:
            {
                rf_vec4 normalized = rf_format_one_pixel_to_normalized(src, src_format);
                ((float*)dst)[0    ] = normalized.x;
                ((float*)dst)[0 + 1] = normalized.y;
                ((float*)dst)[0 + 2] = normalized.z;
                ((float*)dst)[0 + 3] = normalized.w;
            }
            break;

            default: break;
        }
    }
}

#pragma endregion

#pragma region color

// Returns 1 if the two colors have the same values for the rgb components
rf_public rf_bool rf_color_match_rgb(rf_color a, rf_color b)
{
    return a.r == b.r && a.g == b.g && a.b == b.b;
}

// Returns 1 if the two colors have the same values
rf_public rf_bool rf_color_match(rf_color a, rf_color b)
{
    return a.r == b.r && a.g == b.g && a.b == b.b && a.a == b.a;
}

// Returns hexadecimal value for a rf_color
rf_public int rf_color_to_int(rf_color color)
{
    return (((int) color.r << 24) | ((int) color.g << 16) | ((int) color.b << 8) | (int) color.a);
}

// Returns color normalized as float [0..1]
rf_public rf_vec4 rf_color_normalize(rf_color color)
{
    rf_vec4 result;

    result.x = (float) color.r / 255.0f;
    result.y = (float) color.g / 255.0f;
    result.z = (float) color.b / 255.0f;
    result.w = (float) color.a / 255.0f;

    return result;
}

// Returns color from normalized values [0..1]
rf_public rf_color rf_color_from_normalized(rf_vec4 normalized)
{
    rf_color result;

    result.r = normalized.x * 255.0f;
    result.g = normalized.y * 255.0f;
    result.b = normalized.z * 255.0f;
    result.a = normalized.w * 255.0f;

    return result;
}

// Returns HSV values for a rf_color. Hue is returned as degrees [0..360]
rf_public rf_vec3 rf_color_to_hsv(rf_color color)
{
    rf_vec3 rgb = { (float) color.r / 255.0f, (float) color.g / 255.0f, (float) color.b / 255.0f };
    rf_vec3 hsv = { 0.0f, 0.0f, 0.0f };
    float min, max, delta;

    min = rgb.x < rgb.y ? rgb.x : rgb.y;
    min = min < rgb.z ? min : rgb.z;

    max = rgb.x > rgb.y ? rgb.x : rgb.y;
    max = max > rgb.z ? max : rgb.z;

    hsv.z = max; // Value
    delta = max - min;

    if (delta < 0.00001f)
    {
        hsv.y = 0.0f;
        hsv.x = 0.0f; // Undefined, maybe NAN?
        return hsv;
    }

    if (max > 0.0f)
    {
        // NOTE: If max is 0, this divide would cause a crash
        hsv.y = (delta / max); // Saturation
    }
    else
    {
        // NOTE: If max is 0, then r = g = b = 0, s = 0, h is undefined
        hsv.y = 0.0f;
        hsv.x = NAN; // Undefined
        return hsv;
    }

    // NOTE: Comparing float values could not work properly
    if (rgb.x >= max) hsv.x = (rgb.y - rgb.z) / delta; // Between yellow & magenta
    else
    {
        if (rgb.y >= max) hsv.x = 2.0f + (rgb.z - rgb.x) / delta; // Between cyan & yellow
        else hsv.x = 4.0f + (rgb.x - rgb.y) / delta; // Between magenta & cyan
    }

    hsv.x *= 60.0f; // Convert to degrees

    if (hsv.x < 0.0f) hsv.x += 360.0f;

    return hsv;
}

// Returns a rf_color from HSV values. rf_color->HSV->rf_color conversion will not yield exactly the same color due to rounding errors. Implementation reference: https://en.wikipedia.org/wiki/HSL_and_HSV#Alternative_HSV_conversion
rf_public rf_color rf_color_from_hsv(rf_vec3 hsv)
{
    rf_color color = {0, 0, 0, 255};
    float h = hsv.x, s = hsv.y, v = hsv.z;

// Red channel
    float k = fmod((5.0f + h / 60.0f), 6);
    float t = 4.0f - k;
    k = (t < k) ? t : k;
    k = (k < 1) ? k : 1;
    k = (k > 0) ? k : 0;
    color.r = (v - v * s * k) * 255;

// Green channel
    k = fmod((3.0f + h / 60.0f), 6);
    t = 4.0f - k;
    k = (t < k) ? t : k;
    k = (k < 1) ? k : 1;
    k = (k > 0) ? k : 0;
    color.g = (v - v * s * k) * 255;

// Blue channel
    k = fmod((1.0f + h / 60.0f), 6);
    t = 4.0f - k;
    k = (t < k) ? t : k;
    k = (k < 1) ? k : 1;
    k = (k > 0) ? k : 0;
    color.b = (v - v * s * k) * 255;

    return color;
}

// Returns a rf_color struct from hexadecimal value
rf_public rf_color rf_color_from_int(int hex_value)
{
    rf_color color;

    color.r = (unsigned char) (hex_value >> 24) & 0xFF;
    color.g = (unsigned char) (hex_value >> 16) & 0xFF;
    color.b = (unsigned char) (hex_value >> 8) & 0xFF;
    color.a = (unsigned char) hex_value & 0xFF;

    return color;
}

// rf_color fade-in or fade-out, alpha goes from 0.0f to 1.0f
rf_public rf_color rf_fade(rf_color color, float alpha)
{
    if (alpha < 0.0f) alpha = 0.0f;
    else if (alpha > 1.0f) alpha = 1.0f;

    return (rf_color) {color.r, color.g, color.b, (unsigned char) (255.0f * alpha)};
}

#pragma endregion