/**
 * @file         liblava/asset/image_loader.hpp
 * @brief        Load image data from file and memory
 * @authors      Lava Block OÜ and contributors
 * @copyright    Copyright (c) 2018-present, MIT License
 */

#pragma once

#include "liblava/file/file.hpp"
#include "liblava/util/math.hpp"

namespace lava {

/**
 * @brief Load image data from file and memory
 */
struct image_loader {
    /**
     * @brief Construct a new image data from file
     * @param filename    File data to load
     */
    explicit image_loader(string_ref filename);

    /**
     * @brief Construct a new image data from memory
     * @param image    Memory data to load
     */
    explicit image_loader(cdata::ref image);

    /**
     * @brief Destroy the image data
     */
    ~image_loader();

    /**
     * @brief Check if data is ready
     * @return Data is ready or not
     */
    bool ready() const {
        return data != nullptr;
    }

    /**
     * @brief Get image data
     * @return data_ptr    Image data pointer
     */
    data_cptr get() const {
        return data;
    }

    /**
     * @brief Get image data size
     * @return size_t    Image data size
     */
    size_t size() const {
        return channels * dimensions.x * dimensions.y;
    }

    /**
     * @brief Get image dimensions
     * @return uv2    Image dimensions
     */
    uv2 get_dimensions() const {
        return dimensions;
    }

    /**
     * @brief Get image channel count
     * @return ui32    Channel count
     */
    ui32 get_channels() const {
        return channels;
    }

private:
    /// Pointer to data
    data_ptr data = nullptr;

    /// Dimensions
    uv2 dimensions = uv2(0, 0);

    /// Number of channels
    ui32 channels = 0;
};

} // namespace lava
