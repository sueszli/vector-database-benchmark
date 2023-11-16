/**
 * @file         liblava/frame/gamepad.hpp
 * @brief        Gamepad manager
 * @authors      Lava Block OÜ and contributors
 * @copyright    Copyright (c) 2018-present, MIT License
 */

#pragma once

#include "liblava/core/id.hpp"

namespace lava {

/**
 * @brief Gamepad ids
 */
enum class gamepad_id : index {
    _1 = 0,
    _2,
    _3,
    _4,
    _5,
    _6,
    _7,
    _8,
    _9,
    _10,
    _11,
    _12,
    _13,
    _14,
    _15,
    _16,

    last = _16,
};

/// Reference to gamepad id
using gamepad_id_ref = gamepad_id const&;

/**
 * @brief Gamepad buttons
 */
enum class gamepad_button : index {
    a = 0,
    b,
    x,
    y,

    left_bumper,
    right_bumper,

    back,
    start,
    guide,

    left_thumb,
    right_thumb,

    dpad_up,
    dpad_right,
    dpad_down,
    dpad_left,

    last = dpad_left,

    cross = a,
    circle = b,
    square = x,
    triangle = y,
};

/// Reference to gamepad button
using gamepad_button_ref = gamepad_button const&;

/**
 * @brief Gamepad axis
 */
enum class gamepad_axis : index {
    left_x = 0,
    left_y,

    right_x,
    right_y,

    left_trigger,
    right_trigger,

    last = right_trigger,
};

/// Reference to gamepad axis
using gamepad_axis_ref = gamepad_axis const&;

/**
 * @brief Gamepad
 */
struct gamepad {
    /// List of gamepads
    using list = std::vector<gamepad>;

    /// Reference to gamepad
    using ref = gamepad const&;

    /**
     * @brief Construct a new gamepad
     * @param pad_id    Gamepad id
     */
    explicit gamepad(gamepad_id_ref pad_id = gamepad_id::_1);

    /**
     * @brief Check if gamepad is active
     * @return Gamepad is active or not
     */
    bool ready() const;

    /**
     * @brief Update gamepad
     * @return Update was successful or failed
     */
    bool update();

    /**
     * @brief Check if gamepad button is pressed
     * @param button    Gamepad button to check
     * @return Button is pressed or not
     */
    bool pressed(gamepad_button_ref button) const {
        return state.buttons[to_ui32(button)];
    }

    /**
     * @brief Get value of axis
     * @param axis    Target axis
     * @return r32    Axis value
     */
    r32 value(gamepad_axis_ref axis) const {
        return state.axes[to_ui32(axis)];
    }

    /**
     * @brief Get the gamepad id
     * @return gamepad_id_ref    Gamepad id
     */
    gamepad_id_ref get_pad_id() const {
        return pad_id;
    }

    /**
     * @brief Get the gamepad id as integer
     * @return ui32    Integer gamepad id
     */
    ui32 get_id() const {
        return to_ui32(get_pad_id());
    }

    /**
     * @brief Get the name
     * @return name    Name of gamepad
     */
    name get_name() const;

private:
    /// Gamepad id
    gamepad_id pad_id;

    /**
     * @brief Gamepad state
     */
    struct state {
        /// Gamepad buttons
        uchar buttons[15];

        /// Gamepad axes
        r32 axes[6];
    };

    /// Gamepad state
    gamepad::state state;
};

/**
 * @brief Get list of all gamepads
 * @return gamepad::list    List of gamepads
 */
gamepad::list gamepads();

/**
 * @brief Gamepad manager
 */
struct gamepad_manager {
    /// Gamepad listener function
    using listener_func = std::function<bool(gamepad, bool)>;

    /**
     * @brief Get gamepad manager singleton
     * @return gamepad_manager&    Gamepad manager
     */
    static gamepad_manager& singleton() {
        static gamepad_manager manager;
        return manager;
    }

    /**
     * @brief Add listener
     * @param listener    Gamepad listener function
     * @return id         Id of function
     */
    id add(listener_func listener);

    /**
     * @brief Remove listener
     * @param func_id    Id of function
     */
    void remove(id::ref func_id);

private:
    /**
     * @brief Construct a new gamepad manager
     */
    explicit gamepad_manager();

    /// Map of gamepad listeners
    using listener_map = std::map<id, listener_func>;

    /// Map of gamepad listeners
    listener_map map;
};

} // namespace lava
