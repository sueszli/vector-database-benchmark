/*
 * Copyright (c) 2018 Particle Industries, Inc.  All rights reserved.
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation, either
 * version 3 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, see <http://www.gnu.org/licenses/>.
 */
#include "button_hal.h"
#include "nrfx_timer.h"
#include "platform_config.h"
#include "interrupts_hal.h"
#include "pinmap_impl.h"
#include "nrfx_gpiote.h"
#include "gpio_hal.h"
#include "core_hal.h"
#include "dct.h"
#include "logging.h"

hal_button_config_t HAL_Buttons[] = {
    {
        .active         = false,
        .pin            = BUTTON1_PIN,
        .interrupt_mode = BUTTON1_INTERRUPT_MODE,
        .debounce_time  = 0,
    },
    {
        .active         = false,
        .pin            = BUTTON1_MIRROR_PIN,
        .interrupt_mode = BUTTON1_MIRROR_INTERRUPT_MODE,
        .debounce_time  = 0
    }
};

static volatile struct {
    bool        enable;
    uint32_t    timeout;
} systick_button_timer;

static void button_timer_init(void);
static void button_timer_uninit(void);
static void button_timer_start(void);
static void button_timer_stop(void);
static void button_timer_event_handler(void);
static void button_reset(uint16_t button);

static void button_timer_init(void) {
    systick_button_timer.enable = false;
    systick_button_timer.timeout = 0;
}

static void button_timer_uninit(void) {
    systick_button_timer.enable = false;
    systick_button_timer.timeout = 0;
}

static void button_timer_start(void) {
    if (!systick_button_timer.enable) {
        systick_button_timer.enable = true;
        systick_button_timer.timeout = BUTTON_DEBOUNCE_INTERVAL;
    }
}

static void button_timer_stop(void) {
    systick_button_timer.enable = false;
    systick_button_timer.timeout = 0;
}

static void button_timer_event_handler(void) {
    if (HAL_Buttons[HAL_BUTTON1].active && (hal_button_get_state(HAL_BUTTON1) == BUTTON1_PRESSED)) {
        if (!HAL_Buttons[HAL_BUTTON1].debounce_time) {
            HAL_Buttons[HAL_BUTTON1].debounce_time += BUTTON_DEBOUNCE_INTERVAL;
#if MODULE_FUNCTION != MOD_FUNC_BOOTLOADER
            HAL_Notify_Button_State(HAL_BUTTON1, true); 
#endif
        }
        HAL_Buttons[HAL_BUTTON1].debounce_time += BUTTON_DEBOUNCE_INTERVAL;
    } else if (HAL_Buttons[HAL_BUTTON1].active) {
        HAL_Buttons[HAL_BUTTON1].active = false;
        button_reset(HAL_BUTTON1);
    }

    if ((HAL_Buttons[HAL_BUTTON1_MIRROR].pin != PIN_INVALID) && HAL_Buttons[HAL_BUTTON1_MIRROR].active &&
        hal_button_get_state(HAL_BUTTON1_MIRROR) == (HAL_Buttons[HAL_BUTTON1_MIRROR].interrupt_mode == RISING ? 1 : 0)) 
    {
        if (!HAL_Buttons[HAL_BUTTON1_MIRROR].debounce_time) {
            HAL_Buttons[HAL_BUTTON1_MIRROR].debounce_time += BUTTON_DEBOUNCE_INTERVAL;
#if MODULE_FUNCTION != MOD_FUNC_BOOTLOADER
            HAL_Notify_Button_State(HAL_BUTTON1_MIRROR, true);
#endif
        }
        HAL_Buttons[HAL_BUTTON1_MIRROR].debounce_time += BUTTON_DEBOUNCE_INTERVAL;
    } else if ((HAL_Buttons[HAL_BUTTON1_MIRROR].pin != PIN_INVALID) && HAL_Buttons[HAL_BUTTON1_MIRROR].active) {
        HAL_Buttons[HAL_BUTTON1_MIRROR].active = false;
        button_reset(HAL_BUTTON1_MIRROR);
    }
}

static void button_reset(uint16_t button) {
    HAL_Buttons[button].debounce_time = 0x00;

    if (!HAL_Buttons[HAL_BUTTON1].active && !HAL_Buttons[HAL_BUTTON1_MIRROR].active) {
        button_timer_stop();
    }

#if MODULE_FUNCTION != MOD_FUNC_BOOTLOADER
    HAL_Notify_Button_State((hal_button_t)button, false); 
#endif

    /* Enable Button Interrupt */
    hal_button_exti_config((hal_button_t)button, ENABLE);
}

static void BUTTON_Interrupt_Handler(void *data) {
    hal_button_t button = (hal_button_t)data;

    HAL_Buttons[button].debounce_time = 0x00;
    HAL_Buttons[button].active = true;

    hal_button_exti_config(button, DISABLE);

    button_timer_start();
}

void hal_button_timer_handler(void)
{
    if (!systick_button_timer.enable) {
        return;
    }

    if (systick_button_timer.timeout) {
        systick_button_timer.timeout--;
        if (systick_button_timer.timeout == 0) {
            systick_button_timer.timeout = BUTTON_DEBOUNCE_INTERVAL;
            button_timer_event_handler();
        }
    }
}

void hal_button_init(hal_button_t button, hal_button_mode_t Button_Mode) {

    if (!systick_button_timer.enable) {
        // Initialize button timer
        button_timer_init();
    }

    // Configure button pin
    hal_gpio_mode(HAL_Buttons[button].pin, HAL_Buttons[button].interrupt_mode == RISING ? INPUT_PULLDOWN : INPUT_PULLUP);
    if (Button_Mode == HAL_BUTTON_MODE_EXTI)  {
        /* Attach GPIOTE Interrupt */
        hal_button_exti_config(button, ENABLE);
    }

    // Check status when starting up
    if (HAL_Buttons[button].pin != PIN_INVALID && 
        hal_button_get_state(button) == (HAL_Buttons[button].interrupt_mode == RISING ? 1 : 0)) 
    {
        HAL_Buttons[button].active = true;
        button_timer_start();
    }
}

void hal_button_exti_config(hal_button_t button, FunctionalState NewState) {
    hal_interrupt_extra_configuration_t config = {0};
    config.version = HAL_INTERRUPT_EXTRA_CONFIGURATION_VERSION;
    config.keepHandler = false;
    config.flags = HAL_INTERRUPT_DIRECT_FLAG_NONE;

    if (NewState == ENABLE) {
        hal_interrupt_attach(HAL_Buttons[button].pin, 
                              BUTTON_Interrupt_Handler, 
                              (void *)((int)button), 
                              HAL_Buttons[button].interrupt_mode, 
                              &config); 
    } else {
        hal_interrupt_detach(HAL_Buttons[button].pin);
    }
}

/**
 * @brief  Returns the selected Button non-filtered state.
 * @param  Button: Specifies the Button to be checked.
 *   This parameter can be one of following parameters:
 *     @arg HAL_BUTTON1: Button1
 * @retval Actual Button Pressed state.
 */
uint8_t hal_button_get_state(hal_button_t Button) {
    return hal_gpio_read(HAL_Buttons[Button].pin);
}

/**
 * @brief  Returns the selected Button Debounced Time.
 * @param  Button: Specifies the Button to be checked.
 *   This parameter can be one of following parameters:
 *     @arg HAL_BUTTON1: Button1
 * @retval Button Debounced time in millisec.
 */
uint16_t hal_button_get_debounce_time(hal_button_t Button) {
    return HAL_Buttons[Button].debounce_time;
}

void hal_button_reset_debounced_state(hal_button_t Button) {
    HAL_Buttons[Button].debounce_time = 0;
}

static void BUTTON_Mirror_Persist(hal_button_config_t* conf) {
    hal_button_config_t saved_config;
    dct_read_app_data_copy(DCT_MODE_BUTTON_MIRROR_OFFSET, &saved_config, sizeof(saved_config));

    if (conf) {
        if (saved_config.active == 0xFF || memcmp((void*)conf, (void*)&saved_config, sizeof(hal_button_config_t))) {
            dct_write_app_data((void*)conf, DCT_MODE_BUTTON_MIRROR_OFFSET, sizeof(hal_button_config_t));
        }
    } else {
        if (saved_config.active != 0xFF) {
            memset((void*)&saved_config, 0xff, sizeof(hal_button_config_t));
            dct_write_app_data((void*)&saved_config, DCT_MODE_BUTTON_MIRROR_OFFSET, sizeof(hal_button_config_t));
        }
    }
}

void HAL_Core_Button_Mirror_Pin_Disable(uint8_t bootloader, uint8_t button, void* reserved) {
    (void)button; // unused
    int32_t state = HAL_disable_irq();
    if (HAL_Buttons[HAL_BUTTON1_MIRROR].pin != PIN_INVALID) {
        hal_interrupt_detach_ext(HAL_Buttons[HAL_BUTTON1_MIRROR].pin, 1, NULL);
        HAL_Buttons[HAL_BUTTON1_MIRROR].active = 0;
        HAL_Buttons[HAL_BUTTON1_MIRROR].pin = PIN_INVALID;
    }
    HAL_enable_irq(state);

    if (bootloader) {
        BUTTON_Mirror_Persist(NULL);
    }
}

void HAL_Core_Button_Mirror_Pin(uint16_t pin, InterruptMode mode, uint8_t bootloader, uint8_t button, void *reserved) {
    (void)button; // unused
    if (pin > TOTAL_PINS) {
        return;
    }

    if (mode != RISING && mode != FALLING) {
        return;
    }

    hal_button_config_t conf = {
        .pin = pin,
        .debounce_time = 0,
        .interrupt_mode = mode,
    };

    HAL_Buttons[HAL_BUTTON1_MIRROR] = conf;

    hal_button_init(HAL_BUTTON1_MIRROR, HAL_BUTTON_MODE_EXTI);

    if (pin == HAL_Buttons[HAL_BUTTON1].pin) {
        BUTTON_Mirror_Persist(NULL);
        return;
    }

    if (!bootloader) {
        BUTTON_Mirror_Persist(NULL);
        return;
    }

    hal_button_config_t bootloader_conf = {
        .active = 0xAA,
        .debounce_time = 0xBBCC,
        .pin = pin,
        .interrupt_mode = mode,
    };

    BUTTON_Mirror_Persist(&bootloader_conf);
}

void hal_button_init_ext() {
    hal_button_config_t button_config = {0};
    dct_read_app_data_copy(DCT_MODE_BUTTON_MIRROR_OFFSET, &button_config, sizeof(hal_button_config_t));

    if (button_config.active == 0xAA && button_config.debounce_time == 0xBBCC) {
        //int32_t state = HAL_disable_irq();
        memcpy((void*)&HAL_Buttons[HAL_BUTTON1_MIRROR], (void*)&button_config, sizeof(hal_button_config_t));
        HAL_Buttons[HAL_BUTTON1_MIRROR].active = 0;
        HAL_Buttons[HAL_BUTTON1_MIRROR].debounce_time = 0;
        hal_button_init(HAL_BUTTON1_MIRROR, HAL_BUTTON_MODE_EXTI);
        //HAL_enable_irq(state);
    }
}

void hal_button_uninit() {
    button_timer_uninit();
    hal_interrupt_uninit();
}

// Just for compatibility in bootloader
// Check both HAL_BUTTON1 and HAL_BUTTON1_MIRROR
uint8_t hal_button_is_pressed(hal_button_t button) {
    uint8_t pressed = 0;
    pressed = HAL_Buttons[button].active;

    if (button == HAL_BUTTON1 && HAL_Buttons[HAL_BUTTON1_MIRROR].pin != PIN_INVALID) {
        pressed |= HAL_Buttons[HAL_BUTTON1_MIRROR].active;
    }

    return pressed;
}

// Just for compatibility in bootloader
// Check both HAL_BUTTON1 and HAL_BUTTON1_MIRROR
uint16_t hal_button_get_pressed_time(hal_button_t button) {
    uint16_t pressed = 0;

    pressed = HAL_Buttons[button].debounce_time;
    if (button == HAL_BUTTON1 && HAL_Buttons[HAL_BUTTON1_MIRROR].pin != PIN_INVALID) {
        if (HAL_Buttons[HAL_BUTTON1_MIRROR].debounce_time > pressed) {
            pressed = HAL_Buttons[HAL_BUTTON1_MIRROR].debounce_time;
        }
    }

    return pressed;
}
