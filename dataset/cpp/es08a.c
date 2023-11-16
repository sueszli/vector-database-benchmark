/*
 * Author: Yevgeniy Kiveisha <yevgeniy.kiveisha@intel.com>
 *         Abhishek Malik <abhishek.malik@intel.com>
 * Copyright (c) 2014 Intel Corporation.
 *
 * This program and the accompanying materials are made available under the
 * terms of the The MIT License which is available at
 * https://opensource.org/licenses/MIT.
 *
 * SPDX-License-Identifier: MIT
 */
#include "es08a.h"
#include "upm_utilities.h"

es08a_context es08a_init(int32_t pin, int32_t min_pulse_width,
                         int32_t max_pulse_width) {
    // make sure MRAA is initialized
    int mraa_rv;
    if ((mraa_rv = mraa_init()) != MRAA_SUCCESS)
    {
        printf("%s: mraa_init() failed (%d).\n", __FUNCTION__, mraa_rv);
        return NULL;
    }

    es08a_context dev = (es08a_context) malloc(sizeof(struct _es08a_context));

    if(dev == NULL){
        printf("Unable to assign memory to the Servo motor structure");
        return NULL;
    }

    dev->servo_pin = pin;

    // second is the min pulse width
    dev->min_pulse_width = min_pulse_width;
    // third is the max pulse width
    dev->max_pulse_width = max_pulse_width;

    dev->pwm = mraa_pwm_init(dev->servo_pin);
    if(dev->pwm == NULL){
        printf("Unable to initialize the PWM pin");
    }

    es08a_set_angle(dev, 0);
    return dev;
}

void es08a_halt(es08a_context dev){
    mraa_pwm_enable(dev->pwm, 0);
    free(dev);
}

upm_result_t es08a_set_angle(es08a_context dev, int32_t angle){

    if(ES08A_MAX_ANGLE < angle || angle < 0){
        printf("The angle specified is either above the max angle or below 0");
        return UPM_ERROR_UNSPECIFIED;
    }
    printf("setting angle to: %d\n", angle);

    mraa_pwm_enable(dev->pwm, 1);
    mraa_pwm_period_us(dev->pwm, ES08A_PERIOD);
    int32_t val = 0;

    es08a_calc_pulse_travelling(dev, &val, angle);
    mraa_pwm_pulsewidth_us(dev->pwm, val);

    upm_delay(1);
    mraa_pwm_enable(dev->pwm, 0);

    return UPM_SUCCESS;
}

upm_result_t es08a_calc_pulse_travelling(const es08a_context dev,
                                         int32_t* ret_val, int32_t value){
    if (value > (int)dev->max_pulse_width) {
        return dev->max_pulse_width;
    }

    // if less than the boundaries
    if (value  < 0) {
        return dev->min_pulse_width;
    }

    *ret_val = (int) ((float)dev->min_pulse_width + ((float)value / ES08A_MAX_ANGLE) * ((float)dev->max_pulse_width - (float)dev->min_pulse_width));
    return UPM_SUCCESS;
}

void es08a_set_min_pulse_width (es08a_context dev, int width){
    dev->min_pulse_width = width;
}

void es08a_set_max_pulse_width (es08a_context dev, int width){
    dev->max_pulse_width = width;
}

int es08a_get_min_pulse_width (es08a_context dev){
    return dev->min_pulse_width;
}

int es08a_get_max_pulse_width (es08a_context dev){
    return dev->max_pulse_width;
}
