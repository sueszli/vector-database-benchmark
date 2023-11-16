/*
 * A LED driver for the as7936_22xke_led
 *
 * Copyright (C) 2019  Edgecore Networks Corporation.
 * Jostar Yang <jostar_yang@edge-core.com>
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 */

/*#define DEBUG*/

#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/init.h>
#include <linux/platform_device.h>
#include <linux/err.h>
#include <linux/leds.h>
#include <linux/slab.h>
#include <linux/dmi.h>
#include <linux/printk.h>

extern int as7936_22xke_cpld_read(int bus_num, unsigned short cpld_addr, u8 reg);
extern int as7936_22xke_cpld_write(int bus_num, unsigned short cpld_addr, u8 reg, u8 value);

#define DRVNAME "as7936_22xke_led"


/* LED related data
 */
#define LED_CPLD_I2C_BUS_NUM         11
#define LED_CNTRLER_I2C_ADDRESS		(0x60)

#define LED_TYPE_DIAG_REG_MASK	 	(0x1|0x2|0x4)
#define LED_MODE_DIAG_RED_VALUE  	(0x2|0x4)
#define LED_MODE_DIAG_YLW_VALUE  	(0x4)
#define LED_MODE_DIAG_GREEN_VALUE  	(0x1|0x4)
#define LED_MODE_DIAG_BLUE_VALUE  	(0x1|0x2)
#define LED_MODE_DIAG_OFF_VALUE		(0x1|0x2|0x4)

#define LED_TYPE_LOC_REG_MASK	 	(0x10|0x20|0x40)
#define LED_MODE_LOC_RED_VALUE	    (0x20|0x40)
#define LED_MODE_LOC_YLW_VALUE	    (0x40)
#define LED_MODE_LOC_GREEN_VALUE	(0x10|0x40)
#define LED_MODE_LOC_BLUE_VALUE	    (0x10|0x20)
#define LED_MODE_LOC_OFF_VALUE		(0x10|0x20|0x40)

#define LED_TYPE_FAN_REG_MASK       LED_TYPE_LOC_REG_MASK
#define LED_MODE_FAN_RED_VALUE      LED_MODE_LOC_RED_VALUE
#define LED_MODE_FAN_YLW_VALUE      LED_MODE_LOC_YLW_VALUE
#define LED_MODE_FAN_GREEN_VALUE    LED_MODE_LOC_GREEN_VALUE
#define LED_MODE_FAN_BLUE_VALUE     LED_MODE_LOC_BLUE_VALUE
#define LED_MODE_FAN_OFF_VALUE		LED_MODE_LOC_OFF_VALUE

#define LED_TYPE_PSU1_REG_MASK       LED_TYPE_DIAG_REG_MASK
#define LED_MODE_PSU1_RED_VALUE      LED_MODE_DIAG_RED_VALUE
#define LED_MODE_PSU1_YLW_VALUE      LED_MODE_DIAG_YLW_VALUE
#define LED_MODE_PSU1_GREEN_VALUE    LED_MODE_DIAG_GREEN_VALUE
#define LED_MODE_PSU1_BLUE_VALUE     LED_MODE_DIAG_BLUE_VALUE
#define LED_MODE_PSU1_OFF_VALUE		 LED_MODE_DIAG_OFF_VALUE

#define LED_TYPE_PSU2_REG_MASK       LED_TYPE_LOC_REG_MASK
#define LED_MODE_PSU2_RED_VALUE      LED_MODE_LOC_RED_VALUE
#define LED_MODE_PSU2_YLW_VALUE      LED_MODE_LOC_YLW_VALUE
#define LED_MODE_PSU2_GREEN_VALUE    LED_MODE_LOC_GREEN_VALUE
#define LED_MODE_PSU2_BLUE_VALUE     LED_MODE_LOC_BLUE_VALUE
#define LED_MODE_PSU2_OFF_VALUE		 LED_MODE_LOC_OFF_VALUE

enum led_type {
    LED_TYPE_DIAG = 0,
    LED_TYPE_LOC,
    LED_TYPE_FAN,
    LED_TYPE_PSU1,
    LED_TYPE_PSU2,
    LED_TYPE_MAX
};

struct led_reg {
    u32  types;
    u8   reg_addr;
};

static const struct led_reg led_reg_map[] = {
    {(1 << LED_TYPE_LOC) | (1 << LED_TYPE_DIAG) , 0x43},
    {(1 << LED_TYPE_FAN)  , 0x44},
    {(1 << LED_TYPE_PSU2) | (1 << LED_TYPE_PSU1) , 0x45},
};

enum led_light_mode {
    LED_MODE_OFF,
    LED_MODE_RED 				= 10,
    LED_MODE_RED_BLINKING 		= 11,
    LED_MODE_ORANGE 			= 12,
    LED_MODE_ORANGE_BLINKING 	= 13,
    LED_MODE_YELLOW 			= 14,
    LED_MODE_YELLOW_BLINKING 	= 15,
    LED_MODE_GREEN 				= 16,
    LED_MODE_GREEN_BLINKING 	= 17,
    LED_MODE_BLUE 				= 18,
    LED_MODE_BLUE_BLINKING 		= 19,
    LED_MODE_PURPLE 			= 20,
    LED_MODE_PURPLE_BLINKING 	= 21,
    LED_MODE_AUTO 				= 22,
    LED_MODE_AUTO_BLINKING 		= 23,
    LED_MODE_WHITE 				= 24,
    LED_MODE_WHITE_BLINKING 	= 25,
    LED_MODE_CYAN 				= 26,
    LED_MODE_CYAN_BLINKING 		= 27,
    LED_MODE_UNKNOWN			= 99
};


struct led_type_mode {
    enum led_type type;
    enum led_light_mode mode;
    int  reg_bit_mask;
    int  mode_value;
};

static struct led_type_mode led_type_mode_data[] = {
    {LED_TYPE_DIAG, LED_MODE_OFF,  LED_TYPE_DIAG_REG_MASK, LED_MODE_DIAG_OFF_VALUE},
    {LED_TYPE_DIAG, LED_MODE_RED,   LED_TYPE_DIAG_REG_MASK, LED_MODE_DIAG_RED_VALUE},
    {LED_TYPE_DIAG, LED_MODE_YELLOW, LED_TYPE_DIAG_REG_MASK, LED_MODE_DIAG_YLW_VALUE},
    {LED_TYPE_DIAG, LED_MODE_GREEN, LED_TYPE_DIAG_REG_MASK, LED_MODE_DIAG_GREEN_VALUE},
    {LED_TYPE_DIAG, LED_MODE_BLUE,  LED_TYPE_DIAG_REG_MASK, LED_MODE_DIAG_BLUE_VALUE},

    {LED_TYPE_LOC,  LED_MODE_OFF,   LED_TYPE_LOC_REG_MASK,  LED_MODE_LOC_OFF_VALUE},
    {LED_TYPE_LOC,  LED_MODE_RED,   LED_TYPE_LOC_REG_MASK,  LED_MODE_LOC_RED_VALUE},
    {LED_TYPE_LOC,  LED_MODE_YELLOW, LED_TYPE_LOC_REG_MASK,  LED_MODE_LOC_YLW_VALUE},
    {LED_TYPE_LOC,  LED_MODE_GREEN, LED_TYPE_LOC_REG_MASK,  LED_MODE_LOC_GREEN_VALUE},
    {LED_TYPE_LOC,  LED_MODE_BLUE,  LED_TYPE_LOC_REG_MASK,  LED_MODE_LOC_BLUE_VALUE},

    {LED_TYPE_FAN,  LED_MODE_OFF,   LED_TYPE_FAN_REG_MASK,  LED_MODE_FAN_OFF_VALUE},
    {LED_TYPE_FAN,  LED_MODE_RED,   LED_TYPE_FAN_REG_MASK,  LED_MODE_FAN_RED_VALUE},
    {LED_TYPE_FAN,  LED_MODE_YELLOW, LED_TYPE_FAN_REG_MASK,  LED_MODE_FAN_YLW_VALUE},
    {LED_TYPE_FAN,  LED_MODE_GREEN, LED_TYPE_FAN_REG_MASK,  LED_MODE_FAN_GREEN_VALUE},
    {LED_TYPE_FAN,  LED_MODE_BLUE,  LED_TYPE_FAN_REG_MASK,  LED_MODE_FAN_BLUE_VALUE},

    {LED_TYPE_PSU1,  LED_MODE_OFF,   LED_TYPE_PSU1_REG_MASK,  LED_MODE_PSU1_OFF_VALUE},
    {LED_TYPE_PSU1,  LED_MODE_RED,   LED_TYPE_PSU1_REG_MASK,  LED_MODE_PSU1_RED_VALUE},
    {LED_TYPE_PSU1,  LED_MODE_YELLOW, LED_TYPE_PSU1_REG_MASK,  LED_MODE_PSU1_YLW_VALUE},
    {LED_TYPE_PSU1,  LED_MODE_GREEN, LED_TYPE_PSU1_REG_MASK,  LED_MODE_PSU1_GREEN_VALUE},
    {LED_TYPE_PSU1,  LED_MODE_BLUE,  LED_TYPE_PSU1_REG_MASK,  LED_MODE_PSU1_BLUE_VALUE},

    {LED_TYPE_PSU2,  LED_MODE_OFF,   LED_TYPE_PSU2_REG_MASK,  LED_MODE_PSU2_OFF_VALUE},
    {LED_TYPE_PSU2,  LED_MODE_RED,   LED_TYPE_PSU2_REG_MASK,  LED_MODE_PSU2_RED_VALUE},
    {LED_TYPE_PSU2,  LED_MODE_YELLOW, LED_TYPE_PSU2_REG_MASK, LED_MODE_PSU2_YLW_VALUE},
    {LED_TYPE_PSU2,  LED_MODE_GREEN, LED_TYPE_PSU2_REG_MASK,  LED_MODE_PSU2_GREEN_VALUE},
    {LED_TYPE_PSU2,  LED_MODE_BLUE,  LED_TYPE_PSU2_REG_MASK,  LED_MODE_PSU2_BLUE_VALUE},

};

struct LedData {
    struct platform_device *pdev;
    struct mutex	 update_lock;
    char			 valid;		        /* != 0 if registers are valid */
    unsigned long	last_updated;	    /* In jiffies */
    u8			   reg_val[ARRAY_SIZE(led_reg_map)];
};

static struct LedData  *ledctl = NULL;

static int get_regIdx(enum led_type type, u8 *idx)
{
    int i;

    for (i = 0; i < ARRAY_SIZE(led_reg_map); i++) {
        if(led_reg_map[i].types & (1 << type)) {
            *idx = i;
            return 0;
        }
    }
    return -EINVAL;
}

static int get_reg_addr(enum led_type type, u8 *reg)
{
    int i;

    for (i = 0; i < ARRAY_SIZE(led_reg_map); i++) {
        if(led_reg_map[i].types & (1 << type)) {
            *reg = led_reg_map[i].reg_addr;
            return 0;
        }
    }
    return -EINVAL;
}

static int led_reg_val_to_light_mode(enum led_type type, u8 reg_val)
{
    int i;

    for (i = 0; i < ARRAY_SIZE(led_type_mode_data); i++) {

        if (type != led_type_mode_data[i].type)
            continue;

        if ((led_type_mode_data[i].reg_bit_mask & reg_val) ==
                led_type_mode_data[i].mode_value)
        {
            return led_type_mode_data[i].mode;
        }
    }
    return 0;
}

static u8 led_light_mode_to_reg_val(enum led_type type,
                                    enum led_light_mode mode, u8 reg_val) {
    int i;

    for (i = 0; i < ARRAY_SIZE(led_type_mode_data); i++) {
        if (type != led_type_mode_data[i].type)
            continue;

        if (mode != led_type_mode_data[i].mode)
            continue;

        reg_val = led_type_mode_data[i].mode_value |
                  (reg_val & (~led_type_mode_data[i].reg_bit_mask));
        break;
    }
    return reg_val;
}

static int led_read_value(u8 reg)
{
    return as7936_22xke_cpld_read(LED_CPLD_I2C_BUS_NUM, LED_CNTRLER_I2C_ADDRESS, reg);
}

static int led_write_value(u8 reg, u8 value)
{
    return as7936_22xke_cpld_write(LED_CPLD_I2C_BUS_NUM, LED_CNTRLER_I2C_ADDRESS, reg, value);
}

static void data_update(void)
{
    mutex_lock(&ledctl->update_lock);

    if (time_after(jiffies, ledctl->last_updated + HZ + HZ / 2)
            || !ledctl->valid) {
        int i;

        dev_dbg(&ledctl->pdev->dev, "Starting led update\n");

        /* Update LED data
         */
        for (i = 0; i < ARRAY_SIZE(ledctl->reg_val); i++) {
            int status = led_read_value(led_reg_map[i].reg_addr);
            if (status < 0) {
                ledctl->valid = 0;
                dev_dbg(&ledctl->pdev->dev, "reg %d, err %d\n", led_reg_map[i].reg_addr, status);
                goto exit;
            }
            else
            {
                ledctl->reg_val[i] = status;
            }
        }

        ledctl->last_updated = jiffies;
        ledctl->valid = 1;
    }

exit:
    mutex_unlock(&ledctl->update_lock);
}

static void _led_set(struct led_classdev *led_cdev,
                     enum led_brightness led_light_mode,
                     enum led_type type)
{
    int reg_val;
    u8  addr=0;

    if (get_reg_addr(type, &addr)) {
        dev_dbg(&ledctl->pdev->dev, "Not match register for %d.\n", type);
        return;
    }

    mutex_lock(&ledctl->update_lock);
    reg_val = led_read_value(addr);
    if (reg_val < 0) {
        dev_dbg(&ledctl->pdev->dev, "reg %d, err %d\n", addr, reg_val);
        goto exit;
    }
    reg_val = led_light_mode_to_reg_val(type, led_light_mode, reg_val);
    led_write_value(addr, reg_val);

    /* to prevent the slow-update issue */
    ledctl->valid = 0;
exit:
    mutex_unlock(&ledctl->update_lock);
}

#define DEVNAME_PREDIX "as7936_22xke_led::"
#define DEVNAME_STR(_s)  DEVNAME_PREDIX#_s
const char * DevNames[LED_TYPE_MAX] = {DEVNAME_STR(diag),
                                       DEVNAME_STR(loc), DEVNAME_STR(fan),
                                       DEVNAME_STR(psu1),  DEVNAME_STR(psu2)
                                      };

static int getLedType(struct led_classdev *cdev, enum led_type *t)
{
    int i;
    for (i = 0; i < LED_TYPE_MAX; i++) {
        char fullName[32+1] = {0};
        snprintf(fullName, sizeof(fullName), DevNames[i]);
        if (!strcmp(cdev->name, fullName)) {
            break;
        }
    }
    if (i >= LED_TYPE_MAX) {
        dev_dbg(cdev->dev, "[ERROR]%s:Failed to find for %s\n", __func__, cdev->name);
        return -EINVAL;
    }
    *t = i;
    return 0;
}

static enum led_brightness led_get(struct led_classdev *cdev)
{
    int ret;
    enum led_type t;
    u8  regIdx;

    ret = getLedType(cdev, &t);
    if (ret < 0) {
        return ret;
    }
    ret = get_regIdx(t, &regIdx);
    if (ret < 0) {
        return ret;
    }
    data_update();
    return led_reg_val_to_light_mode(t, ledctl->reg_val[regIdx]);
}

static void led_set(struct led_classdev *cdev,
                    enum led_brightness light_mode)
{
    int ret;
    enum led_type t;

    ret = getLedType(cdev, &t);
    if (ret < 0) {
        return ;
    }
    _led_set(cdev, light_mode, t);
}

static struct led_classdev as7936_22xke_leds[] = {
    [LED_TYPE_DIAG] = {
        .name			 = DEVNAME_STR(diag),
        .default_trigger = "unused",
        .brightness_set	 = led_set,
        .brightness_get	 = led_get,
        .flags			 = LED_CORE_SUSPENDRESUME,
        .max_brightness	 = LED_MODE_BLUE,
    },
    [LED_TYPE_LOC] = {
        .name			 = DEVNAME_STR(loc),
        .default_trigger = "unused",
        .brightness_set	 = led_set,
        .brightness_get	 = led_get,
        .flags			 = LED_CORE_SUSPENDRESUME,
        .max_brightness	 = LED_MODE_BLUE,
    },
    [LED_TYPE_FAN] = {
        .name			 = DEVNAME_STR(fan),
        .default_trigger = "unused",
        .brightness_set	 = led_set,
        .brightness_get  = led_get,
        .flags			 = LED_CORE_SUSPENDRESUME,
        .max_brightness  = LED_MODE_BLUE,
    },
    [LED_TYPE_PSU1] = {
        .name			 = DEVNAME_STR(psu1),
        .default_trigger = "unused",
        .brightness_set	 = led_set,
        .brightness_get  = led_get,
        .flags			 = LED_CORE_SUSPENDRESUME,
        .max_brightness  = LED_MODE_BLUE,
    },
    [LED_TYPE_PSU2] = {
        .name			 = DEVNAME_STR(psu2),
        .default_trigger = "unused",
        .brightness_set	 = led_set,
        .brightness_get  = led_get,
        .flags			 = LED_CORE_SUSPENDRESUME,
        .max_brightness  = LED_MODE_BLUE,
    },
};

static int as7936_22xke_led_suspend(struct platform_device *dev,
                                    pm_message_t state)
{
    int i = 0;

    for (i = 0; i < ARRAY_SIZE(as7936_22xke_leds); i++) {
        led_classdev_suspend(&as7936_22xke_leds[i]);
    }
    return 0;
}

static int as7936_22xke_led_resume(struct platform_device *dev)
{
    int i;

    for (i = 0; i < ARRAY_SIZE(as7936_22xke_leds); i++) {
        led_classdev_resume(&as7936_22xke_leds[i]);
    }
    return 0;
}

static int as7936_22xke_led_probe(struct platform_device *pdev)
{
    int ret, i;

    for (i = 0; i < ARRAY_SIZE(as7936_22xke_leds); i++) {
        ret = led_classdev_register(&pdev->dev, &as7936_22xke_leds[i]);

        if (ret < 0)
            break;
    }

    /* Check if all LEDs were successfully registered */
    if (i != ARRAY_SIZE(as7936_22xke_leds)) {
        int j;

        /* only unregister the LEDs that were successfully registered */
        for (j = 0; j < i; j++) {
            led_classdev_unregister(&as7936_22xke_leds[i]);
        }
    }
    return ret;
}

static int as7936_22xke_led_remove(struct platform_device *pdev)
{
    int i;
    for (i = 0; i < ARRAY_SIZE(as7936_22xke_leds); i++) {
        led_classdev_unregister(&as7936_22xke_leds[i]);
    }
    return 0;
}

static struct platform_driver as7936_22xke_led_driver = {
    .probe	 = as7936_22xke_led_probe,
    .remove	 = as7936_22xke_led_remove,
    .suspend = as7936_22xke_led_suspend,
    .resume	 = as7936_22xke_led_resume,
    .driver	 = {
        .name   = DRVNAME,
        .owner  = THIS_MODULE,
    },
};

static int __init as7936_22xke_led_init(void)
{
    int ret;

    ret = platform_driver_register(&as7936_22xke_led_driver);
    if (ret < 0) {
        goto exit;
    }

    ledctl = kzalloc(sizeof(struct LedData), GFP_KERNEL);
    if (!ledctl) {
        ret = -ENOMEM;
        platform_driver_unregister(&as7936_22xke_led_driver);
        goto exit;
    }

    mutex_init(&ledctl->update_lock);

    ledctl->pdev = platform_device_register_simple(DRVNAME, -1, NULL, 0);
    if (IS_ERR(ledctl->pdev)) {
        ret = PTR_ERR(ledctl->pdev);
        platform_driver_unregister(&as7936_22xke_led_driver);
        kfree(ledctl);
        goto exit;
    }

exit:
    return ret;
}

static void __exit as7936_22xke_led_exit(void)
{
    platform_device_unregister(ledctl->pdev);
    platform_driver_unregister(&as7936_22xke_led_driver);
    kfree(ledctl);
}

module_init(as7936_22xke_led_init);
module_exit(as7936_22xke_led_exit);

MODULE_AUTHOR("Jostar Yang <jostar_yang@edge-core.com>");
MODULE_DESCRIPTION("as7936_22xke_led driver");
MODULE_LICENSE("GPL");

