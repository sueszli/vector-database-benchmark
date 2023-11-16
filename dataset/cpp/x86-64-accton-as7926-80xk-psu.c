/*
 * An hwmon driver for accton as7926_80xk Power Module
 *
 * Copyright (C) 2019 Accton Technology Corporation.
 * Phani Karanam <phani_karanam@accton.com.tw>
 *
 * Based on ad7414.c
 * Copyright 2006 Stefan Roese <sr at denx.de>, DENX Software Engineering
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

#include <linux/module.h>
#include <linux/jiffies.h>
#include <linux/i2c.h>
#include <linux/hwmon.h>
#include <linux/hwmon-sysfs.h>
#include <linux/err.h>
#include <linux/mutex.h>
#include <linux/sysfs.h>
#include <linux/slab.h>
#include <linux/delay.h>
#include <linux/dmi.h>

static ssize_t show_status(struct device *dev, struct device_attribute *da, char *buf);
extern int accton_i2c_cpld_read (unsigned short cpld_addr, u8 reg);

/* Addresses scanned 
 */
static const unsigned short normal_i2c[] = { I2C_CLIENT_END };

/* Each client has this additional data 
 */
struct as7926_80xk_psu_data {
    struct device      *hwmon_dev;
    struct mutex        update_lock;
    char                valid;           /* !=0 if registers are valid */
    unsigned long       last_updated;    /* In jiffies */
    u8  index;           /* PSU index */
    u8  psu_present;     /* Status(present) register read from CPLD */
    u8  psu_power_good;  /* Status(power_good) register read from CPLD */
};

static struct as7926_80xk_psu_data *as7926_80xk_psu_update_device(struct device *dev);             

enum as7926_80xk_psu_sysfs_attributes {
    PSU_PRESENT,
    PSU_POWER_GOOD
};

/* sysfs attributes for hwmon 
 */
static SENSOR_DEVICE_ATTR(psu_present,    S_IRUGO, show_status, NULL, PSU_PRESENT);
static SENSOR_DEVICE_ATTR(psu_power_good, S_IRUGO, show_status, NULL, PSU_POWER_GOOD);

static struct attribute *as7926_80xk_psu_attributes[] = {
    &sensor_dev_attr_psu_present.dev_attr.attr,
    &sensor_dev_attr_psu_power_good.dev_attr.attr,
    NULL
};

static ssize_t show_status(struct device *dev, struct device_attribute *da,
             char *buf)
{
    struct sensor_device_attribute *attr = to_sensor_dev_attr(da);
    struct as7926_80xk_psu_data *data = as7926_80xk_psu_update_device(dev);
    u8 status = 0;

    if (!data->valid) {
        return -EIO;
    }

    if (attr->index == PSU_PRESENT) {
        status = !((data->psu_present >> data->index) & 0x1);
    }
    else { /* PSU_POWER_GOOD */
        status = ((data->psu_power_good >> data->index) & 0x1);
    }

    return sprintf(buf, "%d\n", status);
}

static const struct attribute_group as7926_80xk_psu_group = {
    .attrs = as7926_80xk_psu_attributes,
};

static int as7926_80xk_psu_probe(struct i2c_client *client,
            const struct i2c_device_id *dev_id)
{
    struct as7926_80xk_psu_data *data;
    int status;

    if (!i2c_check_functionality(client->adapter, I2C_FUNC_SMBUS_I2C_BLOCK)) {
        status = -EIO;
        goto exit;
    }

    data = kzalloc(sizeof(struct as7926_80xk_psu_data), GFP_KERNEL);
    if (!data) {
        status = -ENOMEM;
        goto exit;
    }

    i2c_set_clientdata(client, data);
    data->valid = 0;
    data->index = dev_id->driver_data;
    mutex_init(&data->update_lock);

    dev_info(&client->dev, "chip found\n");

    /* Register sysfs hooks */
    status = sysfs_create_group(&client->dev.kobj, &as7926_80xk_psu_group);
    if (status) {
        goto exit_free;
    }

    data->hwmon_dev = hwmon_device_register_with_info(&client->dev, "as7926_80xk_psu",
                                                      NULL, NULL, NULL);
    if (IS_ERR(data->hwmon_dev)) {
        status = PTR_ERR(data->hwmon_dev);
        goto exit_remove;
    }

    dev_info(&client->dev, "%s: psu '%s'\n",
         dev_name(data->hwmon_dev), client->name);
    
    return 0;

exit_remove:
    sysfs_remove_group(&client->dev.kobj, &as7926_80xk_psu_group);
exit_free:
    kfree(data);
exit:
    
    return status;
}

static int as7926_80xk_psu_remove(struct i2c_client *client)
{
    struct as7926_80xk_psu_data *data = i2c_get_clientdata(client);

    hwmon_device_unregister(data->hwmon_dev);
    sysfs_remove_group(&client->dev.kobj, &as7926_80xk_psu_group);
    kfree(data);
    
    return 0;
}

enum psu_index 
{ 
    as7926_80xk_psu1, 
    as7926_80xk_psu2,
    as7926_80xk_psu3,
    as7926_80xk_psu4
};

static const struct i2c_device_id as7926_80xk_psu_id[] = {
    { "as7926_80xk_psu1", as7926_80xk_psu1 },
    { "as7926_80xk_psu2", as7926_80xk_psu2 },
    { "as7926_80xk_psu3", as7926_80xk_psu3 },
    { "as7926_80xk_psu4", as7926_80xk_psu4 },
    {}
};
MODULE_DEVICE_TABLE(i2c, as7926_80xk_psu_id);

static struct i2c_driver as7926_80xk_psu_driver = {
    .class        = I2C_CLASS_HWMON,
    .driver = {
        .name     = "as7926_80xk_psu",
    },
    .probe        = as7926_80xk_psu_probe,
    .remove       = as7926_80xk_psu_remove,
    .id_table     = as7926_80xk_psu_id,
    .address_list = normal_i2c,
};

static struct as7926_80xk_psu_data *as7926_80xk_psu_update_device(struct device *dev)
{
    struct i2c_client *client = to_i2c_client(dev);
    struct as7926_80xk_psu_data *data = i2c_get_clientdata(client);
    
    mutex_lock(&data->update_lock);

    if (time_after(jiffies, data->last_updated + HZ + HZ / 2)
        || !data->valid) {
        int psu_present = 0;
        int power_good = 0;

        data->valid = 0;
        dev_dbg(&client->dev, "Starting as7926_80xk update\n");

        /* Read psu present */
        psu_present = accton_i2c_cpld_read(0x60, 0x51);
        
        if (psu_present < 0) {
            dev_dbg(&client->dev, "cpld reg 0x60 err %d\n", psu_present);
            goto exit;
        }
        else {
            data->psu_present = psu_present;
        }
        
        /* Read psu power good */
        power_good = accton_i2c_cpld_read(0x60, 0x52);
        
        if (power_good < 0) {
            dev_dbg(&client->dev, "cpld reg 0x60 err %d\n", power_good);
            goto exit;
        }
        else {
            data->psu_power_good = power_good;
        }
        
        data->last_updated = jiffies;
        data->valid = 1;
    }

exit:
    mutex_unlock(&data->update_lock);

    return data;
}

static int __init as7926_80xk_psu_init(void)
{
    return i2c_add_driver(&as7926_80xk_psu_driver);
}

static void __exit as7926_80xk_psu_exit(void)
{
    i2c_del_driver(&as7926_80xk_psu_driver);
}

module_init(as7926_80xk_psu_init);
module_exit(as7926_80xk_psu_exit);

MODULE_AUTHOR("Phani Karanam <phani_karanam@accton.com.tw>");
MODULE_DESCRIPTION("as7926_80xk_psu driver");
MODULE_LICENSE("GPL");

