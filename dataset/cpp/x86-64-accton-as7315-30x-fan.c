/*
 * A hwmon driver for the Accton as7315_30x fan
 *
 * Copyright (C) 2016 Accton Technology Corporation.
 * Brandon Chuang <brandon_chuang@accton.com.tw>
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
#include <linux/dmi.h>
#include <linux/platform_device.h>

#define DRVNAME "as7315_30x_fan"
#define MAX_FAN_SPEED_RPM	22000

static struct as7315_30x_fan_data *as7315_30x_fan_update_device(
			struct device *dev);
static ssize_t fan_show_value(struct device *dev, struct device_attribute *da,
			char *buf);
static ssize_t set_duty_cycle(struct device *dev, struct device_attribute *da,
			const char *buf, size_t count);
static ssize_t show_version(struct device *dev, struct device_attribute *da,
			char *buf);
extern int as7315_30x_cpld_read(unsigned short cpld_addr, u8 reg);
extern int as7315_30x_cpld_write(unsigned short cpld_addr, u8 reg, u8 value);

/* fan related data, the index should match sysfs_fan_attributes
 */
static const u8 fan_reg[] = {
	0x22, /* fan 1 present/fault status */
	0x32, /* fan 2 present/fault status */
	0x42, /* fan 3 present/fault status */
	0x52, /* fan 4 present/fault status */
	0x62, /* fan 5 present/fault status */
	0x20, /* fan 1 speed(rpm) */
	0x30, /* fan 2 speed(rpm) */
	0x40, /* fan 3 speed(rpm) */
	0x50, /* fan 4 speed(rpm) */
	0x60, /* fan 5 speed(rpm) */
	0x21, /* fan 1 pwm */
	0x31, /* fan 2 pwm */
	0x41, /* fan 3 pwm */
	0x51, /* fan 4 pwm */
	0x61, /* fan 5 pwm */
};

/* fan data */
struct as7315_30x_fan_data {
	struct platform_device *pdev;
	struct device *hwmon_dev;
	struct mutex update_lock;
	char valid;		   /* != 0 if registers are valid */
	unsigned long last_updated;	/* In jiffies */
	u8 reg_val[ARRAY_SIZE(fan_reg)]; /* Register value */
};

enum fan_id {
	FAN1_ID,
	FAN2_ID,
	FAN3_ID,
	FAN4_ID,
	FAN5_ID
};

enum sysfs_fan_attributes {
	FAN1_STATUS_REG, /* fan 1 present/fault status */
	FAN2_STATUS_REG, /* fan 2 present/fault status */
	FAN3_STATUS_REG, /* fan 3 present/fault status */
	FAN4_STATUS_REG, /* fan 4 present/fault status */
	FAN5_STATUS_REG, /* fan 5 present/fault status */
	FAN1_TACH_REG,   /* fan 1 speed(rpm) */
	FAN2_TACH_REG,   /* fan 2 speed(rpm) */
	FAN3_TACH_REG,   /* fan 3 speed(rpm) */
	FAN4_TACH_REG,   /* fan 4 speed(rpm) */
	FAN5_TACH_REG,   /* fan 5 speed(rpm) */
	FAN1_PWM_REG,	/* fan 1 pwm */
	FAN2_PWM_REG,	/* fan 2 pwm */
	FAN3_PWM_REG,	/* fan 3 pwm */
	FAN4_PWM_REG,	/* fan 4 pwm */
	FAN5_PWM_REG,	/* fan 5 pwm */
	FAN1_PRESENT,
	FAN2_PRESENT,
	FAN3_PRESENT,
	FAN4_PRESENT,
	FAN5_PRESENT,
	FAN1_FAULT,
	FAN2_FAULT,
	FAN3_FAULT,
	FAN4_FAULT,
	FAN5_FAULT,
	FAN1_PWM,
	FAN2_PWM,
	FAN3_PWM,
	FAN4_PWM,
	FAN5_PWM,
	FAN1_INPUT, /* R.P.M */
	FAN2_INPUT,
	FAN3_INPUT,
	FAN4_INPUT,
	FAN5_INPUT,
	FAN_MAX_RPM,
	CPLD_VERSION,
	CPLD_SUB_VERSION
};

/* Define attributes
 */
#define DECLARE_FAN_FAULT_SENSOR_DEV_ATTR(index) \
	static SENSOR_DEVICE_ATTR(fan##index##_fault, S_IRUGO, fan_show_value, \
			NULL, FAN##index##_FAULT)
#define DECLARE_FAN_FAULT_ATTR(index) \
			&sensor_dev_attr_fan##index##_fault.dev_attr.attr

#define DECLARE_FAN_DUTY_CYCLE_SENSOR_DEV_ATTR(index) \
	static SENSOR_DEVICE_ATTR(fan##index##_pwm, S_IWUSR | S_IRUGO, \
			fan_show_value, set_duty_cycle, FAN##index##_PWM)
#define DECLARE_FAN_DUTY_CYCLE_ATTR(index) \
			&sensor_dev_attr_fan##index##_pwm.dev_attr.attr

#define DECLARE_FAN_PRESENT_SENSOR_DEV_ATTR(index) \
	static SENSOR_DEVICE_ATTR(fan##index##_present, S_IRUGO, fan_show_value, \
			NULL, FAN##index##_PRESENT)
#define DECLARE_FAN_PRESENT_ATTR(index) \
			&sensor_dev_attr_fan##index##_present.dev_attr.attr

#define DECLARE_FAN_SPEED_RPM_SENSOR_DEV_ATTR(index) \
	static SENSOR_DEVICE_ATTR(fan##index##_input, S_IRUGO, fan_show_value, \
			NULL, FAN##index##_INPUT)
#define DECLARE_FAN_SPEED_RPM_ATTR(index) \
			&sensor_dev_attr_fan##index##_input.dev_attr.attr

static SENSOR_DEVICE_ATTR(version, S_IRUGO, show_version, NULL, CPLD_VERSION);
static SENSOR_DEVICE_ATTR(sub_version, S_IRUGO, show_version, NULL, \
			CPLD_SUB_VERSION);
static SENSOR_DEVICE_ATTR(fan_max_speed_rpm, S_IRUGO, fan_show_value, NULL, \
			FAN_MAX_RPM);
#define DECLARE_FAN_MAX_RPM_ATTR(index) \
			&sensor_dev_attr_fan_max_speed_rpm.dev_attr.attr

/* 5 fan fault attributes in this platform */
DECLARE_FAN_FAULT_SENSOR_DEV_ATTR(1);
DECLARE_FAN_FAULT_SENSOR_DEV_ATTR(2);
DECLARE_FAN_FAULT_SENSOR_DEV_ATTR(3);
DECLARE_FAN_FAULT_SENSOR_DEV_ATTR(4);
DECLARE_FAN_FAULT_SENSOR_DEV_ATTR(5);

/* 5 fan duty cycle attribute in this platform */
DECLARE_FAN_DUTY_CYCLE_SENSOR_DEV_ATTR(1);
DECLARE_FAN_DUTY_CYCLE_SENSOR_DEV_ATTR(2);
DECLARE_FAN_DUTY_CYCLE_SENSOR_DEV_ATTR(3);
DECLARE_FAN_DUTY_CYCLE_SENSOR_DEV_ATTR(4);
DECLARE_FAN_DUTY_CYCLE_SENSOR_DEV_ATTR(5);

/* 5 fan present attributes in this platform */
DECLARE_FAN_PRESENT_SENSOR_DEV_ATTR(1);
DECLARE_FAN_PRESENT_SENSOR_DEV_ATTR(2);
DECLARE_FAN_PRESENT_SENSOR_DEV_ATTR(3);
DECLARE_FAN_PRESENT_SENSOR_DEV_ATTR(4);
DECLARE_FAN_PRESENT_SENSOR_DEV_ATTR(5);

/* 5 fan speed(rpm) attributes in this platform */
DECLARE_FAN_SPEED_RPM_SENSOR_DEV_ATTR(1);
DECLARE_FAN_SPEED_RPM_SENSOR_DEV_ATTR(2);
DECLARE_FAN_SPEED_RPM_SENSOR_DEV_ATTR(3);
DECLARE_FAN_SPEED_RPM_SENSOR_DEV_ATTR(4);
DECLARE_FAN_SPEED_RPM_SENSOR_DEV_ATTR(5);

static struct attribute *as7315_30x_fan_attributes[] = {
	&sensor_dev_attr_version.dev_attr.attr,
	&sensor_dev_attr_sub_version.dev_attr.attr,
	/* fan related attributes */
	DECLARE_FAN_FAULT_ATTR(1),
	DECLARE_FAN_FAULT_ATTR(2),
	DECLARE_FAN_FAULT_ATTR(3),
	DECLARE_FAN_FAULT_ATTR(4),
	DECLARE_FAN_FAULT_ATTR(5),
	DECLARE_FAN_DUTY_CYCLE_ATTR(1),
	DECLARE_FAN_DUTY_CYCLE_ATTR(2),
	DECLARE_FAN_DUTY_CYCLE_ATTR(3),
	DECLARE_FAN_DUTY_CYCLE_ATTR(4),
	DECLARE_FAN_DUTY_CYCLE_ATTR(5),
	DECLARE_FAN_PRESENT_ATTR(1),
	DECLARE_FAN_PRESENT_ATTR(2),
	DECLARE_FAN_PRESENT_ATTR(3),
	DECLARE_FAN_PRESENT_ATTR(4),
	DECLARE_FAN_PRESENT_ATTR(5),
	DECLARE_FAN_SPEED_RPM_ATTR(1),
	DECLARE_FAN_SPEED_RPM_ATTR(2),
	DECLARE_FAN_SPEED_RPM_ATTR(3),
	DECLARE_FAN_SPEED_RPM_ATTR(4),
	DECLARE_FAN_SPEED_RPM_ATTR(5),
	DECLARE_FAN_MAX_RPM_ATTR(),
	NULL
};

#define FAN_DUTY_CYCLE_REG_MASK 0x3F
#define FAN_MAX_DUTY_CYCLE 100
#define FAN_REG_VAL_TO_SPEED_RPM_STEP 180

static int as7315_30x_fan_read_value(struct i2c_client *client, u8 reg)
{
	return i2c_smbus_read_byte_data(client, reg);
}

static int as7315_30x_fan_write_value(struct i2c_client *client, u8 reg,
										u8 value)
{
	return i2c_smbus_write_byte_data(client, reg, value);
}

/* fan utility functions
 */
static u32 reg_val_to_duty_cycle(u8 reg_val)
{
	reg_val &= FAN_DUTY_CYCLE_REG_MASK;

	/* Fix the calculation error */
	switch (reg_val) {
		case 0x0D: return 20;
		case 0x13: return 30;
		case 0x1A: return 40;
		case 0x1F: return 50;
		case 0x25: return 60;
		case 0x2C: return 70;
		case 0x32: return 80;
		case 0x39: return 90;
		default:
			break;
	}

	return (u32)reg_val * 100 / FAN_DUTY_CYCLE_REG_MASK;
}

static u8 duty_cycle_to_reg_val(u8 duty_cycle)
{
	if (duty_cycle == 0)
		return 0;
	else if (duty_cycle >= FAN_MAX_DUTY_CYCLE)
		return FAN_DUTY_CYCLE_REG_MASK;

	/* Fix the calculation error */
	switch (duty_cycle) {
		case 20: return 0x0D;
		case 30: return 0x13;
		case 40: return 0x1A;
		case 50: return 0x1F;
		case 60: return 0x25;
		case 70: return 0x2C;
		case 80: return 0x32;
		case 90: return 0x39;
		default:
			break;
	}

	return ((u32)duty_cycle * 63 / 100) + 1;
}

static u32 reg_val_to_speed_rpm(u8 reg_val)
{
	return (u32)reg_val * FAN_REG_VAL_TO_SPEED_RPM_STEP;
}

static u8 reg_val_to_is_present(u8 reg_val)
{
	return !(reg_val & BIT(0));
}

static u8 reg_val_to_is_fan_fault(u8 status_reg_val, u8 pwm_reg_val)
{
	return (status_reg_val & BIT(1)) && (pwm_reg_val & 0x3F);
}

static ssize_t set_duty_cycle(struct device *dev, struct device_attribute *da,
			const char *buf, size_t count)
{
	int status, value;
	struct i2c_client *client = to_i2c_client(dev);
	struct as7315_30x_fan_data *data = i2c_get_clientdata(client);
	struct sensor_device_attribute *attr = to_sensor_dev_attr(da);

	status = kstrtoint(buf, 10, &value);
	if (status)
		return status;

	if (value < 0 || value > FAN_MAX_DUTY_CYCLE)
		return -EINVAL;

	mutex_lock(&data->update_lock);

	/* Enable/disable the fan power
	 */
	status = as7315_30x_fan_read_value(client, 0x10);
	if (status < 0) {
		dev_dbg(&client->dev, "Unable to read the fan power\n");
		count = status;
		goto exit;
	}

	if (!value)
		status &= ~BIT(attr->index - FAN1_PWM);
	else
		status |= BIT(attr->index - FAN1_PWM);

	status = as7315_30x_fan_write_value(client, 0x10, status);
	if (status != 0) {
		dev_dbg(&client->dev, "Unable to enable the fan power\n");
		count = status;
		goto exit;
	}

	as7315_30x_fan_write_value(client,
					fan_reg[FAN1_PWM_REG + attr->index - FAN1_PWM],
					duty_cycle_to_reg_val(value));
	data->valid = 0;

exit:
	mutex_unlock(&data->update_lock);
	return count;
}

static ssize_t fan_show_value(struct device *dev, struct device_attribute *da,
			 char *buf)
{
	struct i2c_client *client = to_i2c_client(dev);
	struct as7315_30x_fan_data *data = i2c_get_clientdata(client);
	struct sensor_device_attribute *attr = to_sensor_dev_attr(da);
	ssize_t ret = 0;

	mutex_lock(&data->update_lock);

	data = as7315_30x_fan_update_device(dev);
	if (!data->valid) {
		ret = -EIO;
		goto exit;
	}

	switch (attr->index) {
		case FAN1_PRESENT ... FAN5_PRESENT:
		{
			ret = sprintf(buf, "%d\n",
					reg_val_to_is_present(data->reg_val[FAN1_STATUS_REG +
										attr->index - FAN1_PRESENT]));
			break;
		}
		case FAN1_PWM ... FAN5_PWM:
		{
			u32 duty_cycle = reg_val_to_duty_cycle(data->reg_val[FAN1_PWM_REG +
										attr->index - FAN1_PWM]);
			ret = sprintf(buf, "%u\n", duty_cycle);
			break;
		}
		case FAN1_INPUT ... FAN5_INPUT:
		{
			u32 rpm = reg_val_to_speed_rpm(data->reg_val[FAN1_TACH_REG +
										attr->index - FAN1_INPUT]);
			ret = sprintf(buf, "%u\n", rpm);
			break;
		}
		case FAN1_FAULT ... FAN5_FAULT:
		{
			u8 reg_s = data->reg_val[FAN1_STATUS_REG + attr->index-FAN1_FAULT];
			u8 reg_pwm = data->reg_val[FAN1_PWM_REG + attr->index-FAN1_FAULT];
			ret = sprintf(buf, "%d\n", reg_val_to_is_fan_fault(reg_s, reg_pwm));
			break;
		}
		case FAN_MAX_RPM:
			ret = sprintf(buf, "%d\n", MAX_FAN_SPEED_RPM);
			break;
		default:
			break;
	}

exit:
	mutex_unlock(&data->update_lock);
	return ret;
}

static const struct attribute_group as7315_30x_fan_group = {
	.attrs = as7315_30x_fan_attributes,
};

static struct as7315_30x_fan_data *as7315_30x_fan_update_device(
													struct device *dev)
{
	struct i2c_client *client = to_i2c_client(dev);
	struct as7315_30x_fan_data *data = i2c_get_clientdata(client);

	if (time_after(jiffies, data->last_updated + HZ + HZ / 2) ||
		!data->valid) {
		int i;

		dev_dbg(&client->dev, "Starting as7315_30x_fan update\n");
		data->valid = 0;

		/* Update fan data
		 */
		for (i = 0; i < ARRAY_SIZE(data->reg_val); i++) {
			int status = as7315_30x_fan_read_value(client, fan_reg[i]);

			if (status < 0) {
				data->valid = 0;
				mutex_unlock(&data->update_lock);
				dev_dbg(&client->dev, "reg %d, err %d\n", fan_reg[i], status);
				return data;
			}
			else {
				data->reg_val[i] = status;
			}
		}

		data->last_updated = jiffies;
		data->valid = 1;
	}

	return data;
}

static ssize_t show_version(struct device *dev, struct device_attribute *da,
					char *buf)
{
	u8  reg = 0;
	int val = 0;
	struct i2c_client *client = to_i2c_client(dev);
	struct sensor_device_attribute *attr = to_sensor_dev_attr(da);

	reg = (attr->index == CPLD_VERSION) ? 0x1 : 0x2; // 0x2 for CPLD_SUB_VERSION
	val = i2c_smbus_read_byte_data(client, reg);

	if (val < 0) {
		dev_dbg(&client->dev, "fan(0x%x) reg(0x%x) err %d\n",
					client->addr, reg, val);
	}

	return sprintf(buf, "%d\n", val);
}

static int as7315_30x_fan_probe(struct i2c_client *client,
			const struct i2c_device_id *dev_id)
{
	struct as7315_30x_fan_data *data;
	int status;

	if (!i2c_check_functionality(client->adapter, I2C_FUNC_SMBUS_BYTE_DATA)) {
		status = -EIO;
		goto exit;
	}

	data = kzalloc(sizeof(struct as7315_30x_fan_data), GFP_KERNEL);
	if (!data) {
		status = -ENOMEM;
		goto exit;
	}

	i2c_set_clientdata(client, data);
	data->valid = 0;
	mutex_init(&data->update_lock);

	dev_info(&client->dev, "chip found\n");

	/* Register sysfs hooks */
	status = sysfs_create_group(&client->dev.kobj, &as7315_30x_fan_group);
	if (status)
		goto exit_free;

	data->hwmon_dev = hwmon_device_register_with_info(&client->dev,
									"as7315_30x_fan", NULL, NULL, NULL);
	if (IS_ERR(data->hwmon_dev)) {
		status = PTR_ERR(data->hwmon_dev);
		goto exit_remove;
	}

	dev_info(&client->dev, "%s: fan '%s'\n",
		 dev_name(data->hwmon_dev), client->name);
	return 0;

exit_remove:
	sysfs_remove_group(&client->dev.kobj, &as7315_30x_fan_group);
exit_free:
	kfree(data);
exit:

	return status;
}

static int as7315_30x_fan_remove(struct i2c_client *client)
{
	struct as7315_30x_fan_data *data = i2c_get_clientdata(client);
	hwmon_device_unregister(data->hwmon_dev);
	sysfs_remove_group(&client->dev.kobj, &as7315_30x_fan_group);

	return 0;
}

/* Addresses to scan */
static const unsigned short normal_i2c[] = { I2C_CLIENT_END };

static const struct i2c_device_id as7315_30x_fan_id[] = {
	{ "as7315_30x_fan", 0 },
	{}
};
MODULE_DEVICE_TABLE(i2c, as7315_30x_fan_id);

static struct i2c_driver as7315_30x_fan_driver = {
	.class		= I2C_CLASS_HWMON,
	.driver = {
		.name	 = DRVNAME,
	},
	.probe		= as7315_30x_fan_probe,
	.remove	   = as7315_30x_fan_remove,
	.id_table	 = as7315_30x_fan_id,
	.address_list = normal_i2c,
};

module_i2c_driver(as7315_30x_fan_driver);

MODULE_AUTHOR("Brandon Chuang <brandon_chuang@edge-core.com>");
MODULE_DESCRIPTION("as7315_30x_fan driver");
MODULE_LICENSE("GPL");
