/*
 * An hwmon driver for delta AG9032v1 PSU
 * dps_800ab_16_d.c - Support for DPS-800AB-16 D Power Supply Module
 *
 * Copyright (C) 2017 Delta Networks, Inc.
 *
 * Aries Lin <aries.lin@deltaww.com>
 *
 * Based on ym2651y.c
 * Based on ad7414.c
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

#include <linux/kernel.h>
#include <linux/module.h>
#include <linux/init.h>
#include <linux/i2c.h>
#include <linux/slab.h>
#include <linux/mutex.h>
#include <linux/sysfs.h>
#include <linux/hwmon.h>
#include <linux/hwmon-sysfs.h>
#include <linux/err.h>

#define MAX_FAN_DUTY_CYCLE 100
#define SWPLD_REG 0x31
#define SWPLD_PSU_MUX_REG 0x21
#define SELECT_PSU1_EEPROM 0x00
#define SELECT_PSU2_EEPROM 0x20

u8 psu_member_data = 0x00;

/* Address scanned */
static const unsigned short normal_i2c[] = { 0x58, I2C_CLIENT_END };

/* This is additional data */
struct dps_800ab_16_d_data {
	struct device	*hwmon_dev;
	struct mutex	update_lock;
	char		valid;		
	unsigned long	last_updated;	/* In jiffies */

	/* Registers value */
	u8	vout_mode;
	u16	v_in;
	u16	v_out;
	u16	i_in;
	u16	i_out;
	u16	p_in;
	u16	p_out;
	u16	temp_input[2];
	u8	fan_fault;
	u16	fan_duty_cycle[2];
	u16	fan_speed[2];
	u8	mfr_model[16];
	u8	mfr_serial[16];
};

static int two_complement_to_int(u16 data, u8 valid_bit, int mask);
static ssize_t set_fan_duty_cycle(struct device *dev, struct device_attribute \
                                *dev_attr, const char *buf, size_t count);
static ssize_t for_linear_data(struct device *dev, struct device_attribute \
                                                        *dev_attr, char *buf);
static ssize_t for_fan_fault(struct device *dev, struct device_attribute \
                                                        *dev_attr, char *buf);
static ssize_t for_vout_data(struct device *dev, struct device_attribute \
                                                        *dev_attr, char *buf);
static int dps_800ab_16_d_read_byte(struct i2c_client *client, u8 reg);
static int dps_800ab_16_d_read_word(struct i2c_client *client, u8 reg);
static int dps_800ab_16_d_write_word(struct i2c_client *client, u8 reg, \
								u16 value);
static int dps_800ab_16_d_read_block(struct i2c_client *client, u8 command, \
                                                       u8 *data, int data_len);
static struct dps_800ab_16_d_data *dps_800ab_16_d_update_device( \
                                                        struct device *dev);
static ssize_t for_ascii(struct device *dev, struct device_attribute \
                                                        *dev_attr, char *buf);
static ssize_t set_w_member_data(struct device *dev, struct device_attribute \
				*dev_att, const char *buf, size_t count);
static ssize_t for_r_member_data(struct device *dev, struct device_attribute \
	 						*dev_attr, char *buf);
extern int i2c_cpld_write(int bus, unsigned short cpld_addr, u8 reg, u8 value);

enum dps_800ab_16_d_sysfs_attributes {
	PSU_V_IN,
	PSU_V_OUT,
	PSU_I_IN,
	PSU_I_OUT,
	PSU_P_IN,
	PSU_P_OUT,
	PSU_TEMP1_INPUT,
	PSU_FAN1_FAULT,
	PSU_FAN1_DUTY_CYCLE,
	PSU_FAN1_SPEED,
	PSU_MFR_MODEL,
	PSU_MFR_SERIAL,
	PSU_SELECT_MEMBER,
};

static ssize_t set_w_member_data(struct device *dev, struct device_attribute \
				*dev_attr, const char *buf, size_t count)
{
	struct sensor_device_attribute *attr = to_sensor_dev_attr(dev_attr);
	long data;
	int error;
	if (attr->index == PSU_SELECT_MEMBER) {
		error = kstrtol(buf, 16, &data);
		if (error)
			return error;
		if (SELECT_PSU1_EEPROM == data) {
			psu_member_data = SELECT_PSU1_EEPROM;
		} else if (SELECT_PSU2_EEPROM == data) {
			psu_member_data = SELECT_PSU2_EEPROM;
		} else {
			return -EINVAL;
		}
	}
	return count;
}

static ssize_t for_r_member_data(struct device *dev, struct device_attribute \
	 						*dev_attr, char *buf)
{
	return sprintf(buf, "0x%02X\n", psu_member_data);
}

static int two_complement_to_int(u16 data, u8 valid_bit, int mask)
{
	u16  valid_data  = data & mask;
    	bool is_negative = valid_data >> (valid_bit - 1);

    	return is_negative ? (-(((~valid_data) & mask) + 1)) : valid_data;
}

static ssize_t set_fan_duty_cycle(struct device *dev, struct device_attribute \
				*dev_attr, const char *buf, size_t count)
{
	struct sensor_device_attribute *attr = to_sensor_dev_attr(dev_attr);
	struct i2c_client *client = to_i2c_client(dev);
	struct dps_800ab_16_d_data *data = i2c_get_clientdata(client);
	int nr = (attr->index == PSU_FAN1_DUTY_CYCLE) ? 0 : 1;
	long speed;
	int error;
	
	error = kstrtol(buf, 10, &speed);
	if (error)
		return error;

	if (speed < 0 || speed > MAX_FAN_DUTY_CYCLE)
		return -EINVAL;

	/* Select SWPLD PSU offset */
	i2c_cpld_write(6, SWPLD_REG, 
					SWPLD_PSU_MUX_REG, psu_member_data);

	mutex_lock(&data->update_lock);
	data->fan_duty_cycle[nr] = speed;
	dps_800ab_16_d_write_word(client, 0x3B + nr, data->fan_duty_cycle[nr]);
	mutex_unlock(&data->update_lock);

	return count;
}

static ssize_t for_linear_data(struct device *dev, struct device_attribute \
							*dev_attr, char *buf)
{
	struct sensor_device_attribute *attr = to_sensor_dev_attr(dev_attr);
	struct dps_800ab_16_d_data *data = dps_800ab_16_d_update_device(dev);

	u16 value = 0;
	int exponent, mantissa;
	int multiplier = 1000;
	
	switch (attr->index) {
	case PSU_V_IN:
		value = data->v_in;
		break;
	case PSU_I_IN:
		value = data->i_in;
		break;
	case PSU_I_OUT:
		value = data->i_out;
		break;
	case PSU_P_IN:
		value = data->p_in;
		break;
	case PSU_P_OUT:
		value = data->p_out;
		break;
	case PSU_TEMP1_INPUT:
		value = data->temp_input[0];
		break;
	case PSU_FAN1_DUTY_CYCLE:
		multiplier = 1;
		value = data->fan_duty_cycle[0];
		break;
	case PSU_FAN1_SPEED:
		multiplier = 1;
		value = data->fan_speed[0];
		break;
	default:
		break;
	}

	exponent = two_complement_to_int(value >> 11, 5, 0x1f);
	mantissa = two_complement_to_int(value & 0x7ff, 11, 0x7ff);

	return (exponent >= 0) ? sprintf(buf, "%d\n",	\
		(mantissa << exponent) * multiplier) :	\
	    sprintf(buf, "%d\n", (mantissa * multiplier) / (1 << -exponent));	
}

static ssize_t for_fan_fault(struct device *dev, struct device_attribute \
							*dev_attr, char *buf)
{
	struct sensor_device_attribute *attr = to_sensor_dev_attr(dev_attr);
    	struct dps_800ab_16_d_data *data = dps_800ab_16_d_update_device(dev);

    	u8 shift = (attr->index == PSU_FAN1_FAULT) ? 7 : 6;

    	return sprintf(buf, "%d\n", data->fan_fault >> shift);
}

static ssize_t for_vout_data(struct device *dev, struct device_attribute \
		 					*dev_attr, char *buf)
{
	struct dps_800ab_16_d_data *data = dps_800ab_16_d_update_device(dev);
	int exponent, mantissa;
	int multiplier = 1000;
		
	exponent = two_complement_to_int(data->vout_mode, 5, 0x1f);
	mantissa = data->v_out;
    return (exponent > 0) ? sprintf(buf, "%d\n", \
        mantissa * (1 << exponent)) : \
        sprintf(buf, "%d\n", mantissa / (1 << -exponent) * multiplier); 

}

static ssize_t for_ascii(struct device *dev, struct device_attribute \
	 						*dev_attr, char *buf)
{
	struct sensor_device_attribute *attr = to_sensor_dev_attr(dev_attr);
	struct dps_800ab_16_d_data *data = dps_800ab_16_d_update_device(dev);
	u8 *ptr = NULL;

	if (!data->valid)
		return 0;

	switch (attr->index) {
	case PSU_MFR_MODEL:
		ptr = data->mfr_model + 1;
		break;
	case PSU_MFR_SERIAL:
		ptr = data->mfr_serial + 1;
		break;
	default:
		return 0;
	}
	return sprintf(buf, "%s\n", ptr);
}
static int dps_800ab_16_d_read_byte(struct i2c_client *client, u8 reg)
{
	return i2c_smbus_read_byte_data(client, reg);
}

static int dps_800ab_16_d_read_word(struct i2c_client *client, u8 reg)
{
	return i2c_smbus_read_word_data(client, reg);
}

static int dps_800ab_16_d_write_word(struct i2c_client *client, u8 reg, \
								u16 value)
{
	union i2c_smbus_data data;
        data.word = value;
        return i2c_smbus_xfer(client->adapter, client->addr, 
				client->flags |= I2C_CLIENT_PEC,
                              	I2C_SMBUS_WRITE, reg,
                              	I2C_SMBUS_WORD_DATA, &data);
}

static int dps_800ab_16_d_read_block(struct i2c_client *client, u8 command, \
							u8 *data, int data_len)
{
	int result = i2c_smbus_read_i2c_block_data(client, command, data_len,
									data);
	if (unlikely(result < 0))
		goto abort;
	if (unlikely(result != data_len)) {
		result = -EIO;
		goto abort;
	}

	result = 0;
abort:
	return result;

}

struct reg_data_byte {
	u8 reg;
	u8 *value;
};

struct reg_data_word {
	u8 reg;
	u16 *value;
};

static struct dps_800ab_16_d_data *dps_800ab_16_d_update_device( \
							struct device *dev)
{
	struct i2c_client *client = to_i2c_client(dev);
	struct dps_800ab_16_d_data *data = i2c_get_clientdata(client);
	
	mutex_lock(&data->update_lock);

	/* Select SWPLD PSU offset */
	i2c_cpld_write(6, SWPLD_REG, SWPLD_PSU_MUX_REG, psu_member_data);

	if (time_after(jiffies, data->last_updated)) {
		int i, status;
		u8 command;
		struct reg_data_byte regs_byte[] = {
				{0x20, &data->vout_mode},
				{0x81, &data->fan_fault}
		};
		struct reg_data_word regs_word[] = {
				{0x88, &data->v_in},
				{0x8b, &data->v_out},
				{0x89, &data->i_in},
				{0x8c, &data->i_out},
				{0x97, &data->p_in},
				{0x96, &data->p_out},
				{0x8d, &(data->temp_input[0])},
				{0x8e, &(data->temp_input[1])},
				{0x3b, &(data->fan_duty_cycle[0])},
				{0x90, &(data->fan_speed[0])},
		};

		dev_dbg(&client->dev, "start data update\n");

		/* one milliseconds from now */
		data->last_updated = jiffies + HZ / 1000;
		
		data->v_in  = 0;
		data->v_out = 0;
		data->i_in  = 0;
		data->i_out = 0;
		data->p_in  = 0;
		data->p_out = 0;
		data->temp_input[0] = 0;
		data->temp_input[1] = 0;
		data->fan_duty_cycle[0] = 0;
		data->fan_speed[0]  = 0;
		data->mfr_model[0]  = '\0';
		data->mfr_serial[0] = '\0';

		command = 0x9a;		/* PSU mfr_model */
		status = dps_800ab_16_d_read_block(client, command, 
		data->mfr_model, ARRAY_SIZE(data->mfr_model) - 1);
    	data->mfr_model[ARRAY_SIZE(data->mfr_model) - 1] = '\0';
    	if (status < 0) {
            	dev_dbg(&client->dev, "reg %d, err %d\n", command, 
							status);
    	}

    	command = 0x9e;		/* PSU mfr_serial */
    	status = dps_800ab_16_d_read_block(client, command, 
		data->mfr_serial, ARRAY_SIZE(data->mfr_serial) - 1);
    	data->mfr_serial[ARRAY_SIZE(data->mfr_serial) - 1] = '\0';
    	if (status < 0) {
            	dev_dbg(&client->dev, "reg %d, err %d\n", command, 
							status);
		}

		for (i = 0; i < ARRAY_SIZE(regs_byte); i++) {
			status = dps_800ab_16_d_read_byte(client, 
							regs_byte[i].reg);
			if (status < 0) {
				dev_dbg(&client->dev, "reg %d, err %d\n",
					regs_byte[i].reg, status);
			} else {
				*(regs_byte[i].value) = status;
			}
		}

		for (i = 0; i < ARRAY_SIZE(regs_word); i++) {
			status = dps_800ab_16_d_read_word(client,
							regs_word[i].reg);
			if (status < 0) {
				dev_dbg(&client->dev, "reg %d, err %d\n",
					regs_word[i].reg, status);
			} else {
				*(regs_word[i].value) = status;
			}
		}   			
   		

		data->valid = 1;
	}
	
	mutex_unlock(&data->update_lock);
	
	return data;

}

/* sysfs attributes for hwmon */
static SENSOR_DEVICE_ATTR(psu_v_in, S_IRUGO, for_linear_data, NULL, PSU_V_IN);
static SENSOR_DEVICE_ATTR(psu_v_out, S_IRUGO, for_vout_data,   NULL, PSU_V_OUT);
static SENSOR_DEVICE_ATTR(psu_i_in, S_IRUGO, for_linear_data, NULL, PSU_I_IN);
static SENSOR_DEVICE_ATTR(psu_i_out, S_IRUGO, for_linear_data, NULL, PSU_I_OUT);
static SENSOR_DEVICE_ATTR(psu_p_in, S_IRUGO, for_linear_data, NULL, PSU_P_IN);
static SENSOR_DEVICE_ATTR(psu_p_out, S_IRUGO, for_linear_data, NULL, PSU_P_OUT);
static SENSOR_DEVICE_ATTR(psu_temp1_input,	\
		S_IRUGO, for_linear_data,	NULL, PSU_TEMP1_INPUT);
static SENSOR_DEVICE_ATTR(psu_fan1_fault,	\
		S_IRUGO, for_fan_fault, 	NULL, PSU_FAN1_FAULT);
static SENSOR_DEVICE_ATTR(psu_fan1_duty_cycle_percentage, S_IWUGO | S_IRUGO, \
		for_linear_data, set_fan_duty_cycle, PSU_FAN1_DUTY_CYCLE);
static SENSOR_DEVICE_ATTR(psu_fan1_speed_rpm, 	\
		S_IRUGO, for_linear_data,	NULL, PSU_FAN1_SPEED);
static SENSOR_DEVICE_ATTR(psu_mfr_model, 	\
		S_IRUGO, for_ascii, NULL, PSU_MFR_MODEL);
static SENSOR_DEVICE_ATTR(psu_mfr_serial,	\
		S_IRUGO, for_ascii, NULL, PSU_MFR_SERIAL);
static SENSOR_DEVICE_ATTR(psu_select_member, S_IWUGO | S_IRUGO, \
		for_r_member_data, set_w_member_data, PSU_SELECT_MEMBER);

static struct attribute *dps_800ab_16_d_attributes[] = {
	&sensor_dev_attr_psu_v_in.dev_attr.attr,
	&sensor_dev_attr_psu_v_out.dev_attr.attr,
	&sensor_dev_attr_psu_i_in.dev_attr.attr,
	&sensor_dev_attr_psu_i_out.dev_attr.attr,
	&sensor_dev_attr_psu_p_in.dev_attr.attr,
	&sensor_dev_attr_psu_p_out.dev_attr.attr,
	&sensor_dev_attr_psu_temp1_input.dev_attr.attr,
	&sensor_dev_attr_psu_fan1_fault.dev_attr.attr,
	&sensor_dev_attr_psu_fan1_duty_cycle_percentage.dev_attr.attr,
	&sensor_dev_attr_psu_fan1_speed_rpm.dev_attr.attr,
	&sensor_dev_attr_psu_mfr_model.dev_attr.attr,
	&sensor_dev_attr_psu_mfr_serial.dev_attr.attr,
	&sensor_dev_attr_psu_select_member.dev_attr.attr,
	NULL
};

static const struct attribute_group dps_800ab_16_d_group = {
	.attrs = dps_800ab_16_d_attributes,
};

static int dps_800ab_16_d_probe(struct i2c_client *client,
				const struct i2c_device_id *id)
{
	struct dps_800ab_16_d_data *data;
	int status;

	if (!i2c_check_functionality(client->adapter, 
		I2C_FUNC_SMBUS_BYTE_DATA | I2C_FUNC_SMBUS_WORD_DATA)) {
		status = -EIO;
		goto exit;
	}
	
	data = kzalloc(sizeof(*data), GFP_KERNEL);
	if (!data) {
		status = -ENOMEM;
		goto exit;
	}

	i2c_set_clientdata(client, data);
	data->valid = 0;
	mutex_init(&data->update_lock);
	
	dev_info(&client->dev, "new chip found\n");

	/* Register sysfs hooks */
	status = sysfs_create_group(&client->dev.kobj, &dps_800ab_16_d_group);
	if (status) 
		goto exit_sysfs_create_group;

	data->hwmon_dev = hwmon_device_register(&client->dev);
	if (IS_ERR(data->hwmon_dev)) {
		status = PTR_ERR(data->hwmon_dev);
		goto exit_hwmon_device_register;
	}

	return 0;
	
exit_hwmon_device_register:
	sysfs_remove_group(&client->dev.kobj, &dps_800ab_16_d_group);
exit_sysfs_create_group:
	kfree(data);
exit:
	return status;
}

static int dps_800ab_16_d_remove(struct i2c_client *client)
{
	struct dps_800ab_16_d_data *data = i2c_get_clientdata(client);
	hwmon_device_unregister(data->hwmon_dev);
	sysfs_remove_group(&client->dev.kobj, &dps_800ab_16_d_group);
	kfree(data);
	
	return 0;
}

enum id_name {
	dni_ag9032v1_psu,
	dps_800ab_16_d
};

static const struct i2c_device_id dps_800ab_16_d_id[] = {
	{ "dni_ag9032v1_psu", dni_ag9032v1_psu },
	{ "dps_800ab_16_d", dps_800ab_16_d },
	{}
};
MODULE_DEVICE_TABLE(i2c, dps_800ab_16_d_id);

/* This is the driver that will be inserted */
static struct i2c_driver dps_800ab_16_d_driver = {
        .class          = I2C_CLASS_HWMON,
        .driver = {
                .name   = "dps_800ab_16_d",
        },
        .probe          = dps_800ab_16_d_probe,
        .remove         = dps_800ab_16_d_remove,
        .id_table       = dps_800ab_16_d_id,
        .address_list   = normal_i2c,
};

static int __init dps_800ab_16_d_init(void)
{
	return i2c_add_driver(&dps_800ab_16_d_driver);
}

static void __exit dps_800ab_16_d_exit(void)
{
	i2c_del_driver(&dps_800ab_16_d_driver);
}


MODULE_AUTHOR("Aries Lin <aries.lin@deltaww.com>");
MODULE_DESCRIPTION("DPS_800AB_16_D Driver");
MODULE_LICENSE("GPL");

module_init(dps_800ab_16_d_init);
module_exit(dps_800ab_16_d_exit);
