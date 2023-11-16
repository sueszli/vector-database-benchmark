/*
 * A CPLD driver for monitor SFP28 module I/O
 *
 * The CPLD is customize by Quanta for controlling SFP28 module signals,
 * they are TX_FAULT , TX_DIS , PRE_N , RX_LOS
 * Each CPLD control 16 modules, each module use 4 bits in register.
 *
 * Copyright (C) 2015 Quanta Inc.
 *
 * Author: Luffy Cheng <luffy.cheng@quantatw.com>
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
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 */

#define pr_fmt(fmt) KBUILD_MODNAME ": " fmt

#include <linux/module.h>
#include <linux/jiffies.h>
#include <linux/i2c.h>
#include <linux/err.h>
#include <linux/mutex.h>
#include <linux/delay.h>
#include <linux/log2.h>
#include <linux/slab.h>
#include <linux/kdev_t.h>
#include <linux/idr.h>
#include <linux/ctype.h>
#include <linux/string.h>

static DEFINE_IDA(cpld_ida);

enum platform_type {
	SFP = 0,
	SFP28,
	NONE
};

static struct class *cpld_class = NULL;

struct sfp_data {
	struct i2c_client *cpld_client;
	char name[8];
	char type[8];
	u8 port_id;
	u8 cpld_port;
};

struct cpld_data {
	struct mutex lock;
	struct device *port_dev[16];
	struct sfp_data *port_data[16];
};

static int cpld_probe(struct i2c_client *client,
			 const struct i2c_device_id *id);
static int cpld_remove(struct i2c_client *client);

static const struct i2c_device_id cpld_id[] = {
	{ "CPLD-SFP", SFP },
	{ "CPLD-SFP28", SFP28 },
	{ }
};
MODULE_DEVICE_TABLE(i2c, cpld_id);

static struct i2c_driver cpld_driver = {
	.class		= I2C_CLASS_HWMON,
	.driver = {
		.name	= "qci_cpld_sfp28",
	},
	.probe		= cpld_probe,
	.remove		= cpld_remove,
	.id_table	= cpld_id,
//	.address_list	= normal_i2c,
};

#define CPLD_ID_PREFIX "port-"
#define CPLD_ID_FORMAT CPLD_ID_PREFIX "%d"

//SFP28
#define TX_FAULT_MASK 			0x08
#define TX_DIS_MASK	 			0x04
#define PRE_N_MASK 				0x02
#define RX_LOS_MASK 			0x01

static inline u8 get_group_cmd(u8 group)
{
	//FIXME: if group cmd change
	return (group + 1);
}

static inline u8 port_remapping(u8 phy_port)
{
	/* FIXME: implement by hardware design */
	/* The CPLD register port mapping is weird :
	 * MSB -------- LSB		(word data)
	 * P3	P4	P1	P2		(per port 4 bits)
	 * For easy coding bit shift, we treat it as hw port swap
	 */
	return (phy_port % 2) ? (phy_port - 1) : (phy_port + 1);
}

//SFP
static ssize_t get_tx_fault(struct device *dev,
			     struct device_attribute *devattr,
			     char *buf)
{
	struct sfp_data *data = dev_get_drvdata(dev);
	struct i2c_client *client = data->cpld_client;
	u8 group = (u8)(data->cpld_port / 4);
	u8 group_port = data->cpld_port % 4;
	s32 value;

	dev_dbg(&client->dev, "port_id %d => cpld_port %d, group %d(%d)\n", data->port_id,
				data->cpld_port + 1, group + 1, group_port + 1);

	value = i2c_smbus_read_word_data(client, get_group_cmd(group));
	if (value < 0)
		return -ENODEV;

	dev_dbg(&client->dev, "read group%d value= %x\n", group + 1, value);

	value >>= (group_port * 4);
	value &= TX_FAULT_MASK;

	return sprintf(buf, "%d\n", value ? 1 : 0);
}

static ssize_t get_tx_dis(struct device *dev,
			     struct device_attribute *devattr,
			     char *buf)
{
	struct sfp_data *data = dev_get_drvdata(dev);
	struct i2c_client *client = data->cpld_client;
	u8 group = (u8)(data->cpld_port / 4);
	u8 group_port = data->cpld_port % 4;
	s32 value;

	dev_dbg(&client->dev, "port_id %d => cpld_port %d, group %d(%d)\n", data->port_id,
				data->cpld_port + 1, group + 1, group_port + 1);

	value = i2c_smbus_read_word_data(client, get_group_cmd(group));
	if (value < 0)
		return -ENODEV;

	dev_dbg(&client->dev, "read group%d value= %x\n", group + 1, value);

	value >>= (group_port * 4);
	value &= TX_DIS_MASK;

	return sprintf(buf, "%d\n", value ? 1 : 0);
}

static ssize_t get_pre_n(struct device *dev,
			     struct device_attribute *devattr,
			     char *buf)
{
	struct sfp_data *data = dev_get_drvdata(dev);
	struct i2c_client *client = data->cpld_client;
	u8 group = (u8)(data->cpld_port / 4);
	u8 group_port = data->cpld_port % 4;
	s32 value;

	dev_dbg(&client->dev, "port_id %d => cpld_port %d, group %d(%d)\n", data->port_id,
				data->cpld_port + 1, group + 1, group_port + 1);

	value = i2c_smbus_read_word_data(client, get_group_cmd(group));
	if (value < 0)
		return -ENODEV;

	dev_dbg(&client->dev, "read group%d value= %x\n", group + 1, value);

	value >>= (group_port * 4);
	value &= PRE_N_MASK;

	//FIXME: if present is not low active
	return sprintf(buf, "%d\n", value ? 0 : 1);
}

static ssize_t get_rx_los(struct device *dev,
			     struct device_attribute *devattr,
			     char *buf)
{
	struct sfp_data *data = dev_get_drvdata(dev);
	struct i2c_client *client = data->cpld_client;
	u8 group = (u8)(data->cpld_port / 4);
	u8 group_port = data->cpld_port % 4;
	s32 value;

	dev_dbg(&client->dev, "port_id %d => cpld_port %d, group %d(%d)\n", data->port_id,
				data->cpld_port + 1, group + 1, group_port + 1);

	value = i2c_smbus_read_word_data(client, get_group_cmd(group));
	if (value < 0)
		return -ENODEV;

	dev_dbg(&client->dev, "read group%d value= %x\n", group + 1, value);

	value >>= (group_port * 4);
	value &= RX_LOS_MASK;

	return sprintf(buf, "%d\n", value ? 1 : 0);
}
static ssize_t set_tx_dis(struct device *dev,
			    struct device_attribute *devattr,
			    const char *buf,
			    size_t count)
{
	struct sfp_data *data = dev_get_drvdata(dev);
	struct i2c_client *client = data->cpld_client;
	u8 group = (u8)(data->cpld_port / 4);
	u8 group_port = data->cpld_port % 4;
	s32 value;
	long disable;

	dev_dbg(&client->dev, "port_id %d => cpld_port %d, group %d(%d)\n", data->port_id,
				data->cpld_port + 1, group + 1, group_port + 1);

	if (kstrtol(buf, 0, &disable))
		return -EINVAL;

	if ((disable != 1) && (disable != 0))
		return -EINVAL;

//	mutex_lock(&data->lock);
	value = i2c_smbus_read_word_data(client, get_group_cmd(group));
	if (value < 0)
		return -ENODEV;

	dev_dbg(&client->dev, "read group%d value= %x\n", group + 1, value);

	value &= ~(TX_DIS_MASK << (group_port * 4));
	if (disable)
		value |= (TX_DIS_MASK << (group_port * 4));

	dev_dbg(&client->dev, "write group%d value= %x\n", group + 1, value);

	i2c_smbus_write_word_data(client, get_group_cmd(group), (u16)value);
//	mutex_unlock(&data->lock);

	return count;
}
//SFP

//static DEVICE_ATTR(led_enable, S_IWUSR | S_IRUGO, get_led_enable, set_led_enable);
//static DEVICE_ATTR(monitor_enable, S_IWUSR | S_IRUGO, get_monitor_enable, set_monitor_enable);
//SFP
static DEVICE_ATTR(tx_fault, S_IRUGO, get_tx_fault, NULL);
static DEVICE_ATTR(tx_dis, S_IWUSR | S_IRUGO, get_tx_dis, set_tx_dis);
static DEVICE_ATTR(pre_n, S_IRUGO, get_pre_n, NULL);
static DEVICE_ATTR(rx_los, S_IRUGO, get_rx_los, NULL);

static const struct attribute *sfp_attrs[] = {
//	&dev_attr_led_enable.attr,
	&dev_attr_tx_fault.attr,
	&dev_attr_tx_dis.attr,
	&dev_attr_pre_n.attr,
	&dev_attr_rx_los.attr,
	NULL,
};

static const struct attribute_group sfp_attr_group = {
	.attrs = (struct attribute **) sfp_attrs,
};

static int cpld_probe(struct i2c_client *client,
			 const struct i2c_device_id *id)
{
	struct cpld_data *data;
	struct sfp_data *port_data;
//	struct i2c_monitor_data *monitor_data;
	struct device *port_dev;
//	struct device *i2c_dev;
	int port_nr, i=0, err;
	char name[I2C_NAME_SIZE], type[I2C_NAME_SIZE];

	printk("cpld cpld_probe\n");

	while(id->name[i])
	{
		name[i]=tolower(id->name[i]);
		i++;
	}
	name[i]='\0';
	strncpy(type,name+5,strlen(name)-5);
	type[strlen(name)-5]='\0';

	if (!cpld_class)
	{
		cpld_class = class_create(THIS_MODULE, name);
		if (IS_ERR(cpld_class)) {
			pr_err("couldn't create sysfs class\n");
			return PTR_ERR(cpld_class);
		}
	}

	data = devm_kzalloc(&client->dev, sizeof(struct cpld_data),
			    GFP_KERNEL);
	if (!data)
		return -ENOMEM;

	/* register sfp port data to sysfs */
	for (i = 0; i < 16; i++)
	{
		port_nr = ida_simple_get(&cpld_ida, 1, 99, GFP_KERNEL);
		if (port_nr < 0)
			return ERR_PTR(port_nr);

		port_data = kzalloc(sizeof(struct sfp_data), GFP_KERNEL);

		port_dev = device_create(cpld_class, &client->dev, MKDEV(0,0), port_data, CPLD_ID_FORMAT, port_nr);
		if (IS_ERR(port_dev)) {
			err = PTR_ERR(port_dev);
			printk("err_status\n");
		}

		data->port_dev[i] = port_dev;
		data->port_data[i] = port_data;

		strcpy(port_data->type, type);

		dev_info(&client->dev, "Register %s port-%d\n", port_data->type , port_nr);

		/* FIXME: implement Logical/Physical port remapping */
		//port_data->cpld_port = i;
		port_data->cpld_port = port_remapping(i);
		sprintf(port_data->name, "port-%d", port_nr);
		port_data->port_id = port_nr;
		dev_set_drvdata(port_dev, port_data);
		port_dev->init_name = port_data->name;
		port_data->cpld_client = client;

		err = sysfs_create_group(&port_dev->kobj, &sfp_attr_group);
		// if (status)	printk("err status\n");
	}

	i2c_set_clientdata(client, data);
	mutex_init(&data->lock);

	dev_info(&client->dev, "%s device found\n", client->name);


	return 0;

//FIXME: implement error check
//exit_remove:
//	sysfs_remove_group(&client->dev.kobj, &data->attrs);
	return err;
}

/* FIXME: for older kernel doesn't with idr_is_empty function, implement here */
#if 1
static int idr_has_entry(int id, void *p, void *data)
{
	return 1;
}

static bool cpld_idr_is_empty(struct idr *idp)
{
	return !idr_for_each(idp, idr_has_entry, NULL);
}
#endif

static int cpld_remove(struct i2c_client *client)
{
	struct cpld_data *data = i2c_get_clientdata(client);
	int i;
//	int id;

	for (i = 15; i >= 0; i--)
	{
		dev_info(data->port_dev[i], "Remove %s port-%d\n", data->port_data[i]->type , data->port_data[i]->port_id);
		device_unregister(data->port_dev[i]);
		ida_simple_remove(&cpld_ida, data->port_data[i]->port_id);
		kfree(data->port_data[i]);
	}

	if (cpld_idr_is_empty(&cpld_ida.idr))
		class_destroy(cpld_class);

	return 0;
}

module_i2c_driver(cpld_driver);

MODULE_AUTHOR("Jonathan Tsai <jonathan.tsai@quantatw.com>");
MODULE_VERSION("1.0");
MODULE_DESCRIPTION("Quanta Switch SFP28 CPLD driver");
MODULE_LICENSE("GPL");
