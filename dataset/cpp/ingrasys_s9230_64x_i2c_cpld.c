/*
 * A i2c cpld driver for the ingrasys_s9230_64x
 *
 * Copyright (C) 2017 Ingrasys Technology Corporation.
 * Leo Lin <feng.lee.usa@ingrasys.com>
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
#include <linux/i2c.h>
#include <linux/slab.h>
#include <linux/list.h>
#include <linux/dmi.h>
#include <linux/hwmon.h>
#include <linux/hwmon-sysfs.h>
#include <linux/err.h>
#include <linux/mutex.h>

#ifndef INGRASYS_S9230_64X_I2C_CPLD_H
#define INGRASYS_S9230_64X_I2C_CPLD_H

/* CPLD device index value */
enum cpld_id {
    cpld1,
    cpld2,
    cpld3,
    cpld4,
    cpld5
};

/* port number on CPLD */
#define CPLD_1_PORT_NUM 12
#define CPLD_2_PORT_NUM 13

/* QSFP port number */
#define QSFP_MAX_PORT_NUM   64
#define QSFP_MIN_PORT_NUM   1

/* SFP+ port number */
#define SFP_MAX_PORT_NUM    2
#define SFP_MIN_PORT_NUM    1


/* CPLD registers */
#define CPLD_BOARD_TYPE_REG             0x0
#define CPLD_EXT_BOARD_TYPE_REG         0x7
#define CPLD_VERSION_REG                0x1
#define CPLD_ID_REG                     0x2
#define CPLD_QSFP_PORT_STATUS_BASE_REG  0x20
#define CPLD_QSFP_PORT_CONFIG_BASE_REG  0x30
#define CPLD_QSFP_PORT_INTERRUPT_REG    0x40
#define CPLD_SFP_PORT_STATUS_REG        0x2F
#define CPLD_SFP_PORT_CONFIG_REG        0x3F
#define CPLD_QSFP_PORT_INTERRUPT_REG    0x40
#define CPLD_10GMUX_CONFIG_REG          0x41
#define CPLD_BMC_STATUS_REG             0x42
#define CPLD_BMC_WATCHDOG_REG           0x43
#define CPLD_USB_STATUS_REG             0x44
#define CPLD_REST_CONTROL_REG           0x4A


/* bit definition for register value */
enum CPLD_QSFP_PORT_STATUS_BITS {
    CPLD_QSFP_PORT_STATUS_INT_BIT,
    CPLD_QSFP_PORT_STATUS_ABS_BIT,
};
enum CPLD_QSFP_PORT_CONFIG_BITS {
    CPLD_QSFP_PORT_CONFIG_RESET_BIT,
    CPLD_QSFP_PORT_CONFIG_RESERVE_BIT,
    CPLD_QSFP_PORT_CONFIG_LPMODE_BIT,
};
enum CPLD_SFP_PORT_STATUS_BITS {
    CPLD_SFP_PORT_STATUS_PRESENT_BIT,
    CPLD_SFP_PORT_STATUS_TXFAULT_BIT,
    CPLD_SFP_PORT_STATUS_RXLOS_BIT,
};
enum CPLD_SFP_PORT_CONFIG_BITS {
    CPLD_SFP_PORT_CONFIG_TXDIS_BIT,
    CPLD_SFP_PORT_CONFIG_RS_BIT,
    CPLD_SFP_PORT_CONFIG_TS_BIT,
};
enum CPLD_10GMUX_CONFIG_BITS {
    CPLD_10GMUX_CONFIG_ENSMB_BIT,
    CPLD_10GMUX_CONFIG_ENINPUT_BIT,
    CPLD_10GMUX_CONFIG_SEL1_BIT,
    CPLD_10GMUX_CONFIG_SEL0_BIT,
};
enum CPLD_BMC_WATCHDOG_BITS {
    CPLD_10GMUX_CONFIG_ENTIMER_BIT,
    CPLD_10GMUX_CONFIG_TIMEOUT_BIT,
};
enum CPLD_RESET_CONTROL_BITS {
    CPLD_RESET_CONTROL_SWRST_BIT,
    CPLD_RESET_CONTROL_CP2104RST_BIT,
    CPLD_RESET_CONTROL_82P33814RST_BIT,
    CPLD_RESET_CONTROL_BMCRST_BIT,
};

/* bit field structure for register value */
struct cpld_reg_board_type_t {
    u8 build_rev:2;
    u8 hw_rev:2;
    u8 board_id:4;
};

struct cpld_reg_version_t {
    u8 revision:6;
    u8 release:1;
    u8 reserve:1;
};

struct cpld_reg_id_t {
    u8 id:3;
    u8 release:5;
};

/* common manipulation */
#define INVALID(i, min, max)    ((i < min) || (i > max) ? 1u : 0u)
#define READ_BIT(val, bit)      ((0u == (val & (1<<bit))) ? 0u : 1u)
#define SET_BIT(val, bit)       (val |= (1 << bit))
#define CLEAR_BIT(val, bit)     (val &= ~(1 << bit))
#define TOGGLE_BIT(val, bit)    (val ^= (1 << bit))
#define _BIT(n)                 (1<<(n))
#define _BIT_MASK(len)          (BIT(len)-1)

/* bitfield of register manipulation */
#define READ_BF(bf_struct, val, bf_name, bf_value) \
    (bf_value = ((struct bf_struct *)&val)->bf_name)
#define READ_BF_1(bf_struct, val, bf_name, bf_value) \
    bf_struct bf; \
    bf.data = val; \
    bf_value = bf.bf_name
#define BOARD_TYPE_BUILD_REV_GET(val, res) \
    READ_BF(cpld_reg_board_type_t, val, build_rev, res)
#define BOARD_TYPE_HW_REV_GET(val, res) \
    READ_BF(cpld_reg_board_type_t, val, hw_rev, res)
#define BOARD_TYPE_BOARD_ID_GET(val, res) \
    READ_BF(cpld_reg_board_type_t, val, board_id, res)
#define CPLD_VERSION_REV_GET(val, res) \
    READ_BF(cpld_reg_version_t, val, revision, res)
#define CPLD_VERSION_REL_GET(val, res) \
    READ_BF(cpld_reg_version_t, val, release, res)
#define CPLD_ID_ID_GET(val, res) \
    READ_BF(cpld_reg_id_t, val, id, res)
#define CPLD_ID_REL_GET(val, res) \
    READ_BF(cpld_reg_id_t, val, release, res)
/* QSFP/SFP registers manipulation */
#define QSFP_TO_CPLD_IDX(qsfp_port, cpld_index, cpld_port) \
{ \
    if (QSFP_MIN_PORT_NUM <= qsfp_port && qsfp_port <= CPLD_1_PORT_NUM) { \
        cpld_index = cpld1; \
        cpld_port = qsfp_port - 1; \
    } else if (CPLD_1_PORT_NUM < qsfp_port \
        && qsfp_port <= QSFP_MAX_PORT_NUM) { \
        cpld_index = cpld2 + (qsfp_port - 1 - CPLD_1_PORT_NUM) \
                / CPLD_2_PORT_NUM; \
        cpld_port = (qsfp_port - 1 - CPLD_1_PORT_NUM) % \
                CPLD_2_PORT_NUM; \
    } else { \
        cpld_index = 0; \
        cpld_port = 0; \
    } \
}
#define SFP_TO_CPLD_IDX(sfp_port, cpld_index) \
    (cpld_index = sfp_port - SFP_MIN_PORT_NUM)
#define QSFP_PORT_STATUS_REG(cpld_port) \
    (CPLD_QSFP_PORT_STATUS_BASE_REG + cpld_port)
#define QSFP_PORT_CONFIG_REG(cpld_port) \
    (CPLD_QSFP_PORT_CONFIG_BASE_REG + cpld_port)
#define QSFP_PORT_INT_BIT_GET(port_status_value) \
    READ_BIT(port_status_value, CPLD_QSFP_PORT_STATUS_INT_BIT)
#define QSFP_PORT_ABS_BIT_GET(port_status_value) \
    READ_BIT(port_status_value, CPLD_QSFP_PORT_STATUS_ABS_BIT)
#define QSFP_PORT_RESET_BIT_GET(port_config_value) \
    READ_BIT(port_config_value, CPLD_QSFP_PORT_CONFIG_RESET_BIT)
#define QSFP_PORT_LPMODE_BIT_GET(port_config_value) \
    READ_BIT(port_config_value, CPLD_QSFP_PORT_CONFIG_LPMODE_BIT)
#define QSFP_PORT_RESET_BIT_SET(port_config_value) \
    SET_BIT(port_config_value, CPLD_QSFP_PORT_CONFIG_RESET_BIT)
#define QSFP_PORT_RESET_BIT_CLEAR(port_config_value) \
    CLEAR_BIT(port_config_value, CPLD_QSFP_PORT_CONFIG_RESET_BIT)
#define QSFP_PORT_LPMODE_BIT_SET(port_config_value) \
    SET_BIT(port_config_value, CPLD_QSFP_PORT_CONFIG_LPMODE_BIT)
#define QSFP_PORT_LPMODE_BIT_CLEAR(port_config_value) \
    CLEAR_BIT(port_config_value, CPLD_QSFP_PORT_CONFIG_LPMODE_BIT)
#define SFP_PORT_PRESENT_BIT_GET(port_status_value) \
    READ_BIT(port_status_value, CPLD_SFP_PORT_STATUS_PRESENT_BIT)

#define SFP_PORT_TXFAULT_BIT_GET(port_status_value) \
    READ_BIT(port_status_value, CPLD_SFP_PORT_STATUS_TXFAULT_BIT)
#define SFP_PORT_RXLOS_BIT_GET(port_status_value) \
    READ_BIT(port_status_value, CPLD_SFP_PORT_STATUS_RXLOS_BIT)
#define SFP_PORT_TXDIS_BIT_GET(port_status_value) \
    READ_BIT(port_status_value, CPLD_SFP_PORT_CONFIG_TXDIS_BIT)
#define SFP_PORT_RS_BIT_GET(port_config_value) \
    READ_BIT(port_config_value, CPLD_SFP_PORT_CONFIG_RS_BIT)
#define SFP_PORT_TS_BIT_GET(port_config_value) \
    READ_BIT(port_config_value, CPLD_SFP_PORT_CONFIG_TS_BIT)
#define SFP_PORT_TXDIS_BIT_SET(port_config_value) \
    SET_BIT(port_config_value, CPLD_SFP_PORT_CONFIG_TXDIS_BIT)
#define SFP_PORT_TXDIS_BIT_CLEAR(port_config_value) \
    CLEAR_BIT(port_config_value, CPLD_SFP_PORT_CONFIG_TXDIS_BIT)
#define SFP_PORT_RS_BIT_SET(port_config_value) \
    SET_BIT(port_config_value, CPLD_SFP_PORT_CONFIG_RS_BIT)
#define SFP_PORT_RS_BIT_CLEAR(port_config_value) \
    CLEAR_BIT(port_config_value, CPLD_SFP_PORT_CONFIG_RS_BIT)
#define SFP_PORT_TS_BIT_SET(port_config_value) \
    SET_BIT(port_config_value, CPLD_SFP_PORT_CONFIG_TS_BIT)
#define SFP_PORT_TS_BIT_CLEAR(port_config_value) \
    CLEAR_BIT(port_config_value, CPLD_SFP_PORT_CONFIG_TS_BIT)

/* CPLD access functions */
extern int ingrasys_i2c_cpld_get_qsfp_port_status_val(u8 port_num);
extern int ingrasys_i2c_cpld_get_qsfp_port_config_val(u8 port_num);
extern int ingrasys_i2c_cpld_set_qsfp_port_config_val(u8 port_num, u8 reg_val);
extern int ingrasys_i2c_cpld_get_sfp_port_status_val(u8 port_num);
extern int ingrasys_i2c_cpld_get_sfp_port_config_val(u8 port_num);
extern int ingrasys_i2c_cpld_set_sfp_port_config_val(u8 port_num, u8 reg_val);
extern u8 fp_port_to_phy_port(u8 fp_port);
#endif

#ifdef DEBUG
#define DEBUG_PRINT(fmt, args...) \
    printk(KERN_INFO "%s:%s[%d]: " fmt "\r\n", \
            __FILE__, __func__, __LINE__, ##args)
#else
#define DEBUG_PRINT(fmt, args...)
#endif

#define I2C_READ_BYTE_DATA(ret, lock, i2c_client, reg) \
{ \
    mutex_lock(lock); \
    ret = i2c_smbus_read_byte_data(i2c_client, reg); \
    mutex_unlock(lock); \
}
#define I2C_WRITE_BYTE_DATA(ret, lock, i2c_client, reg, val) \
{ \
        mutex_lock(lock); \
        ret = i2c_smbus_write_byte_data(i2c_client, reg, val); \
        mutex_unlock(lock); \
}

/* CPLD sysfs attributes index  */
enum s9230_64x_cpld_sysfs_attributes {
    CPLD_ACCESS_REG,
    CPLD_REGISTER_VAL,
    CPLD_PORT_START,
    CPLD_PORTS,
    CPLD_VERSION,
    CPLD_ID,
    CPLD_BOARD_TYPE,
    CPLD_EXT_BOARD_TYPE,
    CPLD_QSFP_PORT_STATUS_1,
    CPLD_QSFP_PORT_STATUS_2,
    CPLD_QSFP_PORT_STATUS_3,
    CPLD_QSFP_PORT_STATUS_4,
    CPLD_QSFP_PORT_STATUS_5,
    CPLD_QSFP_PORT_STATUS_6,
    CPLD_QSFP_PORT_STATUS_7,
    CPLD_QSFP_PORT_STATUS_8,
    CPLD_QSFP_PORT_STATUS_9,
    CPLD_QSFP_PORT_STATUS_10,
    CPLD_QSFP_PORT_STATUS_11,
    CPLD_QSFP_PORT_STATUS_12,
    CPLD_QSFP_PORT_STATUS_13,
    CPLD_QSFP_PORT_CONFIG_1,
    CPLD_QSFP_PORT_CONFIG_2,
    CPLD_QSFP_PORT_CONFIG_3,
    CPLD_QSFP_PORT_CONFIG_4,
    CPLD_QSFP_PORT_CONFIG_5,
    CPLD_QSFP_PORT_CONFIG_6,
    CPLD_QSFP_PORT_CONFIG_7,
    CPLD_QSFP_PORT_CONFIG_8,
    CPLD_QSFP_PORT_CONFIG_9,
    CPLD_QSFP_PORT_CONFIG_10,
    CPLD_QSFP_PORT_CONFIG_11,
    CPLD_QSFP_PORT_CONFIG_12,
    CPLD_QSFP_PORT_CONFIG_13,
    CPLD_QSFP_PORT_INTERRUPT,
    CPLD_SFP_PORT_STATUS,
    CPLD_SFP_PORT_CONFIG,
    CPLD_10GMUX_CONFIG,
    CPLD_BMC_STATUS,
    CPLD_BMC_WATCHDOG,
    CPLD_USB_STATUS,
};

/* CPLD sysfs attributes hook functions  */
static ssize_t read_access_register(struct device *dev,
                struct device_attribute *da, char *buf);
static ssize_t write_access_register(struct device *dev,
        struct device_attribute *da, const char *buf, size_t count);
static ssize_t read_register_value(struct device *dev,
                struct device_attribute *da, char *buf);
static ssize_t write_register_value(struct device *dev,
        struct device_attribute *da, const char *buf, size_t count);
static ssize_t get_qsfp_port_start(struct device *dev,
                struct device_attribute *da, char *buf);
static ssize_t get_qsfp_ports(struct device *dev,
                struct device_attribute *da, char *buf);
static ssize_t read_cpld_version(struct device *dev,
                struct device_attribute *da, char *buf);
static ssize_t read_cpld_id(struct device *dev,
                struct device_attribute *da, char *buf);
static ssize_t read_board_type(struct device *dev,
                struct device_attribute *da, char *buf);
static ssize_t read_ext_board_type(struct device *dev,
                struct device_attribute *da, char *buf);
static ssize_t read_qsfp_port_status(struct device *dev,
                struct device_attribute *da, char *buf);
static ssize_t read_qsfp_port_config(struct device *dev,
                struct device_attribute *da, char *buf);
static ssize_t write_qsfp_port_config(struct device *dev,
        struct device_attribute *da, const char *buf, size_t count);
static ssize_t read_qsfp_port_interrupt(struct device *dev,
                struct device_attribute *da, char *buf);
static ssize_t read_sfp_port_status(struct device *dev,
                struct device_attribute *da, char *buf);
static ssize_t read_sfp_port_config(struct device *dev,
                struct device_attribute *da, char *buf);
static ssize_t write_sfp_port_config(struct device *dev,
        struct device_attribute *da, const char *buf, size_t count);
static ssize_t read_10gmux_config(struct device *dev,
                struct device_attribute *da, char *buf);
static ssize_t write_10gmux_config(struct device *dev,
        struct device_attribute *da, const char *buf, size_t count);
static ssize_t read_bmc_status(struct device *dev,
                struct device_attribute *da, char *buf);
static ssize_t read_bmc_watchdog(struct device *dev,
                struct device_attribute *da, char *buf);
static ssize_t write_bmc_watchdog(struct device *dev,
        struct device_attribute *da, const char *buf, size_t count);
static ssize_t read_usb_status(struct device *dev,
                struct device_attribute *da, char *buf);

static LIST_HEAD(cpld_client_list);  /* client list for cpld */
static struct mutex list_lock;  /* mutex for client list */

struct cpld_client_node {
    struct i2c_client *client;
    struct list_head   list;
};

struct cpld_data {
    int index;                  /* CPLD index */
    struct mutex access_lock;       /* mutex for cpld access */
    u8 access_reg;              /* register to access */
};

/* CPLD device id and data */
static const struct i2c_device_id ingrasys_i2c_cpld_id[] = {
    { "ingrasys_cpld1",  cpld1 },
    { "ingrasys_cpld2",  cpld2 },
    { "ingrasys_cpld3",  cpld3 },
    { "ingrasys_cpld4",  cpld4 },
    { "ingrasys_cpld5",  cpld5 },
    {}
};

/* Addresses scanned for ingrasys_i2c_cpld */
static const unsigned short cpld_i2c_addr[] = { 0x33, I2C_CLIENT_END };

/* define all support register access of cpld in attribute */
static SENSOR_DEVICE_ATTR(cpld_access_register, S_IWUSR | S_IRUGO,
        read_access_register, write_access_register, CPLD_ACCESS_REG);
static SENSOR_DEVICE_ATTR(cpld_register_value, S_IWUSR | S_IRUGO,
        read_register_value, write_register_value, CPLD_REGISTER_VAL);
static SENSOR_DEVICE_ATTR(cpld_qsfp_port_start, S_IRUGO,
            get_qsfp_port_start, NULL, CPLD_PORT_START);
static SENSOR_DEVICE_ATTR(cpld_qsfp_ports, S_IRUGO,
                get_qsfp_ports, NULL, CPLD_PORTS);
static SENSOR_DEVICE_ATTR(cpld_version, S_IRUGO,
                read_cpld_version, NULL, CPLD_VERSION);
static SENSOR_DEVICE_ATTR(cpld_id, S_IRUGO, read_cpld_id, NULL, CPLD_ID);
static SENSOR_DEVICE_ATTR(cpld_board_type, S_IRUGO,
                read_board_type, NULL, CPLD_BOARD_TYPE);
static SENSOR_DEVICE_ATTR(cpld_ext_board_type, S_IRUGO,
        read_ext_board_type, NULL, CPLD_EXT_BOARD_TYPE);
static SENSOR_DEVICE_ATTR(cpld_qsfp_port_status_1, S_IRUGO,
        read_qsfp_port_status, NULL, CPLD_QSFP_PORT_STATUS_1);
static SENSOR_DEVICE_ATTR(cpld_qsfp_port_status_2, S_IRUGO,
        read_qsfp_port_status, NULL, CPLD_QSFP_PORT_STATUS_2);
static SENSOR_DEVICE_ATTR(cpld_qsfp_port_status_3, S_IRUGO,
        read_qsfp_port_status, NULL, CPLD_QSFP_PORT_STATUS_3);
static SENSOR_DEVICE_ATTR(cpld_qsfp_port_status_4, S_IRUGO,
        read_qsfp_port_status, NULL, CPLD_QSFP_PORT_STATUS_4);
static SENSOR_DEVICE_ATTR(cpld_qsfp_port_status_5, S_IRUGO,
        read_qsfp_port_status, NULL, CPLD_QSFP_PORT_STATUS_5);
static SENSOR_DEVICE_ATTR(cpld_qsfp_port_status_6, S_IRUGO,
        read_qsfp_port_status, NULL, CPLD_QSFP_PORT_STATUS_6);
static SENSOR_DEVICE_ATTR(cpld_qsfp_port_status_7, S_IRUGO,
        read_qsfp_port_status, NULL, CPLD_QSFP_PORT_STATUS_7);
static SENSOR_DEVICE_ATTR(cpld_qsfp_port_status_8, S_IRUGO,
        read_qsfp_port_status, NULL, CPLD_QSFP_PORT_STATUS_8);
static SENSOR_DEVICE_ATTR(cpld_qsfp_port_status_9, S_IRUGO,
        read_qsfp_port_status, NULL, CPLD_QSFP_PORT_STATUS_9);
static SENSOR_DEVICE_ATTR(cpld_qsfp_port_status_10, S_IRUGO,
        read_qsfp_port_status, NULL, CPLD_QSFP_PORT_STATUS_10);
static SENSOR_DEVICE_ATTR(cpld_qsfp_port_status_11, S_IRUGO,
        read_qsfp_port_status, NULL, CPLD_QSFP_PORT_STATUS_11);
static SENSOR_DEVICE_ATTR(cpld_qsfp_port_status_12, S_IRUGO,
        read_qsfp_port_status, NULL, CPLD_QSFP_PORT_STATUS_12);
static SENSOR_DEVICE_ATTR(cpld_qsfp_port_status_13, S_IRUGO,
        read_qsfp_port_status, NULL, CPLD_QSFP_PORT_STATUS_13);
static SENSOR_DEVICE_ATTR(cpld_qsfp_port_config_1, S_IWUSR | S_IRUGO,
    read_qsfp_port_config, write_qsfp_port_config, CPLD_QSFP_PORT_CONFIG_1);
static SENSOR_DEVICE_ATTR(cpld_qsfp_port_config_2, S_IWUSR | S_IRUGO,
                read_qsfp_port_config, write_qsfp_port_config,
                CPLD_QSFP_PORT_CONFIG_2);
static SENSOR_DEVICE_ATTR(cpld_qsfp_port_config_3, S_IWUSR | S_IRUGO,
                read_qsfp_port_config, write_qsfp_port_config,
                CPLD_QSFP_PORT_CONFIG_3);
static SENSOR_DEVICE_ATTR(cpld_qsfp_port_config_4, S_IWUSR | S_IRUGO,
                read_qsfp_port_config, write_qsfp_port_config,
                CPLD_QSFP_PORT_CONFIG_4);
static SENSOR_DEVICE_ATTR(cpld_qsfp_port_config_5, S_IWUSR | S_IRUGO,
                read_qsfp_port_config, write_qsfp_port_config,
                CPLD_QSFP_PORT_CONFIG_5);
static SENSOR_DEVICE_ATTR(cpld_qsfp_port_config_6, S_IWUSR | S_IRUGO,
                read_qsfp_port_config, write_qsfp_port_config,
                CPLD_QSFP_PORT_CONFIG_6);
static SENSOR_DEVICE_ATTR(cpld_qsfp_port_config_7, S_IWUSR | S_IRUGO,
                read_qsfp_port_config, write_qsfp_port_config,
                CPLD_QSFP_PORT_CONFIG_7);
static SENSOR_DEVICE_ATTR(cpld_qsfp_port_config_8, S_IWUSR | S_IRUGO,
                read_qsfp_port_config, write_qsfp_port_config,
                CPLD_QSFP_PORT_CONFIG_8);
static SENSOR_DEVICE_ATTR(cpld_qsfp_port_config_9, S_IWUSR | S_IRUGO,
                read_qsfp_port_config, write_qsfp_port_config,
                CPLD_QSFP_PORT_CONFIG_9);
static SENSOR_DEVICE_ATTR(cpld_qsfp_port_config_10, S_IWUSR | S_IRUGO,
                read_qsfp_port_config, write_qsfp_port_config,
                CPLD_QSFP_PORT_CONFIG_10);
static SENSOR_DEVICE_ATTR(cpld_qsfp_port_config_11, S_IWUSR | S_IRUGO,
                read_qsfp_port_config, write_qsfp_port_config,
                CPLD_QSFP_PORT_CONFIG_11);
static SENSOR_DEVICE_ATTR(cpld_qsfp_port_config_12, S_IWUSR | S_IRUGO,
                read_qsfp_port_config, write_qsfp_port_config,
                CPLD_QSFP_PORT_CONFIG_12);
static SENSOR_DEVICE_ATTR(cpld_qsfp_port_config_13, S_IWUSR | S_IRUGO,
                read_qsfp_port_config, write_qsfp_port_config,
                CPLD_QSFP_PORT_CONFIG_13);
static SENSOR_DEVICE_ATTR(cpld_qsfp_port_interrupt, S_IRUGO,
        read_qsfp_port_interrupt, NULL, CPLD_QSFP_PORT_INTERRUPT);
static SENSOR_DEVICE_ATTR(cpld_sfp_port_status, S_IRUGO,
            read_sfp_port_status, NULL, CPLD_SFP_PORT_STATUS);
static SENSOR_DEVICE_ATTR(cpld_sfp_port_config, S_IWUSR | S_IRUGO,
    read_sfp_port_config, write_sfp_port_config, CPLD_SFP_PORT_CONFIG);
static SENSOR_DEVICE_ATTR(cpld_10gmux_config, S_IWUSR | S_IRUGO,
                read_10gmux_config, write_10gmux_config,
                CPLD_10GMUX_CONFIG);
static SENSOR_DEVICE_ATTR(cpld_bmc_status, S_IRUGO,
                read_bmc_status, NULL, CPLD_BMC_STATUS);
static SENSOR_DEVICE_ATTR(cpld_bmc_watchdog, S_IWUSR | S_IRUGO,
                read_bmc_watchdog, write_bmc_watchdog,
                CPLD_BMC_WATCHDOG);
static SENSOR_DEVICE_ATTR(cpld_usb_status, S_IRUGO,
                read_usb_status, NULL, CPLD_USB_STATUS);


/* define support attributes of cpldx , total 5 */
/* cpld 1 */
static struct attribute *s9230_64x_cpld1_attributes[] = {
    &sensor_dev_attr_cpld_access_register.dev_attr.attr,
    &sensor_dev_attr_cpld_register_value.dev_attr.attr,
    &sensor_dev_attr_cpld_qsfp_port_start.dev_attr.attr,
    &sensor_dev_attr_cpld_qsfp_ports.dev_attr.attr,
    &sensor_dev_attr_cpld_version.dev_attr.attr,
    &sensor_dev_attr_cpld_id.dev_attr.attr,
    &sensor_dev_attr_cpld_board_type.dev_attr.attr,
    &sensor_dev_attr_cpld_ext_board_type.dev_attr.attr,
    &sensor_dev_attr_cpld_qsfp_port_status_1.dev_attr.attr,
    &sensor_dev_attr_cpld_qsfp_port_status_2.dev_attr.attr,
    &sensor_dev_attr_cpld_qsfp_port_status_3.dev_attr.attr,
    &sensor_dev_attr_cpld_qsfp_port_status_4.dev_attr.attr,
    &sensor_dev_attr_cpld_qsfp_port_status_5.dev_attr.attr,
    &sensor_dev_attr_cpld_qsfp_port_status_6.dev_attr.attr,
    &sensor_dev_attr_cpld_qsfp_port_status_7.dev_attr.attr,
    &sensor_dev_attr_cpld_qsfp_port_status_8.dev_attr.attr,
    &sensor_dev_attr_cpld_qsfp_port_status_9.dev_attr.attr,
    &sensor_dev_attr_cpld_qsfp_port_status_10.dev_attr.attr,
    &sensor_dev_attr_cpld_qsfp_port_status_11.dev_attr.attr,
    &sensor_dev_attr_cpld_qsfp_port_status_12.dev_attr.attr,
    &sensor_dev_attr_cpld_qsfp_port_config_1.dev_attr.attr,
    &sensor_dev_attr_cpld_qsfp_port_config_2.dev_attr.attr,
    &sensor_dev_attr_cpld_qsfp_port_config_3.dev_attr.attr,
    &sensor_dev_attr_cpld_qsfp_port_config_4.dev_attr.attr,
    &sensor_dev_attr_cpld_qsfp_port_config_5.dev_attr.attr,
    &sensor_dev_attr_cpld_qsfp_port_config_6.dev_attr.attr,
    &sensor_dev_attr_cpld_qsfp_port_config_7.dev_attr.attr,
    &sensor_dev_attr_cpld_qsfp_port_config_8.dev_attr.attr,
    &sensor_dev_attr_cpld_qsfp_port_config_9.dev_attr.attr,
    &sensor_dev_attr_cpld_qsfp_port_config_10.dev_attr.attr,
    &sensor_dev_attr_cpld_qsfp_port_config_11.dev_attr.attr,
    &sensor_dev_attr_cpld_qsfp_port_config_12.dev_attr.attr,
    &sensor_dev_attr_cpld_qsfp_port_interrupt.dev_attr.attr,
    &sensor_dev_attr_cpld_sfp_port_status.dev_attr.attr,
    &sensor_dev_attr_cpld_sfp_port_config.dev_attr.attr,
    &sensor_dev_attr_cpld_10gmux_config.dev_attr.attr,
    &sensor_dev_attr_cpld_bmc_status.dev_attr.attr,
    &sensor_dev_attr_cpld_bmc_watchdog.dev_attr.attr,
    &sensor_dev_attr_cpld_usb_status.dev_attr.attr,
    NULL
};
/* cpld 2 */
static struct attribute *s9230_64x_cpld2_attributes[] = {
    &sensor_dev_attr_cpld_access_register.dev_attr.attr,
    &sensor_dev_attr_cpld_register_value.dev_attr.attr,
    &sensor_dev_attr_cpld_qsfp_port_start.dev_attr.attr,
    &sensor_dev_attr_cpld_qsfp_ports.dev_attr.attr,
    &sensor_dev_attr_cpld_version.dev_attr.attr,
    &sensor_dev_attr_cpld_id.dev_attr.attr,
    &sensor_dev_attr_cpld_qsfp_port_status_1.dev_attr.attr,
    &sensor_dev_attr_cpld_qsfp_port_status_2.dev_attr.attr,
    &sensor_dev_attr_cpld_qsfp_port_status_3.dev_attr.attr,
    &sensor_dev_attr_cpld_qsfp_port_status_4.dev_attr.attr,
    &sensor_dev_attr_cpld_qsfp_port_status_5.dev_attr.attr,
    &sensor_dev_attr_cpld_qsfp_port_status_6.dev_attr.attr,
    &sensor_dev_attr_cpld_qsfp_port_status_7.dev_attr.attr,
    &sensor_dev_attr_cpld_qsfp_port_status_8.dev_attr.attr,
    &sensor_dev_attr_cpld_qsfp_port_status_9.dev_attr.attr,
    &sensor_dev_attr_cpld_qsfp_port_status_10.dev_attr.attr,
    &sensor_dev_attr_cpld_qsfp_port_status_11.dev_attr.attr,
    &sensor_dev_attr_cpld_qsfp_port_status_12.dev_attr.attr,
    &sensor_dev_attr_cpld_qsfp_port_status_13.dev_attr.attr,
    &sensor_dev_attr_cpld_qsfp_port_config_1.dev_attr.attr,
    &sensor_dev_attr_cpld_qsfp_port_config_2.dev_attr.attr,
    &sensor_dev_attr_cpld_qsfp_port_config_3.dev_attr.attr,
    &sensor_dev_attr_cpld_qsfp_port_config_4.dev_attr.attr,
    &sensor_dev_attr_cpld_qsfp_port_config_5.dev_attr.attr,
    &sensor_dev_attr_cpld_qsfp_port_config_6.dev_attr.attr,
    &sensor_dev_attr_cpld_qsfp_port_config_7.dev_attr.attr,
    &sensor_dev_attr_cpld_qsfp_port_config_8.dev_attr.attr,
    &sensor_dev_attr_cpld_qsfp_port_config_9.dev_attr.attr,
    &sensor_dev_attr_cpld_qsfp_port_config_10.dev_attr.attr,
    &sensor_dev_attr_cpld_qsfp_port_config_11.dev_attr.attr,
    &sensor_dev_attr_cpld_qsfp_port_config_12.dev_attr.attr,
    &sensor_dev_attr_cpld_qsfp_port_config_13.dev_attr.attr,
    &sensor_dev_attr_cpld_qsfp_port_interrupt.dev_attr.attr,
    &sensor_dev_attr_cpld_sfp_port_status.dev_attr.attr,
    &sensor_dev_attr_cpld_sfp_port_config.dev_attr.attr,
    NULL
};
/* cpld 3 / cpld 4 / cpld 5 */
static struct attribute *s9230_64x_cpld345_attributes[] = {
&sensor_dev_attr_cpld_access_register.dev_attr.attr,
    &sensor_dev_attr_cpld_register_value.dev_attr.attr,
    &sensor_dev_attr_cpld_qsfp_port_start.dev_attr.attr,
    &sensor_dev_attr_cpld_qsfp_ports.dev_attr.attr,
    &sensor_dev_attr_cpld_version.dev_attr.attr,
    &sensor_dev_attr_cpld_id.dev_attr.attr,
    &sensor_dev_attr_cpld_qsfp_port_status_1.dev_attr.attr,
    &sensor_dev_attr_cpld_qsfp_port_status_2.dev_attr.attr,
    &sensor_dev_attr_cpld_qsfp_port_status_3.dev_attr.attr,
    &sensor_dev_attr_cpld_qsfp_port_status_4.dev_attr.attr,
    &sensor_dev_attr_cpld_qsfp_port_status_5.dev_attr.attr,
    &sensor_dev_attr_cpld_qsfp_port_status_6.dev_attr.attr,
    &sensor_dev_attr_cpld_qsfp_port_status_7.dev_attr.attr,
    &sensor_dev_attr_cpld_qsfp_port_status_8.dev_attr.attr,
    &sensor_dev_attr_cpld_qsfp_port_status_9.dev_attr.attr,
    &sensor_dev_attr_cpld_qsfp_port_status_10.dev_attr.attr,
    &sensor_dev_attr_cpld_qsfp_port_status_11.dev_attr.attr,
    &sensor_dev_attr_cpld_qsfp_port_status_12.dev_attr.attr,
    &sensor_dev_attr_cpld_qsfp_port_status_13.dev_attr.attr,
    &sensor_dev_attr_cpld_qsfp_port_config_1.dev_attr.attr,
    &sensor_dev_attr_cpld_qsfp_port_config_2.dev_attr.attr,
    &sensor_dev_attr_cpld_qsfp_port_config_3.dev_attr.attr,
    &sensor_dev_attr_cpld_qsfp_port_config_4.dev_attr.attr,
    &sensor_dev_attr_cpld_qsfp_port_config_5.dev_attr.attr,
    &sensor_dev_attr_cpld_qsfp_port_config_6.dev_attr.attr,
    &sensor_dev_attr_cpld_qsfp_port_config_7.dev_attr.attr,
    &sensor_dev_attr_cpld_qsfp_port_config_8.dev_attr.attr,
    &sensor_dev_attr_cpld_qsfp_port_config_9.dev_attr.attr,
    &sensor_dev_attr_cpld_qsfp_port_config_10.dev_attr.attr,
    &sensor_dev_attr_cpld_qsfp_port_config_11.dev_attr.attr,
    &sensor_dev_attr_cpld_qsfp_port_config_12.dev_attr.attr,
    &sensor_dev_attr_cpld_qsfp_port_config_13.dev_attr.attr,
    &sensor_dev_attr_cpld_qsfp_port_interrupt.dev_attr.attr,
    NULL
};

/* cpld 1 attributes group */
static const struct attribute_group s9230_64x_cpld1_group = {
    .attrs = s9230_64x_cpld1_attributes,
};
/* cpld 2 attributes group */
static const struct attribute_group s9230_64x_cpld2_group = {
    .attrs = s9230_64x_cpld2_attributes,
};
/* cpld 3/4/5 attributes group */
static const struct attribute_group s9230_64x_cpld345_group = {
    .attrs = s9230_64x_cpld345_attributes,
};

/* read access register from cpld data */
static ssize_t read_access_register(struct device *dev,
                    struct device_attribute *da,
                    char *buf)
{
    struct i2c_client *client = to_i2c_client(dev);
    struct cpld_data *data = i2c_get_clientdata(client);
    u8 reg = data->access_reg;

    return sprintf(buf, "0x%x\n", reg);
}

/* write access register to cpld data */
static ssize_t write_access_register(struct device *dev,
                    struct device_attribute *da,
                    const char *buf,
                    size_t count)
{
    struct i2c_client *client = to_i2c_client(dev);
    struct cpld_data *data = i2c_get_clientdata(client);
    u8 reg;

    if (kstrtou8(buf, 0, &reg) < 0)
        return -EINVAL;

    data->access_reg = reg;
    return count;
}

/* read the value of access register in cpld data */
static ssize_t read_register_value(struct device *dev,
                    struct device_attribute *da,
                    char *buf)
{
    struct i2c_client *client = to_i2c_client(dev);
    struct cpld_data *data = i2c_get_clientdata(client);
    u8 reg = data->access_reg;
    int reg_val;

    I2C_READ_BYTE_DATA(reg_val, &data->access_lock, client, reg);

    if (reg_val < 0)
        return -1;

    return sprintf(buf, "0x%x\n", reg_val);
}

/* wrtie the value to access register in cpld data */
static ssize_t write_register_value(struct device *dev,
                    struct device_attribute *da,
                    const char *buf,
                    size_t count)
{
    struct i2c_client *client = to_i2c_client(dev);
    struct cpld_data *data = i2c_get_clientdata(client);
    int ret = -EIO;
    u8 reg = data->access_reg;
    u8 reg_val;

    if (kstrtou8(buf, 0, &reg_val) < 0)
        return -EINVAL;

    I2C_WRITE_BYTE_DATA(ret, &data->access_lock, client, reg, reg_val);

    return count;
}

/* get qsfp port start number of the cpld device */
/* the start number use to tranlate qsfp port to cpld port */
/* the cpld port use to access the qsfp port register in cpld */
static ssize_t get_qsfp_port_start(struct device *dev,
                    struct device_attribute *da,
                    char *buf)
{
    struct sensor_device_attribute *attr = to_sensor_dev_attr(da);
    struct i2c_client *client = to_i2c_client(dev);
    struct cpld_data *data = i2c_get_clientdata(client);
    int port_base_num;

    if (attr->index == CPLD_PORT_START) {
        if (data->index == cpld1) {
            port_base_num = 1;
        } else {
            port_base_num = CPLD_1_PORT_NUM +
                CPLD_2_PORT_NUM*(data->index - 1) + 1;
        }
        return sprintf(buf, "%d\n", port_base_num);
    }
    return -1;
}

/* get total qsfp port which contain register in the cpld device */
static ssize_t get_qsfp_ports(struct device *dev,
                    struct device_attribute *da,
                    char *buf)
{
    struct sensor_device_attribute *attr = to_sensor_dev_attr(da);
    struct i2c_client *client = to_i2c_client(dev);
    struct cpld_data *data = i2c_get_clientdata(client);
    int ports;

    if (attr->index == CPLD_PORTS) {
        if (data->index == cpld1)
            ports = CPLD_1_PORT_NUM;
        else
            ports = CPLD_2_PORT_NUM;
        return sprintf(buf, "%d\n", ports);
    }
    return -1;
}

/* get cpdl version register value */
static ssize_t read_cpld_version(struct device *dev,
                    struct device_attribute *da,
                    char *buf)
{
    struct sensor_device_attribute *attr = to_sensor_dev_attr(da);
    struct i2c_client *client = to_i2c_client(dev);
    struct cpld_data *data = i2c_get_clientdata(client);
    u8 reg;
    int reg_val;

    if (attr->index == CPLD_VERSION) {
        reg = CPLD_VERSION_REG;
        I2C_READ_BYTE_DATA(reg_val, &data->access_lock, client, reg);
        if (reg_val < 0)
            return -1;
        return sprintf(buf, "0x%02x\n", reg_val);
    }
    return -1;
}

/* get cpdl id register value */
static ssize_t read_cpld_id(struct device *dev,
                struct device_attribute *da,
                char *buf)
{
    struct sensor_device_attribute *attr = to_sensor_dev_attr(da);
    struct i2c_client *client = to_i2c_client(dev);
    struct cpld_data *data = i2c_get_clientdata(client);
    u8 reg;
    int reg_val;

    if (attr->index == CPLD_ID) {
        reg = CPLD_ID_REG;
        I2C_READ_BYTE_DATA(reg_val, &data->access_lock, client, reg);
        if (reg_val < 0)
            return -1;
        return sprintf(buf, "0x%02x\n", reg_val);
    }
    return -1;
}

/* get board type register value */
static ssize_t read_board_type(struct device *dev,
                struct device_attribute *da,
                char *buf)
{
    struct sensor_device_attribute *attr = to_sensor_dev_attr(da);
    struct i2c_client *client = to_i2c_client(dev);
    struct cpld_data *data = i2c_get_clientdata(client);
    u8 reg;
    int reg_val;

    if (attr->index == CPLD_BOARD_TYPE) {
        reg = CPLD_BOARD_TYPE_REG;
        I2C_READ_BYTE_DATA(reg_val, &data->access_lock, client, reg);
        if (reg_val < 0)
            return -1;
        return sprintf(buf, "0x%02x\n", reg_val);
    }
    return -1;
}

/* get extend board type register value */
static ssize_t read_ext_board_type(struct device *dev,
                    struct device_attribute *da,
                    char *buf)
{
    struct sensor_device_attribute *attr = to_sensor_dev_attr(da);
    struct i2c_client *client = to_i2c_client(dev);
    struct cpld_data *data = i2c_get_clientdata(client);
    u8 reg;
    int reg_val;

    if (attr->index == CPLD_EXT_BOARD_TYPE) {
        reg = CPLD_EXT_BOARD_TYPE_REG;
        I2C_READ_BYTE_DATA(reg_val, &data->access_lock, client, reg);
        if (reg_val < 0)
            return -1;
        return sprintf(buf, "0x%02x\n", reg_val);
    }
    return -1;
}

/* get qsfp port status register value */
static ssize_t read_qsfp_port_status(struct device *dev,
                    struct device_attribute *da,
                    char *buf)
{
    struct sensor_device_attribute *attr = to_sensor_dev_attr(da);
    struct i2c_client *client = to_i2c_client(dev);
    struct cpld_data *data = i2c_get_clientdata(client);
    u8 reg;
    int reg_val;

    if (attr->index >= CPLD_QSFP_PORT_STATUS_1 &&
        attr->index <= CPLD_QSFP_PORT_STATUS_13) {
        reg = CPLD_QSFP_PORT_STATUS_BASE_REG +
            (attr->index - CPLD_QSFP_PORT_STATUS_1);
        I2C_READ_BYTE_DATA(reg_val, &data->access_lock, client, reg);
        if (reg_val < 0)
            return -1;
        return sprintf(buf, "0x%02x\n", reg_val);
    }
    return -1;
}

/* get qsfp port config register value */
static ssize_t read_qsfp_port_config(struct device *dev,
                    struct device_attribute *da,
                    char *buf)
{
    struct sensor_device_attribute *attr = to_sensor_dev_attr(da);
    struct i2c_client *client = to_i2c_client(dev);
    struct cpld_data *data = i2c_get_clientdata(client);
    u8 reg;
    int reg_val;

    if (attr->index >= CPLD_QSFP_PORT_CONFIG_1 &&
        attr->index <= CPLD_QSFP_PORT_CONFIG_13) {
        reg = CPLD_QSFP_PORT_CONFIG_BASE_REG +
            (attr->index - CPLD_QSFP_PORT_CONFIG_1);
        I2C_READ_BYTE_DATA(reg_val, &data->access_lock, client, reg);
        if (reg_val < 0)
            return -1;
        return sprintf(buf, "0x%02x\n", reg_val);
    }
    return -1;
}

/* set value to qsfp port config register */
static ssize_t write_qsfp_port_config(struct device *dev,
                    struct device_attribute *da,
                    const char *buf,
                    size_t count)
{
    struct sensor_device_attribute *attr = to_sensor_dev_attr(da);
    struct i2c_client *client = to_i2c_client(dev);
    struct cpld_data *data = i2c_get_clientdata(client);
    u8 reg, reg_val;
    int ret;

    if (kstrtou8(buf, 0, &reg_val) < 0)
        return -EINVAL;

    if (attr->index >= CPLD_QSFP_PORT_CONFIG_1 &&
        attr->index <= CPLD_QSFP_PORT_CONFIG_13) {
        reg = CPLD_QSFP_PORT_CONFIG_BASE_REG +
            (attr->index - CPLD_QSFP_PORT_CONFIG_1);
        I2C_WRITE_BYTE_DATA(ret, &data->access_lock,
                    client, reg, reg_val);
    }
    return count;
}

/* get qsfp port interrupt register value */
static ssize_t read_qsfp_port_interrupt(struct device *dev,
                    struct device_attribute *da,
                    char *buf)
{
    struct sensor_device_attribute *attr = to_sensor_dev_attr(da);
    struct i2c_client *client = to_i2c_client(dev);
    struct cpld_data *data = i2c_get_clientdata(client);
    u8 reg;
    int reg_val;

    if (attr->index == CPLD_QSFP_PORT_INTERRUPT) {
        reg = CPLD_QSFP_PORT_INTERRUPT_REG;
        I2C_READ_BYTE_DATA(reg_val, &data->access_lock, client, reg);
        if (reg_val < 0)
            return -1;
        return sprintf(buf, "0x%02x\n", reg_val);
    }
    return -1;
}

/* get sfp port status register value */
static ssize_t read_sfp_port_status(struct device *dev,
                    struct device_attribute *da,
                    char *buf)
{
    struct sensor_device_attribute *attr = to_sensor_dev_attr(da);
    struct i2c_client *client = to_i2c_client(dev);
    struct cpld_data *data = i2c_get_clientdata(client);
    u8 reg;
    int reg_val;

    if (attr->index == CPLD_SFP_PORT_STATUS) {
        reg = CPLD_SFP_PORT_STATUS_REG;
        I2C_READ_BYTE_DATA(reg_val, &data->access_lock, client, reg);
        if (reg_val < 0)
            return -1;
        return sprintf(buf, "0x%02x\n", reg_val);
    }
    return -1;
}

/* get sfp port config register value */
static ssize_t read_sfp_port_config(struct device *dev,
                    struct device_attribute *da,
                    char *buf)
{
    struct sensor_device_attribute *attr = to_sensor_dev_attr(da);
    struct i2c_client *client = to_i2c_client(dev);
    struct cpld_data *data = i2c_get_clientdata(client);
    u8 reg;
    int reg_val;

    if (attr->index == CPLD_SFP_PORT_CONFIG) {
        reg = CPLD_SFP_PORT_CONFIG_REG;
        I2C_READ_BYTE_DATA(reg_val, &data->access_lock, client, reg);
        if (reg_val < 0)
            return -1;
        return sprintf(buf, "0x%02x\n", reg_val);
    }
    return -1;
}

/* set value to sfp port config register */
static ssize_t write_sfp_port_config(struct device *dev,
                    struct device_attribute *da,
                    const char *buf,
                    size_t count)
{
    struct sensor_device_attribute *attr = to_sensor_dev_attr(da);
    struct i2c_client *client = to_i2c_client(dev);
    struct cpld_data *data = i2c_get_clientdata(client);
    u8 reg, reg_val;
    int ret;

    if (kstrtou8(buf, 0, &reg_val) < 0)
        return -EINVAL;

    if (attr->index == CPLD_SFP_PORT_CONFIG) {
        reg = CPLD_SFP_PORT_CONFIG_REG;
        I2C_WRITE_BYTE_DATA(ret, &data->access_lock,
                    client, reg, reg_val);
    }
    return count;
}

/* get 10g mux config register value */
static ssize_t read_10gmux_config(struct device *dev,
                    struct device_attribute *da,
                    char *buf)
{
    struct sensor_device_attribute *attr = to_sensor_dev_attr(da);
    struct i2c_client *client = to_i2c_client(dev);
    struct cpld_data *data = i2c_get_clientdata(client);
    u8 reg;
    int reg_val;

    if (attr->index == CPLD_10GMUX_CONFIG) {
        reg = CPLD_10GMUX_CONFIG_REG;
        I2C_READ_BYTE_DATA(reg_val, &data->access_lock, client, reg);
        if (reg_val < 0)
            return -1;
        return sprintf(buf, "0x%02x\n", reg_val);
    }
    return -1;
}

/* set value to 10g mux config register */
static ssize_t write_10gmux_config(struct device *dev,
                    struct device_attribute *da,
                    const char *buf,
                    size_t count)
{
    struct sensor_device_attribute *attr = to_sensor_dev_attr(da);
    struct i2c_client *client = to_i2c_client(dev);
    struct cpld_data *data = i2c_get_clientdata(client);
    u8 reg, reg_val;
    int ret;

    if (kstrtou8(buf, 0, &reg_val) < 0)
        return -EINVAL;

    if (attr->index == CPLD_10GMUX_CONFIG) {
        reg = CPLD_10GMUX_CONFIG_REG;
        I2C_WRITE_BYTE_DATA(ret, &data->access_lock,
                    client, reg, reg_val);
    }
    return count;
}

/* get bmc status register value */
static ssize_t read_bmc_status(struct device *dev,
                    struct device_attribute *da,
                    char *buf)
{
    struct sensor_device_attribute *attr = to_sensor_dev_attr(da);
    struct i2c_client *client = to_i2c_client(dev);
    struct cpld_data *data = i2c_get_clientdata(client);
    u8 reg;
    int reg_val;

    if (attr->index == CPLD_BMC_STATUS) {
        reg = CPLD_BMC_STATUS_REG;
        I2C_READ_BYTE_DATA(reg_val, &data->access_lock, client, reg);
        if (reg_val < 0)
            return -1;
        return sprintf(buf, "0x%02x\n", reg_val);
    }
    return -1;
}

/* get bmc watchdog register value */
static ssize_t read_bmc_watchdog(struct device *dev,
                    struct device_attribute *da,
                    char *buf)
{
    struct sensor_device_attribute *attr = to_sensor_dev_attr(da);
    struct i2c_client *client = to_i2c_client(dev);
    struct cpld_data *data = i2c_get_clientdata(client);
    u8 reg;
    int reg_val;

    if (attr->index == CPLD_BMC_WATCHDOG) {
        reg = CPLD_BMC_WATCHDOG_REG;
        I2C_READ_BYTE_DATA(reg_val, &data->access_lock, client, reg);
        if (reg_val < 0)
            return -1;
        return sprintf(buf, "0x%02x\n", reg_val);
    }
    return -1;
}

/* set value to bmc watchdog register */
static ssize_t write_bmc_watchdog(struct device *dev,
                    struct device_attribute *da,
                    const char *buf,
                    size_t count)
{
    struct sensor_device_attribute *attr = to_sensor_dev_attr(da);
    struct i2c_client *client = to_i2c_client(dev);
    struct cpld_data *data = i2c_get_clientdata(client);
    u8 reg, reg_val;
    int ret;

    if (kstrtou8(buf, 0, &reg_val) < 0)
        return -EINVAL;

    if (attr->index == CPLD_BMC_WATCHDOG) {
        reg = CPLD_BMC_WATCHDOG_REG;
        I2C_WRITE_BYTE_DATA(ret, &data->access_lock,
                    client, reg, reg_val);
    }
    return count;
}

/* get usb status register value */
static ssize_t read_usb_status(struct device *dev,
                struct device_attribute *da,
                char *buf)
{
    struct sensor_device_attribute *attr = to_sensor_dev_attr(da);
    struct i2c_client *client = to_i2c_client(dev);
    struct cpld_data *data = i2c_get_clientdata(client);
    u8 reg;
    int reg_val;

    if (attr->index == CPLD_USB_STATUS) {
        reg = CPLD_USB_STATUS_REG;
        I2C_READ_BYTE_DATA(reg_val, &data->access_lock, client, reg);
        if (reg_val < 0)
            return -1;
        return sprintf(buf, "0x%02x\n", reg_val);
    }
    return -1;
}

/* add valid cpld client to list */
static void ingrasys_i2c_cpld_add_client(struct i2c_client *client)
{
    struct cpld_client_node *node = NULL;

    node = kzalloc(sizeof(struct cpld_client_node), GFP_KERNEL);
    if (!node) {
        dev_info(&client->dev,
            "Can't allocate cpld_client_node for index %d\n",
            client->addr);
        return;
    }

    node->client = client;

    mutex_lock(&list_lock);
    list_add(&node->list, &cpld_client_list);
    mutex_unlock(&list_lock);
}

/* remove exist cpld client in list */
static void ingrasys_i2c_cpld_remove_client(struct i2c_client *client)
{
    struct list_head    *list_node = NULL;
    struct cpld_client_node *cpld_node = NULL;
    int found = 0;

    mutex_lock(&list_lock);
    list_for_each(list_node, &cpld_client_list) {
        cpld_node = list_entry(list_node,
                struct cpld_client_node, list);

        if (cpld_node->client == client) {
            found = 1;
            break;
        }
    }

    if (found) {
        list_del(list_node);
        kfree(cpld_node);
    }
    mutex_unlock(&list_lock);
}

/* cpld drvier probe */
static int ingrasys_i2c_cpld_probe(struct i2c_client *client,
                    const struct i2c_device_id *dev_id)
{
    int status;
    struct cpld_data *data = NULL;
    int ret = -EPERM;
    int err;
    int idx;

    data = kzalloc(sizeof(struct cpld_data), GFP_KERNEL);
    if (!data)
        return -ENOMEM;

    /* init cpld data for client */
    i2c_set_clientdata(client, data);
    mutex_init(&data->access_lock);

    if (!i2c_check_functionality(client->adapter,
                I2C_FUNC_SMBUS_BYTE_DATA)) {
        dev_info(&client->dev,
            "i2c_check_functionality failed (0x%x)\n",
            client->addr);
        status = -EIO;
        goto exit;
    }

    /* get cpld id from device */
    ret = i2c_smbus_read_byte_data(client, CPLD_ID_REG);

    if (ret < 0) {
        dev_info(&client->dev,
            "fail to get cpld id (0x%x) at addr (0x%x)\n",
            CPLD_ID_REG, client->addr);
        status = -EIO;
        goto exit;
    }

    CPLD_ID_ID_GET(ret,  idx);

    if (INVALID(idx, cpld1, cpld5)) {
        dev_info(&client->dev,
            "cpld id %d(device) not valid\n", idx);
        //status = -EPERM;
        //goto exit;
    }

#if 0
    /* change client name for each cpld with index */
    snprintf(client->name, sizeof(client->name), "%s_%d", client->name,
            data->index);
#endif

              data->index = dev_id->driver_data;

    /* register sysfs hooks for different cpld group */
    dev_info(&client->dev, "probe cpld with index %d\n", data->index);
    switch (data->index) {
    case cpld1:
        status = sysfs_create_group(&client->dev.kobj,
                    &s9230_64x_cpld1_group);
        break;
    case cpld2:
        status = sysfs_create_group(&client->dev.kobj,
                    &s9230_64x_cpld2_group);
        break;
    case cpld3:
    case cpld4:
    case cpld5:
        status = sysfs_create_group(&client->dev.kobj,
                    &s9230_64x_cpld345_group);
        break;
    default:
        status = -EINVAL;
    }

    if (status)
        goto exit;

    dev_info(&client->dev, "chip found\n");

    /* add probe chip to client list */
    ingrasys_i2c_cpld_add_client(client);

    return 0;
exit:
    sysfs_remove_group(&client->dev.kobj, &s9230_64x_cpld345_group);
    return status;
}

/* cpld drvier remove */
static int ingrasys_i2c_cpld_remove(struct i2c_client *client)
{
    struct cpld_data *data = i2c_get_clientdata(client);

    switch (data->index) {
    case cpld1:
        sysfs_remove_group(&client->dev.kobj, &s9230_64x_cpld1_group);
        break;
    case cpld2:
        sysfs_remove_group(&client->dev.kobj, &s9230_64x_cpld2_group);
        break;
    case cpld3:
    case cpld4:
    case cpld5:
        sysfs_remove_group(&client->dev.kobj,
                    &s9230_64x_cpld345_group);
        break;
    }

    ingrasys_i2c_cpld_remove_client(client);
    return 0;
}

MODULE_DEVICE_TABLE(i2c, ingrasys_i2c_cpld_id);

static struct i2c_driver ingrasys_i2c_cpld_driver = {
    .class      = I2C_CLASS_HWMON,
    .driver = {
        .name = "ingrasys_i2c_cpld",
    },
    .probe = ingrasys_i2c_cpld_probe,
    .remove = ingrasys_i2c_cpld_remove,
    .id_table = ingrasys_i2c_cpld_id,
    .address_list = cpld_i2c_addr,
};

/* provid cpld register read */
/* cpld_idx indicate the index of cpld device */
int ingrasys_i2c_cpld_read(u8 cpld_idx,
                u8 reg)
{
    struct list_head *list_node = NULL;
    struct cpld_client_node *cpld_node = NULL;
    int ret = -EPERM;
    struct cpld_data *data;

    list_for_each(list_node, &cpld_client_list) {
        cpld_node = list_entry(list_node,
                    struct cpld_client_node, list);
        data = i2c_get_clientdata(cpld_node->client);
        if (data->index == cpld_idx) {
            DEBUG_PRINT("cpld_idx=%d, read reg 0x%02x",
                    cpld_idx, reg);
            I2C_READ_BYTE_DATA(ret, &data->access_lock,
                    cpld_node->client, reg);
            DEBUG_PRINT("cpld_idx=%d, read reg 0x%02x = 0x%02x",
                    cpld_idx, reg, ret);
        break;
        }
    }

    return ret;
}
EXPORT_SYMBOL(ingrasys_i2c_cpld_read);

/* provid cpld register write */
/* cpld_idx indicate the index of cpld device */
int ingrasys_i2c_cpld_write(u8 cpld_idx,
                u8 reg,
                u8 value)
{
    struct list_head *list_node = NULL;
    struct cpld_client_node *cpld_node = NULL;
    int ret = -EIO;
    struct cpld_data *data;

    list_for_each(list_node, &cpld_client_list) {
        cpld_node = list_entry(list_node,
                    struct cpld_client_node, list);
        data = i2c_get_clientdata(cpld_node->client);

        if (data->index == cpld_idx) {
                        I2C_WRITE_BYTE_DATA(ret, &data->access_lock,
                        cpld_node->client,
                        reg, value);
            DEBUG_PRINT("cpld_idx=%d, write reg 0x%02x val 0x%02x, ret=%d",
                            cpld_idx, reg, value, ret);
            break;
        }
    }

    return ret;
}
EXPORT_SYMBOL(ingrasys_i2c_cpld_write);

/* provid qsfp port status register read */
/* port_num indicate the front panel qsfp port number */
int ingrasys_i2c_cpld_get_qsfp_port_status_val(u8 port_num)
{
    u8 cpld_idx, cpld_port, reg;
    int reg_val;

    if (INVALID(port_num, QSFP_MIN_PORT_NUM, QSFP_MAX_PORT_NUM)) {
        DEBUG_PRINT("invalid input value %d", port_num);
        return -1;
    }
    QSFP_TO_CPLD_IDX(port_num, cpld_idx, cpld_port);
    reg = QSFP_PORT_STATUS_REG(cpld_port);
    DEBUG_PRINT("port_num=%d, cpld_idx=%d, cpld_port=%d, reg=0x%x",
                    port_num, cpld_idx, cpld_port, reg);
    reg_val = ingrasys_i2c_cpld_read(cpld_idx, reg);
    return reg_val;
}
EXPORT_SYMBOL(ingrasys_i2c_cpld_get_qsfp_port_status_val);

/* provid qsfp port config register read */
/* port_num indicate the front panel qsfp port number */
int ingrasys_i2c_cpld_get_qsfp_port_config_val(u8 port_num)
{
    u8 cpld_idx, cpld_port, reg;
    int reg_val;

    if (INVALID(port_num, QSFP_MIN_PORT_NUM, QSFP_MAX_PORT_NUM)) {
        DEBUG_PRINT("invalid input value %d", port_num);
        return -1;
    }
    QSFP_TO_CPLD_IDX(port_num, cpld_idx, cpld_port);
    reg = QSFP_PORT_CONFIG_REG(cpld_port);
    DEBUG_PRINT("port_num=%d, cpld_idx=%d, cpld_port=%d, reg=0x%x",
                    port_num, cpld_idx, cpld_port, reg);
    reg_val = ingrasys_i2c_cpld_read(cpld_idx, reg);
    return reg_val;
}
EXPORT_SYMBOL(ingrasys_i2c_cpld_get_qsfp_port_config_val);

/* provid qsfp port config register write */
/* port_num indicate the front panel qsfp port number */
int ingrasys_i2c_cpld_set_qsfp_port_config_val(u8 port_num,
                            u8 reg_val)
{
    u8 cpld_idx, cpld_port, reg, ret;

    if (INVALID(port_num, QSFP_MIN_PORT_NUM, QSFP_MAX_PORT_NUM)) {
        DEBUG_PRINT("invalid input value %d", port_num);
        return -1;
    }
    QSFP_TO_CPLD_IDX(port_num, cpld_idx, cpld_port);
    reg = QSFP_PORT_CONFIG_REG(cpld_port);
    DEBUG_PRINT("port_num=%d, cpld_idx=%d, cpld_port=%d, reg=0x%x",
                    port_num, cpld_idx, cpld_port, reg);
    ret = ingrasys_i2c_cpld_write(cpld_idx, reg, reg_val);
    return ret;
}
EXPORT_SYMBOL(ingrasys_i2c_cpld_set_qsfp_port_config_val);

/* provid sfp port status register read */
/* port_num indicate the front panel qsfp port number */
int ingrasys_i2c_cpld_get_sfp_port_status_val(u8 port_num)
{
    u8 cpld_idx, reg;
    int reg_val;

    if (INVALID(port_num, SFP_MIN_PORT_NUM, SFP_MAX_PORT_NUM)) {
        DEBUG_PRINT("invalid input value %d", port_num);
        return -1;
    }
    SFP_TO_CPLD_IDX(port_num, cpld_idx);
    reg = CPLD_SFP_PORT_STATUS_REG;
    DEBUG_PRINT("port_num=%d, cpld_idx=%d, reg=0x%x",
                    port_num, cpld_idx, reg);
    reg_val = ingrasys_i2c_cpld_read(cpld_idx, reg);
    return reg_val;
}
EXPORT_SYMBOL(ingrasys_i2c_cpld_get_sfp_port_status_val);

/* provid qsfp port config register read */
/* port_num indicate the front panel qsfp port number */
int ingrasys_i2c_cpld_get_sfp_port_config_val(u8 port_num)
{
    u8 cpld_idx, reg;
    int reg_val;

    if (INVALID(port_num, SFP_MIN_PORT_NUM, SFP_MAX_PORT_NUM)) {
        DEBUG_PRINT("invalid input value %d", port_num);
        return -1;
    }
    SFP_TO_CPLD_IDX(port_num, cpld_idx);
    reg = CPLD_SFP_PORT_CONFIG_REG;
    DEBUG_PRINT("port_num=%d, cpld_idx=%d, reg=0x%x",
                port_num, cpld_idx, reg);
    reg_val = ingrasys_i2c_cpld_read(cpld_idx, reg);
    return reg_val;
}
EXPORT_SYMBOL(ingrasys_i2c_cpld_get_sfp_port_config_val);

/* provid qsfp port config register write */
/* port_num indicate the front panel qsfp port number */
int ingrasys_i2c_cpld_set_sfp_port_config_val(u8 port_num,
                            u8 reg_val)
{
    u8 cpld_idx, reg, ret;

    if (INVALID(port_num, SFP_MIN_PORT_NUM, SFP_MAX_PORT_NUM)) {
        DEBUG_PRINT("invalid input value %d", port_num);
        return -1;
    }
    SFP_TO_CPLD_IDX(port_num, cpld_idx);
    reg = CPLD_SFP_PORT_CONFIG_REG;
    DEBUG_PRINT("port_num=%d, cpld_idx=%d, reg=0x%x",
                port_num, cpld_idx, reg);
    ret = ingrasys_i2c_cpld_write(cpld_idx, reg, reg_val);
    return ret;
}
EXPORT_SYMBOL(ingrasys_i2c_cpld_set_sfp_port_config_val);

static int __init ingrasys_i2c_cpld_init(void)
{
    mutex_init(&list_lock);
    return i2c_add_driver(&ingrasys_i2c_cpld_driver);
}

static void __exit ingrasys_i2c_cpld_exit(void)
{
    i2c_del_driver(&ingrasys_i2c_cpld_driver);
}

MODULE_AUTHOR("Leo Lin <feng.lee.usa@ingrasys.com>");
MODULE_DESCRIPTION("ingrasys_i2c_cpld driver");
MODULE_LICENSE("GPL");

module_init(ingrasys_i2c_cpld_init);
module_exit(ingrasys_i2c_cpld_exit);

