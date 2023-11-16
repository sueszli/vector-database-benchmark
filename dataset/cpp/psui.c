/************************************************************
 * <bsn.cl fy=2014 v=onl>
 *
 *        Copyright 2014, 2015 Big Switch Networks, Inc.
 *
 * Licensed under the Eclipse Public License, Version 1.0 (the
 * "License"); you may not use this file except in compliance
 * with the License. You may obtain a copy of the License at
 *
 *        http://www.eclipse.org/legal/epl-v10.html
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
 * either express or implied. See the License for the specific
 * language governing permissions and limitations under the
 * License.
 *
 * </bsn.cl>
 ************************************************************
 *
 *
 *
 ***********************************************************/
#include <onlp/platformi/psui.h>
#include <onlplib/file.h>

#include "platform_lib.h"

static onlp_psu_info_t pinfo[] =
{
    { }, /* Not used */
    {
        {
            PSU_OID_PSU1,
            "PSU-1",
            0,
            {
                FAN_OID_PSU_FAN1,
            },
        }
    },
    {
        {
            PSU_OID_PSU2,
            "PSU-2",
            0,
            {
                FAN_OID_PSU_FAN2,
            },
        }
    }
};

int
onlp_psui_init(void)
{    
    return ONLP_STATUS_OK;
}

int 
psu_status_info_get(int id, onlp_psu_info_t *info)
{   
    int pw_exist, pw_good;
    int rc, i2c_bus;
    char *sys_psu_prefix = NULL;

    if (id == PSU_ID_PSU1) {
        i2c_bus = I2C_BUS_PSU1;
        sys_psu_prefix = SYS_PSU1_PREFIX;
    } else if (id == PSU_ID_PSU2) {
        i2c_bus = I2C_BUS_PSU2;
        sys_psu_prefix = SYS_PSU2_PREFIX;
    } else {
        return ONLP_STATUS_E_INTERNAL;
    }
    
     /* Get power present status */
    if ((rc = psu_present_get(&pw_exist, sys_psu_prefix))
            != ONLP_STATUS_OK) {
        return ONLP_STATUS_E_INTERNAL;
    }
    
    if (pw_exist != PSU_STATUS_PRESENT) {
        info->status &= ~ONLP_PSU_STATUS_PRESENT;
        info->status |=  ONLP_PSU_STATUS_FAILED;
        return ONLP_STATUS_OK;
    }    
    info->status |= ONLP_PSU_STATUS_PRESENT;
    
    /* Get power good status */
    if ((rc = psu_pwgood_get(&pw_good, sys_psu_prefix)) 
            != ONLP_STATUS_OK) {
        return ONLP_STATUS_E_INTERNAL;
    }   
    
    if (pw_good != PSU_STATUS_POWER_GOOD) {        
        info->status |= ONLP_PSU_STATUS_UNPLUGGED;
        return ONLP_STATUS_OK;
    } else {
        info->status &= ~ONLP_PSU_STATUS_UNPLUGGED;
    }
    
    /* Get power eeprom status */
    if ((rc = psu_eeprom_get(info, id)) != ONLP_STATUS_OK) {
        return ONLP_STATUS_E_INTERNAL;
    }
    
    /* Get power iout status */
    if ((rc = psu_iout_get(info, i2c_bus)) != ONLP_STATUS_OK) {
        return ONLP_STATUS_E_INTERNAL;
    }
    
    /* Get power pout status */
    if ((rc = psu_pout_get(info, i2c_bus)) != ONLP_STATUS_OK) {
        return ONLP_STATUS_E_INTERNAL;
    }
    
    /* Get power pin status */
    if ((rc = psu_pin_get(info, i2c_bus)) != ONLP_STATUS_OK) {
        return ONLP_STATUS_E_INTERNAL;
    }

    /* Get power vout status */
    if ((rc = psu_vout_get(info, i2c_bus)) != ONLP_STATUS_OK) {
        return ONLP_STATUS_E_INTERNAL;
    }
    
    return ONLP_STATUS_OK;
}

int
onlp_psui_info_get(onlp_oid_t id, onlp_psu_info_t* info)
{        
    int pid;
    
    pid = ONLP_OID_ID_GET(id);
    memset(info, 0, sizeof(onlp_psu_info_t));
    
    /* Set the onlp_oid_hdr_t */
    *info = pinfo[pid]; 

    switch (pid) {
        case PSU_ID_PSU1:
        case PSU_ID_PSU2:
            return psu_status_info_get(pid, info);
            break;
        default:
            return ONLP_STATUS_E_UNSUPPORTED;
            break;
    }

    return ONLP_STATUS_OK;


}
