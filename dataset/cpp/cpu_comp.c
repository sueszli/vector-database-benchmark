/************************************************************************************//**
* \file         Source/ARMCM33_STM32L5/IAR/cpu_comp.c
* \brief        Bootloader cpu module source file.
* \ingroup      Target_ARMCM33_STM32L5
* \internal
*----------------------------------------------------------------------------------------
*                          C O P Y R I G H T
*----------------------------------------------------------------------------------------
*   Copyright (c) 2021  by Feaser    http://www.feaser.com    All rights reserved
*
*----------------------------------------------------------------------------------------
*                            L I C E N S E
*----------------------------------------------------------------------------------------
* This file is part of OpenBLT. OpenBLT is free software: you can redistribute it and/or
* modify it under the terms of the GNU General Public License as published by the Free
* Software Foundation, either version 3 of the License, or (at your option) any later
* version.
*
* OpenBLT is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
* without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
* PURPOSE. See the GNU General Public License for more details.
*
* You have received a copy of the GNU General Public License along with OpenBLT. It
* should be located in ".\Doc\license.html". If not, contact Feaser to obtain a copy.
*
* \endinternal
****************************************************************************************/

/****************************************************************************************
* Include files
****************************************************************************************/
#include "boot.h"                                /* bootloader generic header          */
#include <intrinsics.h>                          /* IAR intrinsics                     */


/************************************************************************************//**
** \brief     Disable global interrupts.
** \return    none.
**
****************************************************************************************/
void CpuIrqDisable(void)
{
  __disable_interrupt();
} /*** end of CpuIrqDisable ***/


/************************************************************************************//**
** \brief     Enable global interrupts.
** \return    none.
**
****************************************************************************************/
void CpuIrqEnable(void)
{
  __enable_interrupt();
} /*** end of CpuIrqEnable ***/

/*********************************** end of cpu_comp.c *********************************/
