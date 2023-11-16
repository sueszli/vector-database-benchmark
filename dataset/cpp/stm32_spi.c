/***************************************************************************//**
 *   @file   stm32/stm32_spi.c
 *   @brief  Implementation of stm32 spi driver.
 *   @author Darius Berghe (darius.berghe@analog.com)
********************************************************************************
 * Copyright 2020(c) Analog Devices, Inc.
 *
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *  - Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  - Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in
 *    the documentation and/or other materials provided with the
 *    distribution.
 *  - Neither the name of Analog Devices, Inc. nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *  - The use of this software may or may not infringe the patent rights
 *    of one or more patent holders.  This license does not release you
 *    from the requirement that you obtain separate licenses from these
 *    patent holders to use this software.
 *  - Use of the software either in source or binary form, must be run
 *    on or directly connected to an Analog Devices Inc. component.
 *
 * THIS SOFTWARE IS PROVIDED BY ANALOG DEVICES "AS IS" AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, NON-INFRINGEMENT,
 * MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 * IN NO EVENT SHALL ANALOG DEVICES BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, INTELLECTUAL PROPERTY RIGHTS, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*******************************************************************************/
#include <stdlib.h>
#include <errno.h>
#include "no_os_util.h"
#include "no_os_gpio.h"
#include "stm32_gpio.h"
#include "no_os_spi.h"
#include "stm32_spi.h"
#include "no_os_delay.h"
#include "no_os_alloc.h"

static int stm32_spi_config(struct no_os_spi_desc *desc)
{
	int ret;
	uint32_t prescaler = 0u;
	SPI_TypeDef *base = NULL;
	struct stm32_spi_desc *sdesc = desc->extra;

	/* automatically select prescaler based on max_speed_hz */
	if (desc->max_speed_hz != 0u) {
		uint32_t div = sdesc->input_clock / desc->max_speed_hz;

		switch (div) {
		case 0 ... 2:
			prescaler = SPI_BAUDRATEPRESCALER_2;
			break;
		case 3 ... 4:
			prescaler = SPI_BAUDRATEPRESCALER_4;
			break;
		case 5 ... 8:
			prescaler = SPI_BAUDRATEPRESCALER_8;
			break;
		case 9 ... 16:
			prescaler = SPI_BAUDRATEPRESCALER_16;
			break;
		case 17 ... 32:
			prescaler = SPI_BAUDRATEPRESCALER_32;
			break;
		case 33 ... 64:
			prescaler = SPI_BAUDRATEPRESCALER_64;
			break;
		case 65 ... 128:
			prescaler = SPI_BAUDRATEPRESCALER_128;
			break;
		default:
			prescaler = SPI_BAUDRATEPRESCALER_256;
			break;
		}
	} else
		prescaler = SPI_BAUDRATEPRESCALER_64;

	switch (desc->device_id) {
#if defined(SPI1)
	case 1:
		base = SPI1;
		break;
#endif
#if defined(SPI2)
	case 2:
		base = SPI2;
		break;
#endif
#if defined(SPI3)
	case 3:
		base = SPI3;
		break;
#endif
		break;
#if defined(SPI4)
	case 4:
		base = SPI4;
		break;
#endif
#if defined(SPI5)
	case 5:
		base = SPI5;
		break;
#endif
#if defined(SPI6)
	case 6:
		base = SPI6;
		break;
#endif
	default:
		ret = -EINVAL;
		goto error;
	};
	sdesc->hspi.Instance = base;
	sdesc->hspi.Init.Mode = SPI_MODE_MASTER;
	sdesc->hspi.Init.Direction = SPI_DIRECTION_2LINES;
	sdesc->hspi.Init.DataSize = SPI_DATASIZE_8BIT;
	sdesc->hspi.Init.CLKPolarity = desc->mode & NO_OS_SPI_CPOL ?
				       SPI_POLARITY_HIGH :
				       SPI_POLARITY_LOW;
	sdesc->hspi.Init.CLKPhase = desc->mode & NO_OS_SPI_CPHA ? SPI_PHASE_2EDGE :
				    SPI_PHASE_1EDGE;
	sdesc->hspi.Init.NSS = SPI_NSS_SOFT;
	sdesc->hspi.Init.BaudRatePrescaler = prescaler;
	sdesc->hspi.Init.FirstBit = desc->bit_order ? SPI_FIRSTBIT_LSB :
				    SPI_FIRSTBIT_MSB;
	sdesc->hspi.Init.TIMode = SPI_TIMODE_DISABLE;
	sdesc->hspi.Init.CRCCalculation = SPI_CRCCALCULATION_DISABLE;
	sdesc->hspi.Init.CRCPolynomial = 10;
	ret = HAL_SPI_Init(&sdesc->hspi);
	if (ret != HAL_OK) {
		ret = -EIO;
		goto error;
	}

	return 0;
error:
	return ret;
}

/**
 * @brief Initialize the SPI communication peripheral.
 * @param desc - The SPI descriptor.
 * @param param - The structure that contains the SPI parameters.
 * @return 0 in case of success, -1 otherwise.
 */
int32_t stm32_spi_init(struct no_os_spi_desc **desc,
		       const struct no_os_spi_init_param *param)
{
	int32_t ret;
	struct no_os_spi_desc	*spi_desc;

	if (!desc || !param)
		return -EINVAL;

	spi_desc = (struct no_os_spi_desc *)no_os_calloc(1, sizeof(*spi_desc));
	if (!spi_desc)
		return -ENOMEM;

	struct stm32_spi_desc *sdesc;
	struct stm32_spi_init_param *sinit;
	struct no_os_gpio_init_param csip;
	struct stm32_gpio_init_param csip_extra;

	sdesc = (struct stm32_spi_desc*)no_os_calloc(1,sizeof(struct stm32_spi_desc));
	if (!sdesc) {
		ret = -ENOMEM;
		goto error;
	}

	spi_desc->extra = sdesc;
	sinit = param->extra;

	csip_extra.mode = GPIO_MODE_OUTPUT_PP;
	csip_extra.speed = GPIO_SPEED_FREQ_LOW;
	csip.port = sinit->chip_select_port;
	csip.number = param->chip_select;
	csip.pull = NO_OS_PULL_NONE;
	csip.extra = &csip_extra;
	csip.platform_ops = &stm32_gpio_ops;
	ret = no_os_gpio_get(&sdesc->chip_select, &csip);
	if (ret)
		goto error;

	ret = no_os_gpio_direction_output(sdesc->chip_select, NO_OS_GPIO_HIGH);
	if (ret)
		goto error;

	/* copy settings to device descriptor */
	spi_desc->device_id = param->device_id;
	spi_desc->max_speed_hz = param->max_speed_hz;
	spi_desc->mode = param->mode;
	spi_desc->bit_order = param->bit_order;
	spi_desc->chip_select = param->chip_select;
	if (sinit->get_input_clock)
		sdesc->input_clock = sinit->get_input_clock();

	ret = stm32_spi_config(spi_desc);
	if (ret)
		goto error;

	*desc = spi_desc;

	return 0;
error:
	no_os_free(spi_desc);
	no_os_free(sdesc);
	return ret;
}

/**
 * @brief Free the resources allocated by no_os_spi_init().
 * @param desc - The SPI descriptor.
 * @return 0 in case of success, -1 otherwise.
 */
int32_t stm32_spi_remove(struct no_os_spi_desc *desc)
{
	struct stm32_spi_desc *sdesc;

	if (!desc || !desc->extra)
		return -EINVAL;

	sdesc = desc->extra;
	HAL_SPI_DeInit(&sdesc->hspi);
	no_os_gpio_remove(sdesc->chip_select);
	no_os_free(desc->extra);
	no_os_free(desc);
	return 0;
}

/**
 * @brief Write/read multiple messages to/from SPI.
 * @param desc - The SPI descriptor.
 * @param msgs - The messages array.
 * @param len - Number of messages.
 * @return 0 in case of success, errno codes otherwise.
 */
int32_t stm32_spi_transfer(struct no_os_spi_desc *desc,
			   struct no_os_spi_msg *msgs,
			   uint32_t len)
{
	struct stm32_spi_desc *sdesc;
	struct stm32_gpio_desc *gdesc;
	uint64_t slave_id;
	static uint64_t last_slave_id;
	int ret = 0;

	if (!desc || !desc->extra || !msgs)
		return -EINVAL;

	sdesc = desc->extra;
	gdesc = sdesc->chip_select->extra;
#ifdef SPI_SR_TXE
	uint32_t tx_cnt = 0;
	uint32_t rx_cnt = 0;
	SPI_TypeDef * SPIx = sdesc->hspi.Instance;
#endif


	// Compute a slave ID based on SPI instance and chip select.
	// If it did not change since last call to stm32_spi_write_and_read,
	// no need to reconfigure SPI. Otherwise, reconfigure it.
	slave_id = ((uint64_t)(uintptr_t)sdesc->hspi.Instance << 32) |
		   sdesc->chip_select->number;
	if (slave_id != last_slave_id) {
		last_slave_id = slave_id;
		ret = stm32_spi_config(desc);
		if (ret)
			return ret;
	}

	for (uint32_t i = 0; i < len; i++) {

		if (!msgs[i].tx_buff && !msgs[i].rx_buff)
			return -EINVAL;

		/* Assert CS */
		gdesc->port->BSRR = NO_OS_BIT(sdesc->chip_select->number) << 16;

		if(msgs[i].cs_delay_first)
			no_os_udelay(msgs[i].cs_delay_first);

#ifndef SPI_SR_TXE
		/* Some STM32 families have different naming for
		   SPI_SR_TXE, SPI_SR_RXNE and SPIx->DR. In that case, simply
		   use the HAL API for SPI transmission, which is generic
		   for all STM32 families. */

		if (msgs[i].tx_buff && msgs[i].rx_buff)
			ret = HAL_SPI_TransmitReceive(&sdesc->hspi, msgs[i].tx_buff, msgs[i].rx_buff,
						      msgs[i].bytes_number, HAL_MAX_DELAY);

		else if (msgs[i].tx_buff)
			ret = HAL_SPI_Transmit(&sdesc->hspi, msgs[i].tx_buff, msgs[i].bytes_number,
					       HAL_MAX_DELAY);
		else
			ret = HAL_SPI_Receive(&sdesc->hspi, msgs[i].rx_buff, msgs[i].bytes_number,
					      HAL_MAX_DELAY);

		if (ret != HAL_OK) {
			if (ret == HAL_TIMEOUT)
				ret = -ETIMEDOUT;
			else
				ret = -EIO;
		}
#else
		rx_cnt = 0;
		tx_cnt = 0;
		__HAL_SPI_ENABLE(&sdesc->hspi);

		while ((msgs[i].rx_buff && rx_cnt < msgs[i].bytes_number) ||
		       (msgs[i].tx_buff && tx_cnt < msgs[i].bytes_number)) {
			while(!(SPIx->SR & SPI_SR_TXE))
				;
			if (msgs[i].tx_buff)
				*(volatile uint8_t *)&SPIx->DR = msgs[i].tx_buff[tx_cnt++];
			else
				*(volatile uint8_t *)&SPIx->DR = 0;

			while(!(SPIx->SR & SPI_SR_RXNE))
				;
			if (msgs[i].rx_buff)
				msgs[i].rx_buff[rx_cnt++] = *(volatile uint8_t *)&SPIx->DR;
			else
				(void)*(volatile uint8_t *)&SPIx->DR;
		}

		__HAL_SPI_DISABLE(&sdesc->hspi);
#endif

		if(msgs[i].cs_delay_last)
			no_os_udelay(msgs[i].cs_delay_last);

		if (msgs[i].cs_change)
			/* De-assert CS */
			gdesc->port->BSRR = NO_OS_BIT(sdesc->chip_select->number);

		if(msgs[i].cs_change_delay)
			no_os_udelay(msgs[i].cs_change_delay);

		if (ret)
			return ret;
	}

	return 0;
}

/**
 * @brief Write and read data to/from SPI.
 * @param desc - The SPI descriptor.
 * @param data - The buffer with the transmitted/received data.
 * @param bytes_number - Number of bytes to write/read.
 * @return 0 in case of success, -1 otherwise.
 */
int32_t stm32_spi_write_and_read(struct no_os_spi_desc *desc,
				 uint8_t *data,
				 uint16_t bytes_number)
{
	struct no_os_spi_msg msg = {
		.bytes_number = bytes_number,
		.cs_change = true,
		.rx_buff = data,
		.tx_buff = data,
	};

	if (!desc || !desc->extra || !data)
		return -EINVAL;

	if (!bytes_number)
		return 0;

	return stm32_spi_transfer(desc, &msg, 1);
}

/**
 * @brief stm32 platform specific SPI platform ops structure
 */
const struct no_os_spi_platform_ops stm32_spi_ops = {
	.init = &stm32_spi_init,
	.write_and_read = &stm32_spi_write_and_read,
	.remove = &stm32_spi_remove,
	.transfer = &stm32_spi_transfer
};
