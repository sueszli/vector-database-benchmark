// coding: utf-8
/*
 * Copyright (c) 2018, Raphael Lehmann
 * Copyright (c) 2023, Rasmus Kleist Hørlyck Sørensen
 *
 * This file is part of the modm project.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
// ----------------------------------------------------------------------------

#ifndef MODM_BLOCK_DEVICE_SPIFLASH_HPP
#error	"Don't include this file directly, use 'block_device_spiflash.hpp' instead!"
#endif
#include "block_device_spiflash.hpp"

// ----------------------------------------------------------------------------
template <typename Spi, typename Cs, uint32_t flashSize>
modm::ResumableResult<bool>
modm::BdSpiFlash<Spi, Cs, flashSize>::initialize()
{
	RF_BEGIN();
	this->attachConfigurationHandler([]() {
		Spi::setDataMode(Spi::DataMode::Mode0);
		Spi::setDataOrder(Spi::DataOrder::MsbFirst);
	});
	Cs::setOutput(modm::Gpio::High);

	RF_CALL(spiOperation(Instruction::RstEn));
	// Wait T_CPH = 25ns
	RF_CALL(spiOperation(Instruction::Rst));
	RF_CALL(waitWhileBusy());

	RF_CALL(spiOperation(Instruction::WE));
	RF_CALL(spiOperation(Instruction::GBU));
	RF_CALL(waitWhileBusy());

	// Enter 4-Byte Address mode for serial flash memory that support 256M-bit or more
	if (DeviceSize > ExtendedAddressThreshold) {
		RF_CALL(spiOperation(Instruction::En4BAM));
		RF_CALL(waitWhileBusy());
	}

	RF_END_RETURN(true);
}

// ----------------------------------------------------------------------------
template <typename Spi, typename Cs, uint32_t flashSize>
modm::ResumableResult<bool>
modm::BdSpiFlash<Spi, Cs, flashSize>::deinitialize()
{
	RF_BEGIN();
	// nothing
	RF_END_RETURN(true);
}

// ----------------------------------------------------------------------------
template <typename Spi, typename Cs, uint32_t flashSize>
modm::ResumableResult<typename modm::BdSpiFlash<Spi, Cs, flashSize>::JedecId>
modm::BdSpiFlash<Spi, Cs, flashSize>::readId()
{
	RF_BEGIN();

	RF_CALL(spiOperation(Instruction::RJI, nullptr, resultBuffer, 3));

	RF_END_RETURN(JedecId(resultBuffer[0], resultBuffer[1], resultBuffer[2]));
}

// ----------------------------------------------------------------------------
template <typename Spi, typename Cs, uint32_t flashSize>
modm::ResumableResult<bool>
modm::BdSpiFlash<Spi, Cs, flashSize>::read(uint8_t* buffer, bd_address_t address, bd_size_t size)
{
	RF_BEGIN();

	if((size == 0) || (size % BlockSizeRead != 0) || (address + size > flashSize)) {
		RF_RETURN(false);
	}

	RF_CALL(waitWhileBusy());
	RF_CALL(spiOperation(Instruction::FR, address, nullptr, buffer, size, 1));

	RF_END_RETURN(true);
}

// ----------------------------------------------------------------------------
template <typename Spi, typename Cs, uint32_t flashSize>
modm::ResumableResult<bool>
modm::BdSpiFlash<Spi, Cs, flashSize>::program(const uint8_t* buffer, bd_address_t address, bd_size_t size)
{
	RF_BEGIN();

	if((size == 0) || (size % BlockSizeWrite != 0) || (address + size > flashSize)) {
		RF_RETURN(false);
	}

	index = 0;
	while(index < size) {
		RF_CALL(waitWhileBusy());
		RF_CALL(spiOperation(Instruction::WE));
		RF_CALL(spiOperation(Instruction::PP, address + index, &buffer[index], nullptr, BlockSizeWrite));
		index += BlockSizeWrite;
	}

	RF_END_RETURN(true);
}


// ----------------------------------------------------------------------------
template <typename Spi, typename Cs, uint32_t flashSize>
modm::ResumableResult<bool>
modm::BdSpiFlash<Spi, Cs, flashSize>::erase(bd_address_t address, bd_size_t size)
{
	RF_BEGIN();

	if((size == 0) || (size % BlockSizeErase != 0) || (address + size > flashSize)) {
		RF_RETURN(false);
	}

	if (address == 0 && size == flashSize) {
		RF_CALL(waitWhileBusy());
		RF_CALL(spiOperation(Instruction::CE));
	} else {
		index = 0;
		while(index < size) {
			RF_CALL(waitWhileBusy());
			RF_CALL(spiOperation(Instruction::WE));
			RF_CALL(spiOperation(Instruction::SE, address + index));
			index += BlockSizeErase;
		}
	}

	RF_END_RETURN(true);
}


// ----------------------------------------------------------------------------
template <typename Spi, typename Cs, uint32_t flashSize>
modm::ResumableResult<bool>
modm::BdSpiFlash<Spi, Cs, flashSize>::write(const uint8_t* buffer, bd_address_t address, bd_size_t size)
{
	RF_BEGIN();

	if((size == 0) || (size % BlockSizeErase != 0) || (size % BlockSizeWrite != 0) || (address + size > flashSize)) {
		RF_RETURN(false);
	}

	if(!RF_CALL(this->erase(address, size))) {
		RF_RETURN(false);
	}

	if(!RF_CALL(this->program(buffer, address, size))) {
		RF_RETURN(false);
	}

	RF_END_RETURN(true);
}

// ============================================================================

template <typename Spi, typename Cs, uint32_t flashSize>
modm::ResumableResult<typename modm::BdSpiFlash<Spi, Cs, flashSize>::StatusRegister>
modm::BdSpiFlash<Spi, Cs, flashSize>::readStatus()
{
	RF_BEGIN();
	RF_CALL(spiOperation(Instruction::RSR1, nullptr, resultBuffer, 1));
	RF_END_RETURN(static_cast<StatusRegister>(resultBuffer[0]));
}

template <typename Spi, typename Cs, uint32_t flashSize>
modm::ResumableResult<void>
modm::BdSpiFlash<Spi, Cs, flashSize>::selectDie(uint8_t die)
{
	RF_BEGIN();

	RF_CALL(spiOperation(Instruction::SDS, &die, nullptr, 1));
	RF_CALL(waitWhileBusy());

	RF_CALL(spiOperation(Instruction::WE));
	RF_CALL(spiOperation(Instruction::GBU));
	RF_CALL(waitWhileBusy());

	// Enter 4-Byte Address mode for serial flash memory that support 256M-bit or more
	if (DeviceSize > ExtendedAddressThreshold) {
		RF_CALL(spiOperation(Instruction::En4BAM));
		RF_CALL(waitWhileBusy());
	}

	RF_END();
}

template <typename Spi, typename Cs, uint32_t flashSize>
modm::ResumableResult<bool>
modm::BdSpiFlash<Spi, Cs, flashSize>::isBusy()
{
	RF_BEGIN();

	if(RF_CALL(readStatus()) & StatusRegister::Busy) {
		RF_RETURN(true);
	}

	RF_END_RETURN(false);
}


template <typename Spi, typename Cs, uint32_t flashSize>
modm::ResumableResult<void>
modm::BdSpiFlash<Spi, Cs, flashSize>::waitWhileBusy()
{
	RF_BEGIN();
	while(RF_CALL(isBusy())) {
		RF_YIELD();
	}
	RF_END();
}

template <typename Spi, typename Cs, uint32_t flashSize>
modm::ResumableResult<void>
modm::BdSpiFlash<Spi, Cs, flashSize>::spiOperation(Instruction instruction, const uint8_t* tx, uint8_t* rx, std::size_t length, uint8_t nrDummyCycles)
{
	RF_BEGIN();

	i = 0;
	instructionBuffer[i++] = static_cast<uint8_t>(instruction);
	for(uint8_t j = 0; j < nrDummyCycles; j++) {
		instructionBuffer[i++] = 0x00;
	}

	RF_WAIT_UNTIL(this->acquireMaster());
	Cs::reset();

	RF_CALL(Spi::transfer(instructionBuffer, nullptr, i));

	if(length > 0) {
		RF_CALL(Spi::transfer(const_cast<uint8_t*>(tx), rx, length));
	}

	if (this->releaseMaster()) {
		Cs::set();
	}

	RF_END();
}

template <typename Spi, typename Cs, uint32_t flashSize>
modm::ResumableResult<void>
modm::BdSpiFlash<Spi, Cs, flashSize>::spiOperation(Instruction instruction, uint32_t address, const uint8_t* tx, uint8_t* rx, std::size_t length, uint8_t nrDummyCycles)
{
	RF_BEGIN();

	i = 0;
	instructionBuffer[i++] = static_cast<uint8_t>(instruction);
	if constexpr (DeviceSize > ExtendedAddressThreshold) {
		instructionBuffer[i++] = (address >> 24) & 0xFF;
		instructionBuffer[i++] = (address >> 16) & 0xFF;
		instructionBuffer[i++] = (address >> 8) & 0xFF;
		instructionBuffer[i++] = address & 0xFF;
	} else {
		instructionBuffer[i++] = (address >> 16) & 0xFF;
		instructionBuffer[i++] = (address >> 8) & 0xFF;
		instructionBuffer[i++] = address & 0xFF;
	}
	for(uint8_t j = 0; j < nrDummyCycles; j++) {
		instructionBuffer[i++] = 0x00;
	}

	RF_WAIT_UNTIL(this->acquireMaster());
	Cs::reset();

	RF_CALL(Spi::transfer(instructionBuffer, nullptr, i));

	if(length > 0) {
		RF_CALL(Spi::transfer(const_cast<uint8_t*>(tx), rx, length));
	}

	if (this->releaseMaster()) {
		Cs::set();
	}

	RF_END();
}
