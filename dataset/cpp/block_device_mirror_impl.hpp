// coding: utf-8
/*
 * Copyright (c) 2018, Raphael Lehmann
 *
 * This file is part of the modm project.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
// ----------------------------------------------------------------------------

#ifndef MODM_BLOCK_DEVICE_MIRROR_HPP
	#error	"Don't include this file directly, use 'block_device_mirror.hpp' instead!"
#endif
#include "block_device_mirror.hpp"


// ----------------------------------------------------------------------------
template <typename BlockDeviceA, typename BlockDeviceB>
modm::ResumableResult<bool>
modm::BdMirror<BlockDeviceA, BlockDeviceB>::initialize()
{
	RF_BEGIN();

	resultA = RF_CALL(blockDeviceA.initialize());
	resultB = RF_CALL(blockDeviceB.initialize());

	RF_END_RETURN(resultA && resultA);
}

// ----------------------------------------------------------------------------
template <typename BlockDeviceA, typename BlockDeviceB>
modm::ResumableResult<bool>
modm::BdMirror<BlockDeviceA, BlockDeviceB>::deinitialize()
{
	RF_BEGIN();

	resultA = RF_CALL(blockDeviceA.deinitialize());
	resultB = RF_CALL(blockDeviceB.deinitialize());

	RF_END_RETURN(resultA && resultA);
}

// ----------------------------------------------------------------------------
template <typename BlockDeviceA, typename BlockDeviceB>
modm::ResumableResult<bool>
modm::BdMirror<BlockDeviceA, BlockDeviceB>::read(uint8_t* buffer, bd_address_t address, bd_size_t size)
{
	return blockDeviceA.read(buffer, address, size);
}

// ----------------------------------------------------------------------------
template <typename BlockDeviceA, typename BlockDeviceB>
modm::ResumableResult<bool>
modm::BdMirror<BlockDeviceA, BlockDeviceB>::program(const uint8_t* buffer, bd_address_t address, bd_size_t size)
{
	RF_BEGIN();

	if((size == 0) || (size % BlockSizeWrite != 0)) {
		RF_RETURN(false);
	}

	resultA = RF_CALL(blockDeviceA.program(buffer, address, size));
	resultB = RF_CALL(blockDeviceB.program(buffer, address, size));

	RF_END_RETURN(resultA && resultA);
}


// ----------------------------------------------------------------------------
template <typename BlockDeviceA, typename BlockDeviceB>
modm::ResumableResult<bool>
modm::BdMirror<BlockDeviceA, BlockDeviceB>::erase(bd_address_t address, bd_size_t size)
{
	RF_BEGIN();

	if((size == 0) || (size % BlockSizeErase != 0)) {
		RF_RETURN(false);
	}

	resultA = RF_CALL(blockDeviceA.erase(address, size));
	resultB = RF_CALL(blockDeviceB.erase(address, size));

	RF_END_RETURN(resultA && resultA);
}


// ----------------------------------------------------------------------------
template <typename BlockDeviceA, typename BlockDeviceB>
modm::ResumableResult<bool>
modm::BdMirror<BlockDeviceA, BlockDeviceB>::write(const uint8_t* buffer, bd_address_t address, bd_size_t size)
{
	RF_BEGIN();

	if((size == 0) || (size % BlockSizeErase != 0) || (size % BlockSizeWrite != 0)) {
		RF_RETURN(false);
	}

	resultA = RF_CALL(blockDeviceA.write(buffer, address, size));
	resultB = RF_CALL(blockDeviceB.write(buffer, address, size));

	RF_END_RETURN(resultA && resultA);
}
