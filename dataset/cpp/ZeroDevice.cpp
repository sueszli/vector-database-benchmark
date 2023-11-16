/*
	This file is part of duckOS.

	duckOS is free software: you can redistribute it and/or modify
	it under the terms of the GNU General Public License as published by
	the Free Software Foundation, either version 3 of the License, or
	(at your option) any later version.

	duckOS is distributed in the hope that it will be useful,
	but WITHOUT ANY WARRANTY; without even the implied warranty of
	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
	GNU General Public License for more details.

	You should have received a copy of the GNU General Public License
	along with duckOS.  If not, see <https://www.gnu.org/licenses/>.

	Copyright (c) Byteduck 2016-2021. All rights reserved.
*/

#include <kernel/kstd/kstdlib.h>
#include "ZeroDevice.h"
#include <kernel/kstd/cstring.h>

ZeroDevice::ZeroDevice(): CharacterDevice(1, 5) {

}

ssize_t ZeroDevice::read(FileDescriptor &fd, size_t offset, SafePointer<uint8_t> buffer, size_t count) {
	size_t read = min(count, 512);
	buffer.memset(0, 0, read);
	return read;
}

ssize_t ZeroDevice::write(FileDescriptor &fd, size_t offset, SafePointer<uint8_t> buffer, size_t count) {
	return count;
}
