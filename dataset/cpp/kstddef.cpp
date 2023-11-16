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

#include <kernel/kstd/kstddef.h>
#include <kernel/kstd/kstdio.h>
#include <kernel/memory/kliballoc.h>
#include <kernel/kstd/cstring.h>

void *operator new(size_t size) {
	return kmalloc(size);
}

void *operator new(size_t size, void* ptr) {
	return ptr;
}

void *operator new[](size_t size) {
	return kmalloc(size);
}

void *operator new[](size_t size, void* ptr) {
	return ptr;
}

void operator delete(void *p) {
	kfree(p);
}

void operator delete(void *p, size_t size) {
	kfree(p);
}

void operator delete[](void *p) {
	kfree(p);
}

void operator delete[](void *p, size_t size) {
	kfree(p);
}

extern "C" void __cxa_pure_virtual() {
	// Do nothing or print an error message.
}
