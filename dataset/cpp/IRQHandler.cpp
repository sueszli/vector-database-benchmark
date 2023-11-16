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

#include <kernel/kstd/kstdio.h>
#include "IRQHandler.h"
#include "irq.h"

IRQHandler::IRQHandler() {

}

void IRQHandler::handle(Registers* regs) {
	_sent_eoi = false;
	handle_irq(regs);
}

bool IRQHandler::sent_eoi() {
	return _sent_eoi;
}

IRQHandler::IRQHandler(int irq): _irq(irq) {
	Interrupt::irq_set_handler(irq, this);
}

void IRQHandler::set_irq(int irq) {
	_irq = irq;
}

void IRQHandler::uninstall_irq() {
	Interrupt::irq_set_handler(_irq, nullptr);
}

void IRQHandler::reinstall_irq() {
	if(!_irq) return;
	Interrupt::irq_set_handler(_irq, this);
}

bool IRQHandler::mark_in_irq() {
	return true;
}

void IRQHandler::send_eoi() {
	Interrupt::send_eoi(_irq);
	_sent_eoi = true;
}
