/*-
 * SPDX-License-Identifier: MIT
 *
 * MIT License
 *
 * Copyright (c) 2020 Abb1x
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include "PIT.h"
#include <libk/io.h>
#include <libk/logging.h>

void PIT_init(uint32_t frequency)
{
    module("PIT");

    uint32_t divisor = 1193181 / frequency;

    IO_outb(0x43, 0x36);
    IO_outb(0x40, (uint8_t)divisor & 0xFF);
    IO_outb(0x40, (uint8_t)(divisor >> 8) & 0xFF);

    log(INFO, "Initialized PIT with frequency: %d Hz", frequency);
}
volatile uint64_t ticks = 0;

void PIT_add_ticks()
{
    ticks++;
    IO_outb(0x20, 0x20);
}

uint64_t PIT_get_ticks()
{
    return ticks;
}
