/*
 * Copyright 2011-2017 Haiku, Inc. All rights reserved.
 * Distributed under the terms of the MIT License.
 *
 * Authors:
 *		Alexander von Gluck, kallisti5@unixzen.com
 */


#include <debug.h>
#include <arch/arm/reg.h>
#include <arch/generic/debug_uart.h>
#include <arch/arm/arch_uart_pl011.h>
#include <new>


#define PL01x_DR	0x00 // Data read or written
#define PL01x_RSR	0x04 // Receive status, read
#define PL01x_ECR	0x04 // Error clear, write
#define PL010_LCRH	0x08 // Line control, high
#define PL010_LCRM	0x0C // Line control, middle
#define PL010_LCRL	0x10 // Line control, low
#define PL010_CR	0x14 // Control
#define PL01x_FR	0x18 // Flag (r/o)
#define PL010_IIR	0x1C // Interrupt ID (r)
#define PL010_ICR	0x1C // Interrupt clear (w)
#define PL01x_ILPR	0x20 // IrDA low power
#define PL011_IBRD	0x24 // Interrupt baud rate divisor
#define PL011_FBRD	0x28 // Fractional baud rate divisor
#define PL011_LCRH	0x2C // Line control
#define PL011_CR	0x30 // Control
#define PL011_IFLS	0x34 // Interrupt fifo level
#define PL011_IMSC	0x38 // Interrupt mask
#define PL011_RIS	0x3C // Raw interrupt
#define PL011_MIS	0x40 // Masked interrupt
#define PL011_ICR	0x44 // Interrupt clear
#define PL011_DMACR	0x48 // DMA control register

#define PL011_DR_OE		(1 << 11)
#define PL011_DR_BE		(1 << 10)
#define PL011_DR_PE		(1 << 9)
#define PL011_DR_FE		(1 << 8)

#define PL01x_RSR_OE	0x08
#define PL01x_RSR_BE	0x04
#define PL01x_RSR_PE	0x02
#define PL01x_RSR_FE	0x01

#define PL011_FR_RI		0x100
#define PL011_FR_TXFE	0x080
#define PL011_FR_RXFF	0x040
#define PL01x_FR_TXFF	0x020
#define PL01x_FR_RXFE	0x010
#define PL01x_FR_BUSY	0x008
#define PL01x_FR_DCD	0x004
#define PL01x_FR_DSR	0x002
#define PL01x_FR_CTS	0x001
#define PL01x_FR_TMSK	(PL01x_FR_TXFF | PL01x_FR_BUSY)

#define PL011_CR_CTSEN	0x8000 // CTS flow control
#define PL011_CR_RTSEN	0x4000 // RTS flow control
#define PL011_CR_OUT2	0x2000 // OUT2
#define PL011_CR_OUT1	0x1000 // OUT1
#define PL011_CR_RTS	0x0800 // RTS
#define PL011_CR_DTR	0x0400 // DTR
#define PL011_CR_RXE	0x0200 // Receive enable
#define PL011_CR_TXE	0x0100 // Transmit enable
#define PL011_CR_LBE	0x0080 // Loopback enable
#define PL010_CR_RTIE	0x0040
#define PL010_CR_TIE	0x0020
#define PL010_CR_RIE	0x0010
#define PL010_CR_MSIE	0x0008
#define PL01x_CR_IIRLP	0x0004 // SIR low power mode
#define PL01x_CR_SIREN	0x0002 // SIR enable
#define PL01x_CR_UARTEN 0x0001 // UART enable

#define PL011_LCRH_SPS		0x80
#define PL01x_LCRH_WLEN_8	0x60
#define PL01x_LCRH_WLEN_7	0x40
#define PL01x_LCRH_WLEN_6	0x20
#define PL01x_LCRH_WLEN_5	0x00
#define PL01x_LCRH_FEN		0x10
#define PL01x_LCRH_STP2		0x08
#define PL01x_LCRH_EPS		0x04
#define PL01x_LCRH_PEN		0x02
#define PL01x_LCRH_BRK		0x01

#define PL010_IIR_RTIS	0x08
#define PL010_IIR_TIS	0x04
#define PL010_IIR_RIS	0x02
#define PL010_IIR_MIS	0x01

#define PL011_IFLS_RX1_8	(0 << 3)
#define PL011_IFLS_RX2_8	(1 << 3)
#define PL011_IFLS_RX4_8	(2 << 3)
#define PL011_IFLS_RX6_8	(3 << 3)
#define PL011_IFLS_RX7_8	(4 << 3)
#define PL011_IFLS_TX1_8	(0 << 0)
#define PL011_IFLS_TX2_8	(1 << 0)
#define PL011_IFLS_TX4_8	(2 << 0)
#define PL011_IFLS_TX6_8	(3 << 0)
#define PL011_IFLS_TX7_8	(4 << 0)

#define PL011_IFLS_RX_HALF	(5 << 3) // ST vendor only
#define PL011_IFLS_TX_HALF	(5 << 0) // ST vendor only

#define PL011_OEIM		(1 << 10) // overrun error interrupt mask
#define PL011_BEIM		(1 << 9) // break error interrupt mask
#define PL011_PEIM		(1 << 8) // parity error interrupt mask
#define PL011_FEIM		(1 << 7) // framing error interrupt mask
#define PL011_RTIM		(1 << 6) // receive timeout interrupt mask
#define PL011_TXIM		(1 << 5) // transmit interrupt mask
#define PL011_RXIM		(1 << 4) // receive interrupt mask
#define PL011_DSRMIM	(1 << 3) // DSR interrupt mask
#define PL011_DCDMIM	(1 << 2) // DCD interrupt mask
#define PL011_CTSMIM	(1 << 1) // CTS interrupt mask
#define PL011_RIMIM		(1 << 0) // RI interrupt mask
#define PL011_MSKIM		0x7ff	 // Mask all interrupts

#define PL011_OEIS		(1 << 10) // overrun error interrupt state
#define PL011_BEIS		(1 << 9) // break error interrupt state
#define PL011_PEIS		(1 << 8) // parity error interrupt state
#define PL011_FEIS		(1 << 7) // framing error interrupt	state
#define PL011_RTIS		(1 << 6) // receive timeout interrupt state
#define PL011_TXIS		(1 << 5) // transmit interrupt state
#define PL011_RXIS		(1 << 4) // receive interrupt state
#define PL011_DSRMIS	(1 << 3) // DSR interrupt state
#define PL011_DCDMIS	(1 << 2) // DCD interrupt state
#define PL011_CTSMIS	(1 << 1) // CTS interrupt state
#define PL011_RIMIS		(1 << 0) // RI interrupt state

#define PL011_OEIC		(1 << 10) // overrun error interrupt clear
#define PL011_BEIC		(1 << 9) // break error interrupt clear
#define PL011_PEIC		(1 << 8) // parity error interrupt clear
#define PL011_FEIC		(1 << 7) // framing error interrupt clear
#define PL011_RTIC		(1 << 6) // receive timeout interrupt clear
#define PL011_TXIC		(1 << 5) // transmit interrupt clear
#define PL011_RXIC		(1 << 4) // receive interrupt clear
#define PL011_DSRMIC	(1 << 3) // DSR interrupt clear
#define PL011_DCDMIC	(1 << 2) // DCD interrupt clear
#define PL011_CTSMIC	(1 << 1) // CTS interrupt clear
#define PL011_RIMIC		(1 << 0) // RI interrupt clear

#define PL011_DMAONERR	(1 << 2) // disable dma on err
#define PL011_TXDMAE	(1 << 1) // enable transmit dma
#define PL011_RXDMAE	(1 << 0) // enable receive dma


ArchUARTPL011::ArchUARTPL011(addr_t base, int64 clock)
	:
	DebugUART(base, clock)
{
	Barrier();

	// ** Loopback test
	uint32 cr = PL01x_CR_UARTEN;
		// Enable UART
	cr |= PL011_CR_TXE;
		// Enable TX
	cr |= PL011_CR_LBE;
		// Enable Loopback mode
	Out32(PL011_CR, cr);

	Out32(PL011_FBRD, 0);
	Out32(PL011_IBRD, 1);
	Out32(PL011_LCRH, 0); // TODO: ST is different tx, rx lcr

	// Write a 0 to the port and wait for confim..
	Out32(PL01x_DR, 0);

	while (In32(PL01x_FR) & PL01x_FR_BUSY)
		Barrier();

	// ** Disable loopback, enable uart
	cr = PL01x_CR_UARTEN | PL011_CR_RXE | PL011_CR_TXE;
	Out32(PL011_CR, cr);

	// ** Clear interrupts
	Out32(PL011_ICR, PL011_OEIS | PL011_BEIS
		| PL011_PEIS | PL011_FEIS);

	// ** Disable interrupts
	Out32(PL011_IMSC, In32(PL011_IMSC) & ~PL011_MSKIM);
}


ArchUARTPL011::~ArchUARTPL011()
{
}


void
ArchUARTPL011::Out32(int reg, uint32 data)
{
	*(volatile uint32*)(Base() + reg) = data;
}


uint32
ArchUARTPL011::In32(int reg)
{
	return *(volatile uint32*)(Base() + reg);
}


void
ArchUARTPL011::Barrier()
{
	asm volatile ("" : : : "memory");
}


void
ArchUARTPL011::InitPort(uint32 baud)
{
	// Calculate baud divisor
	uint32 baudDivisor = Clock() / (16 * baud);
	uint32 remainder = Clock() % (16 * baud);
	uint32 baudFractional = ((8 * remainder) / baud >> 1)
		+ ((8 * remainder) / baud & 1);

	// Disable UART
	Disable();

	// Set baud divisor
	Out32(PL011_IBRD, baudDivisor);
	Out32(PL011_FBRD, baudFractional);

	// Set LCR 8n1, enable fifo
	Out32(PL011_LCRH, (In32(PL011_LCRH) & ~0xff)
		| PL01x_LCRH_WLEN_8 | PL01x_LCRH_FEN);

	// Set FIFO levels
	Out32(PL011_IFLS, PL011_IFLS_RX4_8 | PL011_IFLS_TX4_8);

	// Enable UART
	Enable();
}


void
ArchUARTPL011::InitEarly()
{
	// Perform special hardware UART configuration
}


void
ArchUARTPL011::Enable()
{
	uint32 cr = PL01x_CR_UARTEN;
		// Enable UART
	cr |= PL011_CR_TXE | PL011_CR_RXE;
		// Enable TX and RX

	Out32(PL011_CR, cr);

	DebugUART::Enable();
}


void
ArchUARTPL011::Disable()
{
	// Disable everything
	Out32(PL011_CR, 0);

	DebugUART::Disable();
}


int
ArchUARTPL011::PutChar(char c)
{
	if (Enabled() == true) {
		// Wait until there is room in fifo
		while ((In32(PL01x_FR) & PL01x_FR_TXFF) != 0)
			Barrier();

		Out32(PL01x_DR, c);
		return 0;
	}

	return -1;
}


int
ArchUARTPL011::GetChar(bool wait)
{
	if (Enabled() == true) {
		if (wait) {
			// Wait until a character is received
			while ((In32(PL01x_FR) & PL01x_FR_RXFE) != 0)
				Barrier();
		} else {
			// Check if there is any data available in RX fifo
			if ((In32(PL01x_FR) & PL01x_FR_RXFE) != 0)
				return -1;
		}
		return In32(PL01x_DR);
	}

	return -1;
}


void
ArchUARTPL011::FlushTx()
{
	// Wait until transmit fifo empty
	while ((In32(PL01x_FR) & PL011_FR_TXFE) == 0)
		Barrier();
}


void
ArchUARTPL011::FlushRx()
{
	// Wait until receive fifo empty
	while ((In32(PL01x_FR) & PL01x_FR_RXFE) == 0)
		Barrier();
}


ArchUARTPL011*
arch_get_uart_pl011(addr_t base, int64 clock)
{
	static char buffer[sizeof(ArchUARTPL011)];
	ArchUARTPL011 *uart = new(buffer) ArchUARTPL011(base, clock);
	return uart;
}
