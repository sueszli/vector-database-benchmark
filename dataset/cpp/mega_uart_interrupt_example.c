/**
 * \file
 *
 * \brief megaAVR STK600 UART interrupt example
 *
 * Copyright (c) 2014-2015 Atmel Corporation. All rights reserved.
 *
 * \asf_license_start
 *
 * \page License
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * 3. The name of Atmel may not be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * 4. This software may only be redistributed and used in connection with an
 *    Atmel microcontroller product.
 *
 * THIS SOFTWARE IS PROVIDED BY ATMEL "AS IS" AND ANY EXPRESS OR IMPLIED
 * WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT ARE
 * EXPRESSLY AND SPECIFICALLY DISCLAIMED. IN NO EVENT SHALL ATMEL BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
 * STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 * ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 * \asf_license_stop
 *
 */
/**
 * \mainpage
 * \section board STK600 development board
 * \section intro Introduction
 * This example demonstrates how to use the megaAVR UART with interrupts.
 *
 * \section files Files:
 * - mega_uart_interrupt_example.c: megaAVR STK600 UART interrupt example
 *
 * \section exampledescription Brief description of the example application
 * This application will initialize the UART, send a string and then receive
 * it and check if it is the same string as we sent.
 *
 * \note The RX and TX pins should be externally connected in order to pass the
 * test.
 *
 * \section compinfo Compilation Info
 * This software was written for the <A href="http://gcc.gnu.org/">GNU GCC</A>
 * for AVR. \n
 * Other compilers may or may not work.
 *
 * \section contactinfo Contact Information
 * For further information, visit
 * <A href="http://www.atmel.com/">Atmel</A>.\n
 */
/*
 * Support and FAQ: visit <a href="http://www.atmel.com/design-support/">Atmel Support</a>
 */

 // _ASSERT_ENABLE_ is used for enabling assert, typical for debug purposes
#define _ASSERT_ENABLE_
#include <string.h>
#include "compiler.h"

/**
 * \def BUFFER_SIZE
 * \brief The size of the UART buffer
 */
#define BUFFER_SIZE 64

// set the correct BAUD and F_CPU defines before including setbaud.h
#include "conf_clock.h"
#include "conf_uart.h"

/**
 * \name avr_libc_inc avr libc include files
 * @{
 */
#include <util/setbaud.h>
#include <util/delay.h>
#include <avr/interrupt.h>
//! @}

#include "ring_buffer.h"

void port_init(void);
uint8_t getCheckSum(uint8_t *data, uint8_t length);
// buffers for use with the ring buffer (belong to the UART)
uint8_t out_buffer[BUFFER_SIZE];
uint8_t in_buffer[BUFFER_SIZE];

// the string we send and receive on UART
char test_string[] = "CCM";
uint8_t tbuf[64] = {0,};
uint8_t rbuf[64] = {0,};


//! ring buffer to use for the UART transmission
struct ring_buffer ring_buffer_out;
//! ring buffer to use for the UART reception
struct ring_buffer ring_buffer_in;

/**
 * \brief UART data register empty interrupt handler
 *
 * This handler is called each time the UART data register is available for
 * sending data.
 */
ISR(UART0_DATA_EMPTY_IRQ)
{
	// if there is data in the ring buffer, fetch it and send it
	if (!ring_buffer_is_empty(&ring_buffer_out)) {
		UDR0 = ring_buffer_get(&ring_buffer_out);
	}
	else {
		// no more data to send, turn off data ready interrupt
		UCSR0B &= ~(1 << UDRIE0);
	}
}

/**
 * \brief Data RX interrupt handler
 *
 * This is the handler for UART receive data
 */
ISR(UART0_RX_IRQ)
{
	ring_buffer_put(&ring_buffer_in, UDR0);
	
}

/**
 * \brief Initialize the UART with correct baud rate settings
 *
 * This function will initialize the UART baud rate registers with the correct
 * values using the AVR libc setbaud utility. In addition set the UART to
 * 8-bit, 1 stop and no parity.
 */

void port_init(void)
{
	
	DDRB =(0<<DDB7)|(0<<DDB6)|(0<<DDB5)|(0<<DDB4)|(0<<DDB3)|(1<<DDB2)|(1<<DDB1)|(1<<DDB0);
	DDRC =(1<<DDC6)|(1<<DDC5)|(1<<DDC4)|(1<<DDC3)|(1<<DDC2)|(1<<DDC1)|(1<<DDC0);
	DDRD =(1<<DDD7)|(1<<DDD6)|(1<<DDD5)|(0<<DDD4)|(0<<DDD3)|(0<<DDD2)|(0<<DDD0);
	
	PORTB = 0x00;
	PORTC = 0x00;
	PORTD = 0x00;
	
	PORTC = PORTC ^ (1<<PORTC0);_delay_ms(100);
	PORTC = PORTC ^ (1<<PORTC1);_delay_ms(100);
	PORTC = PORTC ^ (1<<PORTC2);_delay_ms(100);
	PORTC = PORTC ^ (1<<PORTC3);_delay_ms(100);
	PORTC = PORTC ^ (1<<PORTC4);_delay_ms(100);
	PORTC = PORTC ^ (1<<PORTC5);_delay_ms(100);
	
	PORTB = PORTB ^ (1<<PORTB0); _delay_ms(100);
	PORTB = PORTB ^ (1<<PORTB1); _delay_ms(100);
	PORTB = PORTB ^ (1<<PORTB2); _delay_ms(100);

	PORTD = PORTD ^ (1<<PORTD7); _delay_ms(100);
	PORTD = PORTD ^ (1<<PORTD6); _delay_ms(100);
	PORTD = PORTD ^ (1<<PORTD5); _delay_ms(100);
		
	_delay_ms(100);
	
	PORTC = PORTC ^ (1<<PORTC0);_delay_ms(100);
	PORTC = PORTC ^ (1<<PORTC1);_delay_ms(100);
	PORTC = PORTC ^ (1<<PORTC2);_delay_ms(100);
	PORTC = PORTC ^ (1<<PORTC3);_delay_ms(100);
	PORTC = PORTC ^ (1<<PORTC4);_delay_ms(100);
	PORTC = PORTC ^ (1<<PORTC5);_delay_ms(100);
	
	PORTB = PORTB ^ (1<<PORTB0); _delay_ms(100);
	PORTB = PORTB ^ (1<<PORTB1); _delay_ms(100);
	PORTB = PORTB ^ (1<<PORTB2); _delay_ms(100);

	PORTD = PORTD ^ (1<<PORTD7); _delay_ms(100);
	PORTD = PORTD ^ (1<<PORTD6); _delay_ms(100);
	PORTD = PORTD ^ (1<<PORTD5); _delay_ms(100);
	
	PORTB = 0x00;
	PORTC = 0x00;
	PORTD = 0x00;
}
static void uart_init(void)
{
	//uint32_t ubrr = MYUBRR;
	//UBRR0H = (uint8_t)(ubrr >>8);
	//UBRR0L = (uint8_t)(ubrr);	
	
#if defined UBRR0H
	// get the values from the setbaud tool
	UBRR0H = UBRRH_VALUE;
	UBRR0L = UBRRL_VALUE;
#else
#error "Device is not supported by the driver"
#endif

#if USE_2X
	UCSR0A |= (1 << U2X0);
#endif

	// enable RX and TX and set interrupts on rx complete
	UCSR0B = (1 << RXEN0) | (1 << TXEN0) | (1 << RXCIE0);

	// 8-bit, 1 stop bit, no parity, asynchronous UART
	UCSR0C = (1 << UCSZ01) | (1 << UCSZ00) | (0 << USBS0) |
			(0 << UPM01) | (0 << UPM00) | (0 << UMSEL01) |
			(0 << UMSEL00);
			
	UCSR0B |=  (1 << UDRIE0);

	// initialize the in and out buffer for the UART
	ring_buffer_out = ring_buffer_init(out_buffer, BUFFER_SIZE);
	ring_buffer_in = ring_buffer_init(in_buffer, BUFFER_SIZE);
}

/**
 * \brief Function for putting a char in the UART buffer
 *
 * \param data the data to add to the UART buffer and send
 *
 */
static inline void uart_putchar(uint8_t data)
{
	// Disable interrupts to get exclusive access to ring_buffer_out.
	cli();
	if (ring_buffer_is_empty(&ring_buffer_out)) {
		// First data in buffer, enable data ready interrupt
		UCSR0B |=  (1 << UDRIE0);
	}
	// Put data in buffer
	ring_buffer_put(&ring_buffer_out, data);

	// Re-enable interrupts
	sei();
}

/**
 * \brief Function for getting a char from the UART receive buffer
 *
 * \retval Next data byte in receive buffer
 */
static inline uint8_t uart_getchar(void)
{
	return ring_buffer_get(&ring_buffer_in);
}


/**
 * \brief Function to check if we have a char waiting in the UART receive buffer
 *
 * \retval true if data is waiting
 * \retval false if no data is waiting
 */
static inline bool uart_char_waiting(void)
{
	return !ring_buffer_is_empty(&ring_buffer_in);
}

uint8_t getCheckSum(uint8_t *data, uint8_t length)
{
	uint8_t csum;

	csum = 0;
	for(;length>0;length--)
	{
		csum += *data++;
	}

	return 0xFF - csum;
}

/**
 * \brief The main application
 *
 * This application will initialize the UART, send a character and then receive
 * it and check if it is the same character as was sent.
 *
 * \note The RX and TX pins should be externally connected in order to pass the
 * test.
 */
int main(void)
{
	//uint8_t rdata;
	uint8_t cnt;
	uint8_t sposition=0;
	port_init();
	cli();
	uart_init();
	sei();

	// Send the test string
	for (cnt = 0; cnt < strlen(test_string); cnt++) {
		uart_putchar(test_string[cnt]);
	}
	while(true){
		//for (cnt = 0; cnt < strlen(test_string); cnt++) {
			//uart_putchar(test_string[cnt]);
		//}

		// Check if we have received the string we sent
		
		cnt = 0;
		do {
			// Wait for next character
			while (!uart_char_waiting());
			rbuf[cnt] = uart_getchar();
			cnt++;

		} while (cnt<13);//rbuf[cnt-1] == 'E' && rbuf[cnt-2] == 'M');
		

		if(rbuf[0] == 0x43 && rbuf[1] == 0x43 && rbuf[2] == 0x4D && rbuf[3] == 0x53)
		{
			//PORTC = PORTC ^ (1<<PORTC3);
			if(rbuf[8] == getCheckSum((uint8_t*)(rbuf + sposition), 8)){
				//PORTC = PORTC ^ (1<<PORTC4);
				uint8_t type = rbuf[sposition + 4];
				uint8_t data = rbuf[sposition + 7];

				switch(type){
					case 1 : { //
							switch(data){
								case 0x01 : PORTC = PORTC ^ (1<<PORTC0); break;
								case 0x11 : PORTC = PORTC | (1<<PORTC0); break;
								case 0x02 : PORTC = PORTC ^ (1<<PORTC1); break;
								case 0x12 : PORTC = PORTC | (1<<PORTC1); break;
								case 0x03 : PORTC = PORTC ^ (1<<PORTC2); break;
								case 0x13 : PORTC = PORTC | (1<<PORTC2); break;
								case 0x04 : PORTC = PORTC ^ (1<<PORTC3); break;
								case 0x14 : PORTC = PORTC | (1<<PORTC3); break;
								case 0x05 : PORTC = PORTC ^ (1<<PORTC4); break;
								case 0x15 : PORTC = PORTC | (1<<PORTC4); break;
								case 0x06 : PORTC = PORTC ^ (1<<PORTC5); break;
								case 0x16 : PORTC = PORTC | (1<<PORTC5); break;
								case 0x07 : PORTB = PORTB ^ (1<<PORTB0); break;
								case 0x17 : PORTB = PORTB | (1<<PORTB0); break;
								case 0x08 : PORTB = PORTB ^ (1<<PORTB1); break;
								case 0x18 : PORTB = PORTB | (1<<PORTB1); break;
								//case 0x09 : PORTC = PORTC ^ (1<<PORTC0); break;
								//case 0x19 : PORTC = PORTC | (1<<PORTC0); break;
								//case 0x0A : PORTC = PORTC ^ (1<<PORTC1); break;
								//case 0x1A : PORTC = PORTC | (1<<PORTC1); break;
								//case 0x0B : PORTC = PORTC ^ (1<<PORTC2); break;
								//case 0x1B : PORTC = PORTC | (1<<PORTC2); break;
								//case 0x0C : PORTC = PORTC ^ (1<<PORTC3); break;
								//case 0x1C : PORTC = PORTC | (1<<PORTC3); break;
								//case 0x0D : PORTC = PORTC ^ (1<<PORTC4); break;
								//case 0x1D : PORTC = PORTC | (1<<PORTC4); break;
								//case 0x0E : PORTC = PORTC ^ (1<<PORTC5); break;
								//case 0x1E : PORTC = PORTC | (1<<PORTC5); break;
								//case 0x0F : PORTB = PORTB ^ (1<<PORTB0); break;
								//case 0x1F : PORTB = PORTB | (1<<PORTB0); break;
								
							}
						break;
						}
					case 2 : {
							switch(data){
								case 0x21 : PORTD = PORTD | (1<<PORTD5); break;
								case 0x22 : PORTD = PORTD | (1<<PORTD6); break;
								case 0x23 : PORTD = PORTD | (1<<PORTD7); break;
								case 0x24 : PORTD = PORTD & ~(1<<PORTD5); break;
								case 0x25 : PORTD = PORTD & ~(1<<PORTD6); break;
								case 0x26 : PORTD = PORTD & ~(1<<PORTD7); break;
							}
						break;
						}
					case 3 : {
							switch(data){
								case 0x31 : PORTB = PORTB | (1<<PORTB2); break;
								case 0x32 : PORTB = PORTB & ~(1<<PORTB2); break;
							}
						break;
						}
					case 4 : {
						PORTB = 0; PORTC = 0;PORTD = 0;break;
					}
				}
				
			}
			else
			{
				
			}
			
		}
	}

	//while (true);
}
