/*
 * Copyright (c) 2011-2012, Fabian Greif
 * Copyright (c) 2012, 2015, Sascha Schade
 * Copyright (c) 2012-2014, 2016-2017, Niklas Hauser
 *
 * This file is part of the modm project.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
// ----------------------------------------------------------------------------

#ifndef MODM_FT245_HPP
#define MODM_FT245_HPP

#include <modm/architecture/interface/gpio.hpp>

namespace modm
{
	/**
	 * \ingroup	modm_driver_ft245
	 */
	template < typename PORT,
	           typename RD,
	           typename WR,
	           typename RXF,
	           typename TXE >
	class Ft245
	{
	public:
		/// Write a single byte to the FIFO
		/// \param	data	Single byte to write
		static bool
		write(uint8_t data);

		/**
		 * Write a block of bytes to the FIFO
		 *
		 * This blocks until the buffer is written.
		 *
		 * \param	*buffer	Buffer of the data that should be written
		 * \param	nbyte	Length of buffer
		 *
		 */
		static void
		write(const uint8_t *buffer, uint8_t nbyte);

		static void
		flushWriteBuffer() {};

		/**
		 * Read a single byte from the FIFO
		 *
		 * \param	c		Byte read, if any
		 *
		 * \return	\c true if a byte was received, \c false otherwise
		 */
		static bool
		read(uint8_t &c);

		/**
		 * Read a block of bytes from the FIFO
		 *
		 * This is blocking.
		 *
		 * \param	*buffer	Buffer for the received data.
		 * \param	nbyte	Length of buffer
		 *
		 */
		static uint8_t
		read(uint8_t *buffer, uint8_t nbyte);

	protected:
		static PORT port;
		static RD   rd;
		static WR   wr;
		static RXF  rxf;
		static TXE  txe;
	};
}

#include "ft245_impl.hpp"

#endif // MODM_FT2425_HPP
