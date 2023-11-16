/*
 * Copyright (c) 2009-2010, Fabian Greif
 * Copyright (c) 2010, Martin Rosekeit
 * Copyright (c) 2012, Niklas Hauser
 * Copyright (c) 2012, Sascha Schade
 * Copyright (c) 2015, Kevin Läufer
 *
 * This file is part of the modm project.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
// ----------------------------------------------------------------------------

#ifndef	UNITTEST_REPORTER_HPP
#define	UNITTEST_REPORTER_HPP

#include <stdint.h>

#include <modm/io/iostream.hpp>
#include <modm/architecture/interface/accessor_flash.hpp>

namespace unittest
{
	/**
	 * \brief	Reporter
	 *
	 * Used to generate the visible output.
	 *
	 * \author	Fabian Greif
	 * \ingroup	modm_unittest
	 */
	class Reporter
	{
	public:
		/**
		 * \brief	Constructor
		 *
		 * \param	device	IODevice used for printing
		 */
		Reporter(modm::IODevice& device);

		/**
		 * \brief	Switch to the next test suite
		 *
		 * \param	name	Name of the test suite
		 */
		void
		nextTestSuite(modm::accessor::Flash<char> name);

		/**
		 * \brief	Report a passed test
		 *
		 * Doesn't generate any output, but increments the number of
		 * passed tests
		 */
		void
		reportPass();

		/**
		 * \brief	Reported a failed test
		 *
		 * Generates a basic failure message, the returned stream can then
		 * be used to write some more specific information about the failure.
		 */
		modm::IOStream&
		reportFailure(unsigned int lineNumber);

		/**
		 * \brief	Writes a summary of all the tests
		 *
		 * Basically the total number of failed and passed tests and then
		 * 'OK' if there was no failure or 'FAIL' otherwise.
		 * @return 0 if all tests passed
		 */
		uint8_t
		printSummary();

	private:
		modm::IOStream outputStream;
		modm::accessor::Flash<char> testName;

		int_fast16_t testsPassed;
		int_fast16_t testsFailed;
	};
}

#endif	// UNITTEST_REPORTER_HPP
