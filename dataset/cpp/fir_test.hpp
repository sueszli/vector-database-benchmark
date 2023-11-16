/*
 * Copyright (c) 2009, Martin Rosekeit
 * Copyright (c) 2009, Thorsten Lajewski
 * Copyright (c) 2009-2011, Fabian Greif
 * Copyright (c) 2012, Kevin Läufer
 * Copyright (c) 2012, Niklas Hauser
 *
 * This file is part of the modm project.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
// ----------------------------------------------------------------------------

#include <unittest/testsuite.hpp>

/// @ingroup modm_test_test_math
class FirTest : public unittest::TestSuite
{
public:

	void
	testFir();

private:
	/* Length of results array needs to be len(taps) + len(coeff) */
	template<typename T, int N, int BLOCK_SIZE, unsigned int ScaleFactor>
	void testFilter(const float (&coeff)[N],
		const T taps[], int taps_length, const T results[]);
};
