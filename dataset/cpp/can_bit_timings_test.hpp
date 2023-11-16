/*
 * Copyright (c) 2009, Martin Rosekeit
 * Copyright (c) 2009-2010, Fabian Greif
 * Copyright (c) 2012, 2015, Niklas Hauser
 * Copyright (c) 2016, Kevin Läufer
 *
 * This file is part of the modm project.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
// ----------------------------------------------------------------------------

#include <unittest/testsuite.hpp>

/// @ingroup modm_test_test_platform
class CanBitTimingsTest : public unittest::TestSuite
{
public:
	void
	testPrecalculatedValues();
};
