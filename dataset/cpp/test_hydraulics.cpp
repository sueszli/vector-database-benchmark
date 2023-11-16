/*
 ******************************************************************************
 Project:      OWA EPANET
 Version:      2.2
 Module:       test_hydraulics.cpp
 Description:  Tests EPANET toolkit api functions
 Authors:      see AUTHORS
 Copyright:    see AUTHORS
 License:      see LICENSE
 Last Updated: 03/21/2019
 ******************************************************************************
*/

#include <boost/test/unit_test.hpp>
#include <boost/filesystem.hpp>

#include "test_toolkit.hpp"


BOOST_AUTO_TEST_SUITE (test_hydraulics)

BOOST_FIXTURE_TEST_CASE(test_solveH, FixtureOpenClose)
{
    error = EN_solveH(ph);
    BOOST_REQUIRE(error == 0);
}

BOOST_FIXTURE_TEST_CASE(test_hyd_step, FixtureOpenClose)
{
    int flag = 00;
    long t, tstep;

    error = EN_openH(ph);
    BOOST_REQUIRE(error == 0);

    error = EN_initH(ph, flag);
    BOOST_REQUIRE(error == 0);

    do {
        error = EN_runH(ph, &t);
        BOOST_REQUIRE(error == 0);

        error = EN_nextH(ph, &tstep);
        BOOST_REQUIRE(error == 0);

    } while (tstep > 0);

    error = EN_closeH(ph);
    BOOST_REQUIRE(error == 0);
}

BOOST_FIXTURE_TEST_CASE(test_hydr_save, FixtureOpenClose)
{
    error = EN_solveH(ph);
    BOOST_REQUIRE(error == 0);

    error = EN_saveH(ph);
    BOOST_REQUIRE(error == 0);

    error = EN_report(ph);
    BOOST_REQUIRE(error == 0);
}

BOOST_FIXTURE_TEST_CASE(test_hydr_savefile, FixtureOpenClose)
{
    error = EN_solveH(ph);
    BOOST_REQUIRE(error == 0);

    error = EN_savehydfile(ph, "test_savefile.hyd");
    BOOST_REQUIRE(error == 0);

    BOOST_CHECK(boost::filesystem::exists("test_savefile.hyd") == true);
}

BOOST_FIXTURE_TEST_CASE(test_hydr_usefile, FixtureOpenClose, * boost::unit_test::depends_on("test_hydraulics/test_hydr_savefile"))
{
    error = EN_usehydfile(ph, "test_savefile.hyd");
    BOOST_REQUIRE(error == 0);

    error = EN_solveQ(ph);
    BOOST_REQUIRE(error == 0);
}

BOOST_AUTO_TEST_SUITE_END()
