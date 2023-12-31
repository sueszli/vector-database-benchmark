/**
 * GroundInterfaceRules.hpp:
 *
 * This file specifies Rule classes for testing of the Svc::GroundInterface. These rules can then be used by the main
 * testing program to test the code. These rules support rule-based random testing.
 *
 * GroundInterface rules:
 *
 * 1. On read-callback of sufficient parts, an uplink-out is produced
 * 2. On schedIn a sufficient number of times, an uplink-out is produced
 * 3. On a call to Log, TextLog, downlink, a framed call to write is produced.
 *
 * @author mstarch
 */
#ifndef FPRIME_SVC_GROUND_INTERFACE_HPP
#define FPRIME_SVC_GROUND_INTERFACE_HPP

#include <FpConfig.hpp>
#include <Fw/Types/EightyCharString.hpp>
#include "GroundInterfaceTester.hpp"
#include <STest/STest/Rule/Rule.hpp>
#include <STest/STest/Pick/Pick.hpp>


namespace Svc {

    /**
     * RandomizeRule:
     *
     * This rule sets up random state
     */
    struct RandomizeRule : public STest::Rule<GroundInterfaceTester> {
        // Constructor
        RandomizeRule(const Fw::EightyCharString& name);

        // Always valid
        bool precondition(const GroundInterfaceTester& state);

        // Will randomize the test state
        void action(GroundInterfaceTester& truth);
    };

    struct DownlinkRule : public STest::Rule<GroundInterfaceTester> {
        // Constructor
        DownlinkRule(const Fw::EightyCharString& name);

        // Always valid
        bool precondition(const GroundInterfaceTester& state);

        // Will randomize the test state
        void action(GroundInterfaceTester& truth);
    };

    struct FileDownlinkRule : public STest::Rule<GroundInterfaceTester> {
        // Constructor
        FileDownlinkRule(const Fw::EightyCharString& name);

        // Always valid
        bool precondition(const GroundInterfaceTester& state);

        // Will randomize the test state
        void action(GroundInterfaceTester& truth);
    };

    struct SendAvailableRule : public STest::Rule<GroundInterfaceTester> {
        // Constructor
        SendAvailableRule(const Fw::EightyCharString& name);

        // Always valid
        bool precondition(const GroundInterfaceTester& state);

        // Will randomize the test state
        void action(GroundInterfaceTester& truth);
    };

};
#endif //FPRIME_SVC_GROUND_INTERFACE_HPP
