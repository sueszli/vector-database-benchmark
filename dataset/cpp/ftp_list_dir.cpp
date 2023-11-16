#include "log.h"
#include "mavsdk.h"
#include "plugins/ftp/ftp.h"
#include "plugins/ftp_server/ftp_server.h"
#include "fs_helpers.h"

#include <algorithm>
#include <filesystem>
#include <gtest/gtest.h>
#include <chrono>

using namespace mavsdk;

static constexpr double reduced_timeout_s = 0.1;

// TODO: make this compatible for Windows using GetTempPath2

static const fs::path temp_dir_provided = "/tmp/mavsdk_systemtest_temp_data/provided";

static const std::string temp_dir = "folder";
static const std::string temp_file = "file";

TEST(SystemTest, FtpListDir)
{
    ASSERT_TRUE(reset_directories(temp_dir_provided));

    std::vector<std::string> truth_list;

    for (unsigned i = 0; i < 100; ++i) {
        auto foldername = std::string(temp_dir + std::to_string(i));
        auto filename = std::string(temp_file + std::to_string(i));
        ASSERT_TRUE(reset_directories(temp_dir_provided / fs::path(foldername)));
        ASSERT_TRUE(create_temp_file(temp_dir_provided / fs::path(filename), i));

        truth_list.push_back(std::string("D") + foldername);
        truth_list.push_back(std::string("F") + filename + std::string("\t") + std::to_string(i));
    }

    std::sort(truth_list.begin(), truth_list.end());

    Mavsdk mavsdk_groundstation;
    mavsdk_groundstation.set_configuration(
        Mavsdk::Configuration{Mavsdk::Configuration::UsageType::GroundStation});
    mavsdk_groundstation.set_timeout_s(reduced_timeout_s);

    Mavsdk mavsdk_autopilot;
    mavsdk_autopilot.set_configuration(
        Mavsdk::Configuration{Mavsdk::Configuration::UsageType::Autopilot});
    mavsdk_autopilot.set_timeout_s(reduced_timeout_s);

    ASSERT_EQ(mavsdk_groundstation.add_any_connection("udp://:17000"), ConnectionResult::Success);
    ASSERT_EQ(
        mavsdk_autopilot.add_any_connection("udp://127.0.0.1:17000"), ConnectionResult::Success);

    auto ftp_server = FtpServer{
        mavsdk_autopilot.server_component_by_type(Mavsdk::ServerComponentType::Autopilot)};

    auto maybe_system = mavsdk_groundstation.first_autopilot(10.0);
    ASSERT_TRUE(maybe_system);
    auto system = maybe_system.value();

    ASSERT_TRUE(system->has_autopilot());
    auto ftp = Ftp{system};

    // First we try to list a folder without the root directory set.
    // We expect an error as we don't have any permission.
    EXPECT_EQ(ftp.list_directory("./").first, Ftp::Result::ProtocolError);

    // Now we set the root dir and expect it to work.
    ftp_server.set_root_dir(temp_dir_provided.string());

    auto ret = ftp.list_directory("./");
    EXPECT_EQ(ret.first, Ftp::Result::Success);

    EXPECT_EQ(ret.second, truth_list);
}
