#include <gtest/gtest.h>

#include <cstdint>

#include <functional>
#include <future>
#include <vector>

#include <boost/asio/io_service.hpp>

#include "ssf/system/system_interfaces.h"
#include "ssf/system/system_routers.h"

class RouterSystemTestFixture : public ::testing::Test {
 protected:
  RouterSystemTestFixture()
      : system_multiple_interfaces_config_filename_(
            "./system/system_multiple_config.json"),
        system_router_config_filename_("./system/router_config.json"),
        fail_system_router_config_filename_(
            "./system/fail_router_config.json") {}

  virtual ~RouterSystemTestFixture() {}

  virtual void SetUp() {}

  virtual void TearDown() {}

 protected:
  std::string system_multiple_interfaces_config_filename_;
  std::string system_router_config_filename_;
  std::string fail_system_router_config_filename_;
};

TEST_F(RouterSystemTestFixture, FailImportRouters) {
  boost::asio::io_service io_service;
  boost::system::error_code ec;

  ssf::system::SystemRouters system_routers(io_service);

  system_routers.Start();

  std::promise<bool> all_up;
  auto all_routers_up = [&all_up](const boost::system::error_code& ec) {
    all_up.set_value(!ec);
  };

  ASSERT_EQ(0, system_routers.AsyncConfig(
                   system_multiple_interfaces_config_filename_,
                   fail_system_router_config_filename_, ec, all_routers_up));

  ASSERT_NE(0, ec.value()) << "Error when configuring routers : "
                           << ec.message();

  std::vector<std::thread> threads;
  for (uint16_t i = 1; i <= std::thread::hardware_concurrency(); ++i) {
    threads.emplace_back([&io_service]() { io_service.run(); });
  }

  ASSERT_FALSE(all_up.get_future().get()) << "All routers up";

  system_routers.Stop();

  for (auto& thread : threads) {
    if (thread.joinable()) {
      thread.join();
    }
  }
}

TEST_F(RouterSystemTestFixture, ImportRouters) {
  boost::asio::io_service io_service;

  ssf::system::SystemRouters system_routers(io_service);

  boost::system::error_code ec;
  system_routers.Start();

  std::promise<bool> all_up;

  auto all_routers_up = [&all_up](const boost::system::error_code& ec) {
    all_up.set_value(!ec);
  };

  ASSERT_EQ(2, system_routers.AsyncConfig(
                   system_multiple_interfaces_config_filename_,
                   system_router_config_filename_, ec, all_routers_up));

  ASSERT_EQ(0, ec.value()) << "Error when configuring routers : "
                           << ec.message();

  std::vector<std::thread> threads;
  for (uint16_t i = 1; i <= std::thread::hardware_concurrency(); ++i) {
    threads.emplace_back([&io_service]() { io_service.run(); });
  }

  ASSERT_TRUE(all_up.get_future().get()) << "All routers not up";

  system_routers.Stop();

  for (auto& thread : threads) {
    if (thread.joinable()) {
      thread.join();
    }
  }
}
