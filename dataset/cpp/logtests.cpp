#include "thirdparty/doctest.h"
#include "lib/log.hpp"
#include "lib/developermode.hpp"

TEST_CASE("log")
{
	auto init_log = []()
	{
		lib::log::clear();
		lib::log::set_log_to_stdout(false);
	};

	auto verify_messages = []()
	{
		auto messages = lib::log::get_messages();
		CHECK_EQ(messages.size(), 2);

		for (auto &message: messages)
		{
			CHECK_EQ(message.get_message(), "hello world");
		}
	};

	SUBCASE("info")
	{
		init_log();
		lib::log::info("hello world");
		lib::log::info("hello {}", "world");
		verify_messages();
	}

	SUBCASE("warn")
	{
		init_log();
		lib::log::warn("hello world");
		lib::log::warn("hello {}", "world");
		verify_messages();
	}

	SUBCASE("error")
	{
		init_log();
		lib::log::error("hello world");
		lib::log::error("hello {}", "world");
		verify_messages();
	}

	SUBCASE("debug")
	{
		init_log();

		lib::log::debug("hello world");
		lib::log::debug("hello {}", "world");
		CHECK_EQ(lib::log::get_messages().size(), 0);

		lib::developer_mode::enabled = true;
		lib::log::debug("hello world");
		lib::log::debug("hello {}", "world");
		verify_messages();
	}
}
