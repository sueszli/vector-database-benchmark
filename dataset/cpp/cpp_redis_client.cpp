// The MIT License (MIT)
//
// Copyright (c) 2015-2017 Simon Ninon <simon.ninon@gmail.com>
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
#include <string>
#include <cpp_redis/cpp_redis>
#include <cpp_redis/misc/macro.hpp>
#include "examples/winsock_initializer.h"

#define ENABLE_SESSION = 1

int main(void) {
	winsock_initializer winsock_init;
	//! Enable logging
	cpp_redis::active_logger = std::unique_ptr<cpp_redis::logger>(new cpp_redis::logger);

	cpp_redis::client client;

	client.connect("127.0.0.1", 6379,
	               [](const std::string &host, std::size_t port, cpp_redis::connect_state status) {
			               if (status == cpp_redis::connect_state::dropped) {
				               std::cout << "client disconnected from " << host << ":" << port << std::endl;
			               }
	               });

	auto replcmd = [](const cpp_redis::reply &reply) {
			std::cout << "set hello 42: " << reply << std::endl;
			// if (reply.is_string())
			//   do_something_with_string(reply.as_string())
	};

	const std::string group_name = "groupone";
	const std::string session_name = "sessone";
	const std::string consumer_name = "ABCD";

	std::multimap<std::string, std::string> ins;
	ins.insert(std::pair<std::string, std::string>{"message", "hello"});

#ifdef ENABLE_SESSION

	client.xadd(session_name, "*", ins, replcmd);
	client.xgroup_create(session_name, group_name, "0", replcmd);

	client.sync_commit();

	client.xrange(session_name, {"-", "+", 10}, replcmd);

	client.xreadgroup({group_name,
	                   consumer_name,
	                   {{session_name}, {">"}},
	                   1, // Count
	                   0, // block milli
	                   false, // no ack
	                  }, [](cpp_redis::reply &reply) {
			std::cout << "set hello 42: " << reply << std::endl;
			auto msg = reply.as_array();
			std::cout << "Mes: " << msg[0] << std::endl;
			// if (reply.is_string())
			//   do_something_with_string(reply.as_string())
	});

#else

	// same as client.send({ "SET", "hello", "42" }, ...)
	client.set("hello", "42", [](cpp_redis::reply &reply) {
			std::cout << "set hello 42: " << reply << std::endl;
			// if (reply.is_string())
			//   do_something_with_string(reply.as_string())
	});

	// same as client.send({ "DECRBY", "hello", 12 }, ...)
	client.decrby("hello", 12, [](cpp_redis::reply &reply) {
			std::cout << "decrby hello 12: " << reply << std::endl;
			// if (reply.is_integer())
			//   do_something_with_integer(reply.as_integer())
	});

	// same as client.send({ "GET", "hello" }, ...)
	client.get("hello", [](cpp_redis::reply &reply) {
			std::cout << "get hello: " << reply << std::endl;
			// if (reply.is_string())
			//   do_something_with_string(reply.as_string())
	});

#endif

	// commands are pipelined and only sent when client.commit() is called
	// client.commit();

	// synchronous commit, no timeout
	client.sync_commit();

	// synchronous commit, timeout
	// client.sync_commit(std::chrono::milliseconds(100));

	return 0;
}
