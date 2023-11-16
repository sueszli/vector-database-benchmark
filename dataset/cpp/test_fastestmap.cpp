#include "gtest/gtest/gtest.h"
#include "../Includes/FasterDefs.h"
#include "../Share/TimeUtils.hpp"
#include "../Share/fmtlib.h"

USING_NS_WTP;

/*
 *	By Wesley @ 2023.08.16
 *	ankerlд���ٶȱ�robin�úܶ࣬��ſ�1/3��������������40w���ڵ�ʱ��
 *	����robin�Ķ�ȡ�ٶȱ�robin�ã���������30w���������ڣ����Ͳ���
 *	����wondertrader�ĳ���������ankerlҪ�úܶ�
 */
TEST(test_fastestmap, test_perform)
{
	fastest_hashmap<std::string, std::string> a;
	wt_hashmap<std::string, std::string> b;

	uint32_t times = 300000;

	char buffer[16] = { 0 };
	TimeUtils::Ticker ticker;

	for (uint32_t i = 0; i < times; i++)
	{
		char* s = fmtutil::format_to(buffer, "{:09d}", i);
		a[buffer] = buffer;
	}
	uint64_t t1 = ticker.nano_seconds();

	ticker.reset();
	for (uint32_t i = 0; i < times; i++)
	{
		char* s = fmtutil::format_to(buffer, "{:09d}", i);
		b[buffer] = buffer;
	}
	uint64_t t2 = ticker.nano_seconds();

	ticker.reset();
	for (uint32_t i = 0; i < times; i++)
	{
		char* s = fmtutil::format_to(buffer, "{:09d}", i);
		a[buffer];
	}
	uint64_t t3 = ticker.nano_seconds();

	ticker.reset();
	for (uint32_t i = 0; i < times; i++)
	{
		char* s = fmtutil::format_to(buffer, "{:09d}", i);
		b[buffer];
	}
	uint64_t t4 = ticker.nano_seconds();

	fmt::print("robin_write: {} - ankerl_write: {} - robin_read: {} - ankerl_read: {}\n", t1, t2, t3, t4);
}