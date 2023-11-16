#include "common.h"
#include "vnstat_tests.h"
#include "iflist_tests.h"
#include "iflist.h"

START_TEST(iflistfree_can_free_null)
{
	iflist *ifl = NULL;
    iflistfree(&ifl);
}
END_TEST

START_TEST(iflistadd_can_add)
{
    int ret;
    iflist *ifl = NULL;
    ck_assert_ptr_eq(ifl, NULL);

    ret = iflistadd(&ifl, "eth0", 1, 0);
    ck_assert_int_eq(ret, 1);
    ck_assert_str_eq(ifl->interface, "eth0");
    ck_assert_int_eq(ifl->id, 1);
    ck_assert_int_eq(ifl->bandwidth, 0);
    ck_assert_ptr_eq(ifl->next, NULL);

    ret = iflistadd(&ifl, "eth1", 2, 1);
    ck_assert_int_eq(ret, 1);
    ck_assert_str_eq(ifl->interface, "eth0");
    ck_assert_int_eq(ifl->id, 1);
    ck_assert_int_eq(ifl->bandwidth, 0);
    ck_assert_ptr_ne(ifl->next, NULL);
    ck_assert_str_eq(ifl->next->interface, "eth1");
    ck_assert_int_eq(ifl->next->id, 2);
    ck_assert_int_eq(ifl->next->bandwidth, 1);

    ret = iflistadd(&ifl, "eth0", 3, 2);
    ck_assert_int_eq(ret, 1);
    ck_assert_str_eq(ifl->interface, "eth0");
    ck_assert_int_eq(ifl->id, 1);
    ck_assert_int_eq(ifl->bandwidth, 0);
    ck_assert_ptr_ne(ifl->next, NULL);
    ck_assert_str_eq(ifl->next->interface, "eth1");
    ck_assert_int_eq(ifl->next->id, 2);
    ck_assert_int_eq(ifl->next->bandwidth, 1);
    ck_assert_ptr_ne(ifl->next->next, NULL);
    ck_assert_str_eq(ifl->next->next->interface, "eth0");
    ck_assert_int_eq(ifl->next->next->id, 3);
    ck_assert_int_eq(ifl->next->next->bandwidth, 2);

    iflistfree(&ifl);
    ck_assert_ptr_eq(ifl, NULL);
}
END_TEST

START_TEST(iflistsearch_can_search)
{
    int ret;
    iflist *ifl = NULL;
    ck_assert_ptr_eq(ifl, NULL);

    ret = iflistsearch(&ifl, "eth0");
    ck_assert_int_eq(ret, 0);

    ret = iflistadd(&ifl, "eth0", 0, 0);
    ck_assert_int_eq(ret, 1);

    ret = iflistsearch(&ifl, "eth0");
    ck_assert_int_eq(ret, 1);

    ret = iflistadd(&ifl, "eth1", 0, 1);
    ck_assert_int_eq(ret, 1);

    ret = iflistsearch(&ifl, "eth0");
    ck_assert_int_eq(ret, 1);

    ret = iflistadd(&ifl, "eth0", 0, 2);
    ck_assert_int_eq(ret, 1);

    ret = iflistsearch(&ifl, "eth0");
    ck_assert_int_eq(ret, 1);

    ret = iflistadd(&ifl, "eth2", 0, 10);
    ck_assert_int_eq(ret, 1);

    ret = iflistsearch(&ifl, "eth0");
    ck_assert_int_eq(ret, 1);

    ret = iflistsearch(&ifl, "eth2");
    ck_assert_int_eq(ret, 1);

    ret = iflistsearch(&ifl, "eth0");
    ck_assert_int_eq(ret, 1);

    ret = iflistsearch(&ifl, "eth1");
    ck_assert_int_eq(ret, 1);

    ret = iflistsearch(&ifl, "eth3");
    ck_assert_int_eq(ret, 0);

    ret = iflistadd(&ifl, "eth3", 0, 0);
    ck_assert_int_eq(ret, 1);

    ret = iflistsearch(&ifl, "eth3");
    ck_assert_int_eq(ret, 1);

    ret = iflistadd(&ifl, "eth4", 0, 0);
    ck_assert_int_eq(ret, 1);

    ret = iflistsearch(&ifl, "eth3");
    ck_assert_int_eq(ret, 1);

    iflistfree(&ifl);
    ck_assert_ptr_eq(ifl, NULL);
}
END_TEST

void add_iflist_tests(Suite *s)
{
	TCase *tc_iflist = tcase_create("Iflist");
	tcase_add_checked_fixture(tc_iflist, setup, teardown);
	tcase_add_unchecked_fixture(tc_iflist, setup, teardown);
	tcase_add_test(tc_iflist, iflistfree_can_free_null);
    tcase_add_test(tc_iflist, iflistadd_can_add);
    tcase_add_test(tc_iflist, iflistsearch_can_search);
	suite_add_tcase(s, tc_iflist);
}
