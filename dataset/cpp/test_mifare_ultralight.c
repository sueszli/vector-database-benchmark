#include <cutter.h>
#include <errno.h>
#include <string.h>

#include <freefare.h>
#include "freefare_internal.h"

#include "fixture.h"

void
test_mifare_ultralight_write(void)
{
    int res;

    MifareUltralightPage initial;
    MifareUltralightPage page;
    MifareUltralightPage payload1  = { 0x12, 0x34, 0x56, 0x78 };
    MifareUltralightPage payload2  = { 0xaa, 0x55, 0x00, 0xff };

    MifareUltralightPageNumber n = 7;

    /* Read and save current value (should be { 0x00 0x00 0x00 0x00 }) */
    res = mifare_ultralight_read(tag, n, &initial);
    cut_assert_equal_int(0, res, cut_message("mifare_ultralight_read() failed"));

    /* Write payload1 */
    res = mifare_ultralight_write(tag, n, payload1);
    cut_assert_equal_int(0, res, cut_message("mifare_ultralight_write() failed"));

    /* Check it */
    res = mifare_ultralight_read(tag, n, &page);
    cut_assert_equal_int(0, res, cut_message("mifare_ultralight_read() failed"));
    cut_assert_equal_memory(payload1, sizeof(payload1), page, sizeof(page), cut_message("Wrong data"));

    /* Write payload2 */
    res = mifare_ultralight_write(tag, n, payload2);
    cut_assert_equal_int(0, res, cut_message("mifare_ultralight_write() failed"));

    /* Check it */
    res = mifare_ultralight_read(tag, n, &page);
    cut_assert_equal_int(0, res, cut_message("mifare_ultralight_read() failed"));
    cut_assert_equal_memory(payload2, sizeof(payload2), page, sizeof(page), cut_message("Wrong data"));

    /* Write initial data */
    res = mifare_ultralight_write(tag, n, initial);
    cut_assert_equal_int(0, res, cut_message("mifare_ultralight_write() failed"));

    /* While here check it (no reason to fail since the rest of the test passed) */
    res = mifare_ultralight_read(tag, n, &page);
    cut_assert_equal_int(0, res, cut_message("mifare_ultralight_read() failed"));
    cut_assert_equal_memory(initial, sizeof(initial), page, sizeof(page), cut_message("Wrong data"));
}

void
test_mifare_ultralight_invalid_page(void)
{
    int res;
    MifareUltralightPage page = { 0x00, 0x00, 0x00, 0x00 };

    int invalid_page;
    if (is_mifare_ultralightc(tag)) {
	invalid_page = MIFARE_ULTRALIGHT_C_PAGE_COUNT;
    } else {
	invalid_page = MIFARE_ULTRALIGHT_PAGE_COUNT;
    }
    res = mifare_ultralight_read(tag, invalid_page, &page);
    cut_assert_equal_int(-1, res, cut_message("mifare_ultralight_read() succeeded"));
    cut_assert_equal_int(EINVAL, errno, cut_message("Wrong errno value"));

    res = mifare_ultralight_write(tag, invalid_page, page);
    cut_assert_equal_int(-1, res, cut_message("mifare_ultralight_write() succeeded"));
    cut_assert_equal_int(EINVAL, errno, cut_message("Wrong errno value"));
}

void
test_mifare_ultralight_cache(void)
{
    int res;
    MifareUltralightPage page;

    res = mifare_ultralight_read(tag, 0, &page);
    cut_assert_equal_int(0, res, cut_message("mifare_ultralight_read() failed"));

    /* Check cached pages consistency */
    for (int i = 0; i <= 3; i++) {
	cut_assert_equal_int(1, MIFARE_ULTRALIGHT(tag)->cached_pages[i], cut_message("Wrong page cache value for tag->cached_pages[%d]", i));
    }
    for (int i = 4; i < MIFARE_ULTRALIGHT_PAGE_COUNT; i++) {
	cut_assert_equal_int(0, MIFARE_ULTRALIGHT(tag)->cached_pages[i], cut_message("Wrong page cache value for tag->cached_pages[%d]", i));
    }
}

void
test_mifare_ultralight_cache_hit(void)
{
    int res;

    MifareUltralightPage page1;
    MifareUltralightPage page2;

    res = mifare_ultralight_read(tag, 0, &page1);
    cut_assert_equal_int(0, res, cut_message("mifare_ultralight_read() failed"));

    res = mifare_ultralight_read(tag, 0, &page2);
    cut_assert_equal_int(0, res, cut_message("mifare_ultralight_read() failed"));
    cut_assert_equal_memory(page1, sizeof(page1), page2, sizeof(page2), cut_message("Wrong cached data"));
}


void
test_mifare_ultralight_cache_wrap(void)
{
    int res;
    MifareUltralightPage page;
    int last_page;
    if (is_mifare_ultralightc(tag)) {
	// Last 4 blocks are for 3DES key and cannot be read, read will wrap from 0x2b
	last_page = MIFARE_ULTRALIGHT_C_PAGE_COUNT_READ - 1;
	// Actually engineering samples require auth to read above page 0x28 so we skip the test entirely
	cut_omit("mifare_ultralight_read() on last page skipped on UltralightC");
    } else {
	last_page = MIFARE_ULTRALIGHT_PAGE_COUNT - 1;
    }
    res = mifare_ultralight_read(tag, last_page, &page);
    cut_assert_equal_int(0, res, cut_message("mifare_ultralight_read() failed"));

    /* Check cached pages consistency */
    for (int i = 0; i <= 2; i++) {
	cut_assert_equal_int(1, MIFARE_ULTRALIGHT(tag)->cached_pages[i], cut_message("Wrong page cache value for tag->cached_pages[%d]", i));
    }
    for (int i = 3; i < last_page; i++) {
	cut_assert_equal_int(0, MIFARE_ULTRALIGHT(tag)->cached_pages[i], cut_message("Wrong page cache value for tag->cached_pages[%d]", i));
    }
    cut_assert_equal_int(1, MIFARE_ULTRALIGHT(tag)->cached_pages[last_page], cut_message("Wrong page cache value for tag->cached_pages[%d]", last_page));
}

void
test_mifare_ultralight_get_uid(void)
{
    char *uid;

    uid = freefare_get_tag_uid(tag);

    cut_assert_not_null(uid, cut_message("mifare_ultralight_get_uid() failed"));
    cut_assert_equal_int(14, strlen(uid), cut_message("Wrong UID length"));

    free(uid);
}

void
test_mifare_ultralight_tag_friendly_name(void)
{
    const char *name = freefare_get_tag_friendly_name(tag);

    cut_assert_not_null(name, cut_message("freefare_get_tag_friendly_name() failed"));
}

void
test_mifare_ultralightc_authenticate(void)
{
    int res;
    MifareDESFireKey key;

    if (is_mifare_ultralightc(tag)) {
	uint8_t key1_3des_data[16] = { 0x49, 0x45, 0x4D, 0x4B, 0x41, 0x45, 0x52, 0x42, 0x21, 0x4E, 0x41, 0x43, 0x55, 0x4F, 0x59, 0x46 };
	key = mifare_desfire_3des_key_new(key1_3des_data);
	res = mifare_ultralightc_authenticate(tag, key);
	cut_assert_equal_int(0, res, cut_message("mifare_ultralightc_authenticate() failed"));
	mifare_desfire_key_free(key);

	MifareUltralightPage page;
	int last_page = MIFARE_ULTRALIGHT_C_PAGE_COUNT_READ - 1;
	res = mifare_ultralight_read(tag, last_page, &page);
	cut_assert_equal_int(0, res, cut_message("mifare_ultralight_read() failed"));
    } else {
	cut_omit("mifare_ultralightc_authenticate() skipped on Ultralight");
    }
}

void
test_mifare_ultralightc_set_key(void)
{
    int res;
    const uint8_t default_key_data[16] = { 0x49, 0x45, 0x4D, 0x4B, 0x41, 0x45, 0x52, 0x42, 0x21, 0x4E, 0x41, 0x43, 0x55, 0x4F, 0x59, 0x46 };
    const uint8_t test_key_data[16] = { 0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0xAA, 0xBB, 0xCC, 0xDD, 0XEE, 0xFF };
    MifareDESFireKey default_key;
    MifareDESFireKey test_key;

    if (is_mifare_ultralightc(tag)) {
	default_key = mifare_desfire_3des_key_new(default_key_data);
	test_key = mifare_desfire_3des_key_new(test_key_data);

	// Change the key to the new key
	res = mifare_ultralightc_set_key(tag, test_key);
	cut_assert_equal_int(0, res, cut_message("mifare_ultralightc_set_key() failed"));

	res = mifare_ultralight_disconnect(tag);
	cut_assert_equal_int(0, res, cut_message("mifare_ultralight_disconnect() failed"));

	res = mifare_ultralight_connect(tag);
	cut_assert_equal_int(0, res, cut_message("mifare_ultralight_connect() failed"));

	// Test the new key
	res = mifare_ultralightc_authenticate(tag, test_key);
	cut_assert_equal_int(0, res, cut_message("mifare_ultralightc_authenticate() failed"));

	// Change the key back to the default key
	res = mifare_ultralightc_set_key(tag, default_key);
	cut_assert_equal_int(0, res, cut_message("mifare_ultralightc_set_key() failed"));

	res = mifare_ultralight_disconnect(tag);
	cut_assert_equal_int(0, res, cut_message("mifare_ultralight_disconnect() failed"));

	res = mifare_ultralight_connect(tag);
	cut_assert_equal_int(0, res, cut_message("mifare_ultralight_connect() failed"));

	// Test the default key
	res = mifare_ultralightc_authenticate(tag, default_key);
	cut_assert_equal_int(0, res, cut_message("mifare_ultralightc_authenticate() failed"));

	mifare_desfire_key_free(default_key);
	mifare_desfire_key_free(test_key);
    } else {
	cut_omit("mifare_ultralightc_set_key() skipped on Ultralight");
    }
}
