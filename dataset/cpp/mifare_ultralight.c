/*
 * This implementation was written based on information provided by the
 * following documents:
 *
 * Contactless Single-trip Ticket IC
 * MF0 IC U1
 * Functional Specification
 * Revision 3.0
 * March 2003
 */

#if defined(HAVE_CONFIG_H)
    #include "config.h"
#endif

#if defined(HAVE_SYS_TYPES_H)
    #include <sys/types.h>
#endif

#include <errno.h>
#include <stdlib.h>
#include <string.h>

#ifdef WITH_DEBUG
    #include <libutil.h>
#endif

#include <freefare.h>
#include "freefare_internal.h"

#define ASSERT_VALID_PAGE(tag, page, mode_write) \
    do { \
	if (is_mifare_ultralightc (tag)) { \
	    if (mode_write) { \
		if (page >= MIFARE_ULTRALIGHT_C_PAGE_COUNT) return errno = EINVAL, -1; \
	    } else { \
		if (page >= MIFARE_ULTRALIGHT_C_PAGE_COUNT_READ) return errno = EINVAL, -1; \
	    } \
	} else { \
	    if (page >= MIFARE_ULTRALIGHT_PAGE_COUNT) return errno = EINVAL, -1; \
	} \
    } while (0)

#define ULTRALIGHT_TRANSCEIVE(tag, msg, res) \
    do { \
	errno = 0; \
	DEBUG_XFER (msg, __##msg##_n, "===> "); \
	int _res; \
	if ((_res = nfc_initiator_transceive_bytes (tag->device, msg, __##msg##_n, res, __##res##_size, 0)) < 0) { \
	    return errno = EIO, -1; \
	} \
	__##res##_n = _res; \
	DEBUG_XFER (res, __##res##_n, "<=== "); \
    } while (0)

#define ULTRALIGHT_TRANSCEIVE_RAW(tag, msg, res) \
    do { \
	errno = 0; \
	if (nfc_device_set_property_bool (tag->device, NP_EASY_FRAMING, false) < 0) { \
	    errno = EIO; \
	    return -1; \
	} \
	DEBUG_XFER (msg, __##msg##_n, "===> "); \
	int _res; \
	if ((_res = nfc_initiator_transceive_bytes (tag->device, msg, __##msg##_n, res, __##res##_size, 0)) < 0) { \
	    nfc_device_set_property_bool (tag->device, NP_EASY_FRAMING, true); \
	    return errno = EIO, -1; \
	} \
	__##res##_n = _res; \
	DEBUG_XFER (res, __##res##_n, "<=== "); \
	if (nfc_device_set_property_bool (tag->device, NP_EASY_FRAMING, true) < 0) { \
	    errno = EIO; \
	    return -1; \
	} \
    } while (0)

static bool
taste(nfc_target target)
{
    return target.nm.nmt == NMT_ISO14443A && target.nti.nai.btSak == 0x00;
}

bool
mifare_ultralight_taste(nfc_device *device, nfc_target target)
{
    return taste(target) && !is_mifare_ultralightc_on_reader(device, target.nti.nai);
}

bool
mifare_ultralightc_taste(nfc_device *device, nfc_target target)
{
    return taste(target) && is_mifare_ultralightc_on_reader(device, target.nti.nai);
}


/*
 * Memory management functions.
 */

/*
 * Allocates and initialize a MIFARE UltraLight tag.
 */
static FreefareTag
_mifare_ultralightc_tag_new(nfc_device *device, nfc_target target, bool is_ultralightc)
{
    FreefareTag tag;

    if ((tag = malloc(sizeof(struct mifare_ultralight_tag)))) {
	tag->type = (is_ultralightc) ? MIFARE_ULTRALIGHT_C : MIFARE_ULTRALIGHT;
	tag->free_tag = mifare_ultralightc_tag_free;
	tag->device = device;
	tag->info = target;
	tag->active = 0;
    }

    return tag;
}

FreefareTag
mifare_ultralight_tag_new(nfc_device *device, nfc_target target)
{
    return _mifare_ultralightc_tag_new(device, target, false);
}

FreefareTag
mifare_ultralightc_tag_new(nfc_device *device, nfc_target target)
{
    return _mifare_ultralightc_tag_new(device, target, true);
}

/*
 * Free the provided tag.
 */
void
mifare_ultralight_tag_free(FreefareTag tag)
{
    free(tag);
}

void
mifare_ultralightc_tag_free(FreefareTag tag)
{
    mifare_ultralight_tag_free(tag);
}


/*
 * MIFARE card communication preparation functions
 *
 * The following functions send NFC commands to the initiator to prepare
 * communication with a MIFARE card, and perform required cleanups after using
 * the target.
 */


/*
 * Establish connection to the provided tag.
 */
int
mifare_ultralight_connect(FreefareTag tag)
{
    ASSERT_INACTIVE(tag);

    nfc_target pnti;
    nfc_modulation modulation = {
	.nmt = NMT_ISO14443A,
	.nbr = NBR_106
    };
    if (nfc_initiator_select_passive_target(tag->device, modulation, tag->info.nti.nai.abtUid, tag->info.nti.nai.szUidLen, &pnti) >= 0) {
	tag->active = 1;
	for (int i = 0; i < MIFARE_ULTRALIGHT_MAX_PAGE_COUNT; i++)
	    MIFARE_ULTRALIGHT(tag)->cached_pages[i] = 0;
    } else {
	errno = EIO;
	return -1;
    }
    return 0;
}

/*
 * Terminate connection with the provided tag.
 */
int
mifare_ultralight_disconnect(FreefareTag tag)
{
    ASSERT_ACTIVE(tag);

    if (nfc_initiator_deselect_target(tag->device) >= 0) {
	tag->active = 0;
    } else {
	errno = EIO;
	return -1;
    }
    return 0;
}


/*
 * Card manipulation functions
 *
 * The following functions perform direct communication with the connected
 * MIFARE UltraLight tag.
 */

/*
 * Read data from the provided MIFARE tag.
 */
int
mifare_ultralight_read(FreefareTag tag, MifareUltralightPageNumber page, MifareUltralightPage *data)
{
    ASSERT_ACTIVE(tag);
    ASSERT_VALID_PAGE(tag, page, false);

    if (!MIFARE_ULTRALIGHT(tag)->cached_pages[page]) {
	BUFFER_INIT(cmd, 2);
	BUFFER_ALIAS(res, MIFARE_ULTRALIGHT(tag)->cache[page], sizeof(MifareUltralightPage) * 4);

	BUFFER_APPEND(cmd, 0x30);
	BUFFER_APPEND(cmd, page);

	ULTRALIGHT_TRANSCEIVE(tag, cmd, res);

	/* Handle wrapped pages */
	int iPageCount;
	if (is_mifare_ultralightc(tag)) {
	    iPageCount = MIFARE_ULTRALIGHT_C_PAGE_COUNT_READ;
	} else {
	    iPageCount = MIFARE_ULTRALIGHT_PAGE_COUNT;
	}
	for (int i = iPageCount; i <= page + 3; i++) {
	    memcpy(MIFARE_ULTRALIGHT(tag)->cache[i % iPageCount], MIFARE_ULTRALIGHT(tag)->cache[i], sizeof(MifareUltralightPage));
	}

	/* Mark pages as cached */
	for (int i = page; i <= page + 3; i++) {
	    MIFARE_ULTRALIGHT(tag)->cached_pages[i % iPageCount] = 1;
	}
    }

    memcpy(data, MIFARE_ULTRALIGHT(tag)->cache[page], sizeof(*data));
    return 0;
}

/*
 * Read data to the provided MIFARE tag.
 */
int
mifare_ultralight_write(FreefareTag tag, const MifareUltralightPageNumber page, const MifareUltralightPage data)
{
    ASSERT_ACTIVE(tag);
    ASSERT_VALID_PAGE(tag, page, true);

    BUFFER_INIT(cmd, 6);
    BUFFER_INIT(res, 1);

    BUFFER_APPEND(cmd, 0xA2);
    BUFFER_APPEND(cmd, page);
    BUFFER_APPEND_BYTES(cmd, data, sizeof(MifareUltralightPage));

    ULTRALIGHT_TRANSCEIVE(tag, cmd, res);

    /* Invalidate page in cache */
    MIFARE_ULTRALIGHT(tag)->cached_pages[page] = 0;

    return 0;
}

/*
 * Authenticate to the provided MIFARE tag.
 */
int
mifare_ultralightc_authenticate(FreefareTag tag, const MifareDESFireKey key)
{
    ASSERT_ACTIVE(tag);

    BUFFER_INIT(cmd1, 2);
    BUFFER_INIT(res, 9);
    BUFFER_APPEND(cmd1, 0x1A);
    BUFFER_APPEND(cmd1, 0x00);

    ULTRALIGHT_TRANSCEIVE_RAW(tag, cmd1, res);

    uint8_t PICC_E_RndB[8];
    memcpy(PICC_E_RndB, res + 1, 8);

    uint8_t PICC_RndB[8];
    memcpy(PICC_RndB, PICC_E_RndB, 8);
    uint8_t ivect[8];
    memset(ivect, '\0', sizeof(ivect));
    mifare_cypher_single_block(key, PICC_RndB, ivect, MCD_RECEIVE, MCO_DECYPHER, 8);

    uint8_t PCD_RndA[8];
    DES_random_key((DES_cblock *)&PCD_RndA);

    uint8_t PCD_r_RndB[8];
    memcpy(PCD_r_RndB, PICC_RndB, 8);
    rol(PCD_r_RndB, 8);

    uint8_t token[16];
    memcpy(token, PCD_RndA, 8);
    memcpy(token + 8, PCD_r_RndB, 8);
    size_t offset = 0;

    while (offset < 16) {
	mifare_cypher_single_block(key, token + offset, ivect, MCD_SEND, MCO_ENCYPHER, 8);
	offset += 8;
    }

    BUFFER_INIT(cmd2, 17);

    BUFFER_APPEND(cmd2, 0xAF);
    BUFFER_APPEND_BYTES(cmd2, token, 16);

    ULTRALIGHT_TRANSCEIVE_RAW(tag, cmd2, res);

    uint8_t PICC_E_RndA_s[8];
    memcpy(PICC_E_RndA_s, res + 1, 8);

    uint8_t PICC_RndA_s[8];
    memcpy(PICC_RndA_s, PICC_E_RndA_s, 8);
    mifare_cypher_single_block(key, PICC_RndA_s, ivect, MCD_RECEIVE, MCO_DECYPHER, 8);

    uint8_t PCD_RndA_s[8];
    memcpy(PCD_RndA_s, PCD_RndA, 8);
    rol(PCD_RndA_s, 8);

    if (0 != memcmp(PCD_RndA_s, PICC_RndA_s, 8)) {
	errno = EACCES;
	return -1;
    }
    // XXX Should we store the state "authenticated" in the tag struct??
    return 0;
}

/*
 * Set the authentication key for the provided MIFARE tag.
 */
int
mifare_ultralightc_set_key(FreefareTag tag, MifareDESFireKey key)
{
    MifareUltralightPage data;

    if (key->type != MIFARE_KEY_2K3DES) {
	errno = EINVAL;
	return -1;
    }

    data[0] = key->data[7];
    data[1] = key->data[6];
    data[2] = key->data[5];
    data[3] = key->data[4];

    if (mifare_ultralight_write(tag, 0x2C, data)<0) {
	return -1;
    }

    data[0] = key->data[3];
    data[1] = key->data[2];
    data[2] = key->data[1];
    data[3] = key->data[0];

    if (mifare_ultralight_write(tag, 0x2D, data)<0) {
	return -1;
    }

    data[0] = key->data[15];
    data[1] = key->data[14];
    data[2] = key->data[13];
    data[3] = key->data[12];

    if (mifare_ultralight_write(tag, 0x2E, data)<0) {
	return -1;
    }

    data[0] = key->data[11];
    data[1] = key->data[10];
    data[2] = key->data[9];
    data[3] = key->data[8];

    if (mifare_ultralight_write(tag, 0x2F, data)<0) {
	return -1;
    }

    return 0;
}

bool
is_mifare_ultralight(FreefareTag tag)
{
    return tag->type == MIFARE_ULTRALIGHT;
}

bool
is_mifare_ultralightc(FreefareTag tag)
{
    return tag->type == MIFARE_ULTRALIGHT_C;
}

/*
 * Callback for freefare_tag_new to test presence of a MIFARE UltralightC on the reader.
 */
bool
is_mifare_ultralightc_on_reader(nfc_device *device, nfc_iso14443a_info nai)
{
    int ret;
    uint8_t cmd_step1[2];
    uint8_t res_step1[9];
    cmd_step1[0] = 0x1A;
    cmd_step1[1] = 0x00;

    nfc_target pnti;
    nfc_modulation modulation = {
	.nmt = NMT_ISO14443A,
	.nbr = NBR_106
    };
    nfc_initiator_select_passive_target(device, modulation, nai.abtUid, nai.szUidLen, &pnti);
    nfc_device_set_property_bool(device, NP_EASY_FRAMING, false);
    ret = nfc_initiator_transceive_bytes(device, cmd_step1, sizeof(cmd_step1), res_step1, sizeof(res_step1), 0);
    nfc_device_set_property_bool(device, NP_EASY_FRAMING, true);
    nfc_initiator_deselect_target(device);
    return ret >= 0;
}
