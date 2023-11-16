/*
 * This implementation was written based on information provided by the
 * following documents:
 *
 * Draft ISO/IEC FCD 14443-4
 * Identification cards
 *   - Contactless integrated circuit(s) cards
 *     - Proximity cards
 *       - Part 4: Transmission protocol
 * Final Committee Draft - 2000-03-10
 *
 * http://ridrix.wordpress.com/2009/09/19/mifare-desfire-communication-example/
 */

#if defined(HAVE_CONFIG_H)
    #include "config.h"
#endif

#if defined(HAVE_SYS_TYPES_H)
    #include <sys/types.h>
#endif

#if defined(HAVE_SYS_ENDIAN_H)
    #include <sys/endian.h>
#endif

#if defined(HAVE_ENDIAN_H)
    #include <endian.h>
#endif

#if defined(HAVE_COREFOUNDATION_COREFOUNDATION_H)
    #include <CoreFoundation/CoreFoundation.h>
#endif

#if defined(HAVE_BYTESWAP_H)
    #include <byteswap.h>
#endif

#include <errno.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>

#ifdef WITH_DEBUG
    #include <libutil.h>
#endif

#include <openssl/rand.h>

#include <freefare.h>
#include "freefare_internal.h"

#pragma pack (push)
#pragma pack (1)
struct mifare_desfire_raw_file_settings {
    uint8_t file_type;
    uint8_t communication_settings;
    uint16_t access_rights;
    union {
	struct {
	    uint8_t file_size[3];
	} standard_file;
	struct {
	    int32_t lower_limit;
	    int32_t upper_limit;
	    int32_t limited_credit_value;
	    uint8_t limited_credit_enabled;
	} value_file;
	struct {
	    uint8_t record_size[3];
	    uint8_t max_number_of_records[3];
	    uint8_t current_number_of_records[3];
	} linear_record_file;
    } settings;
};
#pragma pack (pop)

#define MAX_APPLICATION_COUNT 28
#define MAX_FILE_COUNT 32

#define CMAC_LENGTH 8

static struct mifare_desfire_file_settings cached_file_settings[MAX_FILE_COUNT];
static bool cached_file_settings_current[MAX_FILE_COUNT];

static int	 desfire_transceive(FreefareTag tag, const uint8_t *msg, size_t msg_len, uint8_t *res, size_t res_size, size_t *res_len);
static int	 authenticate(FreefareTag tag, uint8_t cmd, uint8_t key_no, MifareDESFireKey key);
static int	 create_file1(FreefareTag tag, uint8_t command, uint8_t file_no, int has_iso_file_id, uint16_t iso_file_id, uint8_t communication_settings, uint16_t access_rights, uint32_t file_size);
static int	 create_file2(FreefareTag tag, uint8_t command, uint8_t file_no, int has_iso_file_id, uint16_t iso_file_id, uint8_t communication_settings, uint16_t access_rights, uint32_t record_size, uint32_t max_number_of_records);
static ssize_t	 write_data(FreefareTag tag, uint8_t command, uint8_t file_no, off_t offset, size_t length, const void *data, int cs);
static ssize_t	 read_data(FreefareTag tag, uint8_t command, uint8_t file_no, off_t offset, size_t length, void *data, int cs);

#define NOT_YET_AUTHENTICATED 255

#define ASSERT_AUTHENTICATED(tag) \
    do { \
	if (MIFARE_DESFIRE (tag)->authenticated_key_no == NOT_YET_AUTHENTICATED) { \
	    return errno = EINVAL, -1;\
	} \
    } while (0)

/*
 * XXX: cs < 0 is a CommunicationSettings detection error. Other values are
 * user errors. We may need to distinguish them.
 */
#define ASSERT_CS(cs) \
    do { \
	if (cs < 0) { \
	    return errno = EINVAL, -1; \
	} else if (cs == 0x02) { \
	    return errno = EINVAL, -1; \
	} else if (cs > 0x03) { \
	    return errno = EINVAL, -1; \
	} \
    } while (0)

#define ASSERT_NOT_NULL(argument) \
    do { \
	if (!argument) { \
	    return errno = EINVAL, -1; \
	} \
    } while (0)




/* Native DESFire APDUs will be wrapped in ISO7816-4 APDUs:

   CAPDUs will be 5 bytes longer (CLA+P1+P2+Lc+Le)
   RAPDUs will be 1 byte longer  (SW1 SW2 instead of 1 status byte)

   Max APDU sizes to be ISO encapsulated by desfire_transceive()
   From MIFARE DESFire Functional specification:
   MAX_CAPDU_SIZE:   "The length of the total wrapped DESFire
                      command is not longer than 55 byte long."
   MAX_RAPDU_SIZE:   1 status byte + 59 bytes
 */
#define MAX_CAPDU_SIZE 55
#define MAX_RAPDU_SIZE 60

/*
 * Transmit the message msg to the NFC tag and receive the response res.  The
 * response buffer's size is set according to the quantity of data received.
 *
 * The Mifare DESFire function return value which is returned at the end of the
 * response is copied at the beginning to match the PICC documentation.
 */
static int
desfire_transceive(FreefareTag tag, const uint8_t *msg, size_t msg_len, uint8_t *res, size_t res_size, size_t *res_len)
{
    uint8_t msg_buf[MAX_CAPDU_SIZE + 5];
    uint8_t res_buf[MAX_RAPDU_SIZE + 1];

    size_t len = 5;
    int rc;

    if (!msg) {
	errno = EINVAL;
	return -1;
    }

    memset (msg_buf, 0, sizeof(msg_buf));
    msg_buf[0] = 0x90; // CLA
    msg_buf[1] = msg[0]; // INS
    // P1 & P2 is zero by default

    if (msg_len > 1) {
	len += msg_len;
	msg_buf[4] = msg_len - 1; // Lc
	memcpy (msg_buf + 5, msg + 1, msg_len - 1);
    }

    MIFARE_DESFIRE (tag)->last_picc_error = OPERATION_OK;
    MIFARE_DESFIRE (tag)->last_pcd_error = OPERATION_OK;

    DEBUG_XFER (msg_buf, len, "===> ");

    if ((rc = nfc_initiator_transceive_bytes (tag->device, msg_buf, len, res_buf, sizeof(res_buf), tag->timeout)) < 2) {
	errno = (errno == ETIMEDOUT) ? errno : EIO;
	return -1;
    }

    DEBUG_XFER (res_buf, rc, "<=== ");

    rc--;
    if (rc > (int)res_size) {
	errno = ENOBUFS;
	return -1;
    }

    int lpe = res_buf[rc];

    res[rc - 1] = lpe;

    if ((1 == rc) && (ADDITIONAL_FRAME != res_buf[rc]) && (OPERATION_OK != res_buf[rc])) {

	if (AUTHENTICATION_ERROR == lpe)
	    errno = EACCES;

	MIFARE_DESFIRE (tag)->last_picc_error = lpe;
	return -1;
    }

    if (res_len)
	*res_len = rc;
    memcpy (res, res_buf, rc - 1);

    errno = 0;
    return 0;
}


/*
 * Convenience macros.
 */

#ifdef WITH_DEBUG
// define a small but cute wrapper to allow debug output
#define MIFARE_DESFIRE_TRANSCEIVE(tag, msg, msg_len, res, res_size, res_len) ({ \
	int __err = 0; \
	do { \
	    DEBUG_FUNCTION(); \
	    __err = desfire_transceive(tag, msg, msg_len, res, res_size, res_len); \
	} while (0); \
	__err; \
    })
#else
#define MIFARE_DESFIRE_TRANSCEIVE desfire_transceive
#endif

/*
 * Miscellaneous low-level memory manipulation functions.
 */

static int32_t	 le24toh(uint8_t data[3]);

static int
madame_soleil_get_read_communication_settings(FreefareTag tag, uint8_t file_no)
{
    struct mifare_desfire_file_settings settings;
    if (mifare_desfire_get_file_settings(tag, file_no, &settings))
	return -1;

    if ((MIFARE_DESFIRE(tag)->authenticated_key_no == MDAR_READ(settings.access_rights)) ||
	(MIFARE_DESFIRE(tag)->authenticated_key_no == MDAR_READ_WRITE(settings.access_rights)))
	return settings.communication_settings;
    else
	return 0;
}

static int
madame_soleil_get_write_communication_settings(FreefareTag tag, uint8_t file_no)
{
    struct mifare_desfire_file_settings settings;
    if (mifare_desfire_get_file_settings(tag, file_no, &settings))
	return -1;

    if ((MIFARE_DESFIRE(tag)->authenticated_key_no == MDAR_WRITE(settings.access_rights)) ||
	(MIFARE_DESFIRE(tag)->authenticated_key_no == MDAR_READ_WRITE(settings.access_rights)))
	return settings.communication_settings;
    else
	return 0;
}

static int32_t
le24toh(uint8_t data[3])
{
    return (data[2] << 16) | (data[1] << 8) | data[0];
}

bool
mifare_desfire_taste(nfc_device *device, nfc_target target)
{
    // We have three different ATS prefixes to
    // check for, standalone, JCOP and JCOP3.
    static const char STANDALONE_DESFIRE[] = { 0x75, 0x77, 0x81, 0x02 };
    static const char JCOP_DESFIRE[] = { 0x75, 0xf7, 0xb1, 0x02 };
    static const char JCOP3_DESFIRE[] = { 0x78, 0x77, 0x71, 0x02 };

    (void) device;

    return target.nm.nmt == NMT_ISO14443A &&
      target.nti.nai.btSak == 0x20 && ((
	   target.nti.nai.szAtsLen >= 5 && (
	       memcmp(target.nti.nai.abtAts, STANDALONE_DESFIRE, sizeof(STANDALONE_DESFIRE)) == 0 ||
	       memcmp(target.nti.nai.abtAts, JCOP_DESFIRE, sizeof(JCOP_DESFIRE)) == 0))
	|| (target.nti.nai.szAtsLen == 4 &&
	       memcmp(target.nti.nai.abtAts, JCOP3_DESFIRE, sizeof(JCOP3_DESFIRE)) == 0));
}

/*
 * Memory management functions.
 */

/*
 * Allocates and initialize a MIFARE DESFire tag.
 */
FreefareTag
mifare_desfire_tag_new(nfc_device *device, nfc_target target)
{
    FreefareTag tag;
    if ((tag = malloc(sizeof(struct mifare_desfire_tag)))) {
	MIFARE_DESFIRE(tag)->last_picc_error = OPERATION_OK;
	MIFARE_DESFIRE(tag)->last_pcd_error = OPERATION_OK;
	MIFARE_DESFIRE(tag)->session_key = NULL;
	MIFARE_DESFIRE(tag)->crypto_buffer = NULL;
	MIFARE_DESFIRE(tag)->crypto_buffer_size = 0;
	tag->type = MIFARE_DESFIRE;
	tag->free_tag = mifare_desfire_tag_free;
	tag->device = device;
	tag->info = target;
	tag->active = 0;
    }
    return tag;
}

/*
 * Free the provided tag.
 */
void
mifare_desfire_tag_free(FreefareTag tag)
{
    free(MIFARE_DESFIRE(tag)->session_key);
    free(MIFARE_DESFIRE(tag)->crypto_buffer);
    free(tag);
}

/*
 * MIFARE card communication preparation functions
 *
 * The following functions send NFC commands to the initiator to prepare
 * communication with a MIFARE card, and perform required cleannups after using
 * the target.
 */

/*
 * Establish connection to the provided tag.
 */
int
mifare_desfire_connect(FreefareTag tag)
{
    ASSERT_INACTIVE(tag);

    nfc_target pnti;
    nfc_modulation modulation = {
	.nmt = NMT_ISO14443A,
	.nbr = NBR_424
    };
    if (nfc_initiator_select_passive_target(tag->device, modulation, tag->info.nti.nai.abtUid, tag->info.nti.nai.szUidLen, &pnti) >= 0) {
	// The registered ISO AID of DESFire D2760000850100
	// Selecting this AID selects the MF
	BUFFER_INIT(cmd, 12);
	BUFFER_INIT(res, 2);
	BUFFER_APPEND(cmd, 0x00);
	BUFFER_APPEND(cmd, 0xa4);
	BUFFER_APPEND(cmd, 0x04);
	BUFFER_APPEND(cmd, 0x00);
	uint8_t AID[] = { 0xd2, 0x76, 0x00, 0x00, 0x85, 0x01, 0x00};
	BUFFER_APPEND(cmd, sizeof(AID));
	BUFFER_APPEND_BYTES(cmd, AID, sizeof(AID));
	if ((nfc_initiator_transceive_bytes(tag->device, cmd, BUFFER_SIZE(cmd), res, BUFFER_MAXSIZE(cmd), tag->timeout) < 0) || (res[0] != 0x90 || res[1] != 0x00)) {
	    errno = (errno == ETIMEDOUT) ? errno : EIO;
	    return -1;
	}
	tag->active = 1;
	free(MIFARE_DESFIRE(tag)->session_key);
	MIFARE_DESFIRE(tag)->session_key = NULL;
	MIFARE_DESFIRE(tag)->last_picc_error = OPERATION_OK;
	MIFARE_DESFIRE(tag)->last_pcd_error = OPERATION_OK;
	MIFARE_DESFIRE(tag)->authenticated_key_no = NOT_YET_AUTHENTICATED;
	MIFARE_DESFIRE(tag)->selected_application = 0;
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
mifare_desfire_disconnect(FreefareTag tag)
{
    ASSERT_ACTIVE(tag);

    free(MIFARE_DESFIRE(tag)->session_key);
    MIFARE_DESFIRE(tag)->session_key = NULL;

    if (nfc_initiator_deselect_target(tag->device) >= 0) {
	tag->active = 0;
    }
    return 0;
}


#define AUTHENTICATE_LEGACY 0x0A
#define AUTHENTICATE_ISO 0x1A
#define AUTHENTICATE_AES 0xAA

static int
authenticate(FreefareTag tag, uint8_t cmd, uint8_t key_no, MifareDESFireKey key)
{
    int rc;

    ASSERT_ACTIVE(tag);

    memset(MIFARE_DESFIRE(tag)->ivect, 0, MAX_CRYPTO_BLOCK_SIZE);

    MIFARE_DESFIRE(tag)->authenticated_key_no = NOT_YET_AUTHENTICATED;
    free(MIFARE_DESFIRE(tag)->session_key);
    MIFARE_DESFIRE(tag)->session_key = NULL;

    MIFARE_DESFIRE(tag)->authentication_scheme = (AUTHENTICATE_LEGACY == cmd) ? AS_LEGACY : AS_NEW;

    BUFFER_INIT(cmd1, 2);
    BUFFER_INIT(res, 17);

    BUFFER_APPEND(cmd1, cmd);
    BUFFER_APPEND(cmd1, key_no);

    if ((rc = MIFARE_DESFIRE_TRANSCEIVE(tag, cmd1, __cmd1_n, res, __res_size, &__res_n)) < 0)
	return rc;

    size_t key_length = __res_n - 1;

    uint8_t PICC_E_RndB[16];
    memcpy(PICC_E_RndB, res, key_length);

    uint8_t PICC_RndB[16];
    memcpy(PICC_RndB, PICC_E_RndB, key_length);
    mifare_cypher_blocks_chained(tag, key, MIFARE_DESFIRE(tag)->ivect, PICC_RndB, key_length, MCD_RECEIVE, MCO_DECYPHER);

    uint8_t PCD_RndA[16];
    RAND_bytes(PCD_RndA, 16);

    uint8_t PCD_r_RndB[16];
    memcpy(PCD_r_RndB, PICC_RndB, key_length);
    rol(PCD_r_RndB, key_length);

    uint8_t token[32];
    memcpy(token, PCD_RndA, key_length);
    memcpy(token + key_length, PCD_r_RndB, key_length);

    mifare_cypher_blocks_chained(tag, key, MIFARE_DESFIRE(tag)->ivect, token, 2 * key_length, MCD_SEND, (AUTHENTICATE_LEGACY == cmd) ? MCO_DECYPHER : MCO_ENCYPHER);

    BUFFER_INIT(cmd2, 33);

    BUFFER_APPEND(cmd2, 0xAF);
    BUFFER_APPEND_BYTES(cmd2, token, 2 * key_length);

    if ((rc = MIFARE_DESFIRE_TRANSCEIVE(tag, cmd2, __cmd2_n, res, __res_size, &__res_n)) < 0)
	return rc;

    uint8_t PICC_E_RndA_s[16];
    memcpy(PICC_E_RndA_s, res, key_length);

    uint8_t PICC_RndA_s[16];
    memcpy(PICC_RndA_s, PICC_E_RndA_s, key_length);
    mifare_cypher_blocks_chained(tag, key, MIFARE_DESFIRE(tag)->ivect, PICC_RndA_s, key_length, MCD_RECEIVE, MCO_DECYPHER);

    uint8_t PCD_RndA_s[key_length];
    memcpy(PCD_RndA_s, PCD_RndA, key_length);
    rol(PCD_RndA_s, key_length);


    if (0 != memcmp(PCD_RndA_s, PICC_RndA_s, key_length)) {
#ifdef WITH_DEBUG
	hexdump(PCD_RndA_s, key_length, "PCD  ", 0);
	hexdump(PICC_RndA_s, key_length, "PICC ", 0);
#endif
	errno = EACCES;
	return -1;
    }

    MIFARE_DESFIRE(tag)->authenticated_key_no = key_no;
    MIFARE_DESFIRE(tag)->session_key = mifare_desfire_session_key_new(PCD_RndA, PICC_RndB, key);
    memset(MIFARE_DESFIRE(tag)->ivect, 0, MAX_CRYPTO_BLOCK_SIZE);

    switch (MIFARE_DESFIRE(tag)->authentication_scheme) {
    case AS_LEGACY:
	break;
    case AS_NEW:
	cmac_generate_subkeys(MIFARE_DESFIRE(tag)->session_key);
	break;
    }

    return 0;
}

int
mifare_desfire_authenticate(FreefareTag tag, uint8_t key_no, MifareDESFireKey key)
{
    switch (key->type) {
    case MIFARE_KEY_DES:
    case MIFARE_KEY_2K3DES:
	return authenticate(tag, AUTHENTICATE_LEGACY, key_no, key);
	break;
    case MIFARE_KEY_3K3DES:
	return authenticate(tag, AUTHENTICATE_ISO, key_no, key);
	break;
    case MIFARE_KEY_AES128:
	return authenticate(tag, AUTHENTICATE_AES, key_no, key);
	break;
    }

    return -1; /* NOTREACHED */
}

int
mifare_desfire_authenticate_iso(FreefareTag tag, uint8_t key_no, MifareDESFireKey key)
{
    return authenticate(tag, AUTHENTICATE_ISO, key_no, key);
}

int
mifare_desfire_authenticate_aes(FreefareTag tag, uint8_t key_no, MifareDESFireKey key)
{
    return authenticate(tag, AUTHENTICATE_AES, key_no, key);
}

int
mifare_desfire_change_key_settings(FreefareTag tag, uint8_t settings)
{
    int rc;

    ASSERT_ACTIVE(tag);
    ASSERT_AUTHENTICATED(tag);

    BUFFER_INIT(cmd, 9 + CMAC_LENGTH);
    BUFFER_INIT(res, 1 + CMAC_LENGTH);

    BUFFER_APPEND(cmd, 0x54);
    BUFFER_APPEND(cmd, settings);

    uint8_t *p = mifare_cryto_preprocess_data(tag, cmd, &__cmd_n, 1, MDCM_ENCIPHERED | ENC_COMMAND);

    if ((rc = MIFARE_DESFIRE_TRANSCEIVE(tag, p, __cmd_n, res, __res_size, &__res_n)) < 0)
	return rc;

    ssize_t n = __res_n;
    p = mifare_cryto_postprocess_data(tag, res, &n, MDCM_PLAIN | CMAC_COMMAND | CMAC_VERIFY | MAC_COMMAND | MAC_VERIFY);

    if (!p)
	return errno = EINVAL, -1;

    return 0;
}

int
mifare_desfire_get_key_settings(FreefareTag tag, uint8_t *settings, uint8_t *max_keys)
{
    int rc;

    ASSERT_ACTIVE(tag);

    BUFFER_INIT(cmd, 1);
    BUFFER_INIT(res, 3 + CMAC_LENGTH);

    BUFFER_APPEND(cmd, 0x45);

    uint8_t *p = mifare_cryto_preprocess_data(tag, cmd, &__cmd_n, 1, MDCM_PLAIN | CMAC_COMMAND);

    if ((rc = MIFARE_DESFIRE_TRANSCEIVE(tag, cmd, __cmd_n, res, __res_size, &__res_n)) < 0)
	return rc;

    ssize_t n = __res_n;
    p = mifare_cryto_postprocess_data(tag, res, &n, MDCM_PLAIN | CMAC_COMMAND | CMAC_VERIFY);

    if (!p)
	return errno = EINVAL, -1;

    if (settings)
	*settings = p[0];
    if (max_keys)
	*max_keys = p[1] & 0x0F;

    return 0;
}

int
mifare_desfire_change_key(FreefareTag tag, uint8_t key_no, MifareDESFireKey new_key, MifareDESFireKey old_key)
{
    int rc;

    ASSERT_ACTIVE(tag);
    ASSERT_AUTHENTICATED(tag);

    BUFFER_INIT(cmd, 42);
    BUFFER_INIT(res, 1 + CMAC_LENGTH);

    key_no &= 0x0F;

    /*
     * Because new crypto methods can be setup only at application creation,
     * changing the card master key to one of them require a key_no tweak.
     */
    if (0x000000 == MIFARE_DESFIRE(tag)->selected_application) {
	switch (new_key->type) {
	case MIFARE_KEY_DES:
	case MIFARE_KEY_2K3DES:
	    break;
	case MIFARE_KEY_3K3DES:
	    key_no |= 0x40;
	    break;
	case MIFARE_KEY_AES128:
	    key_no |= 0x80;
	    break;
	}
    }

    BUFFER_APPEND(cmd, 0xC4);
    BUFFER_APPEND(cmd, key_no);

    int new_key_length;
    switch (new_key->type) {
    case MIFARE_KEY_DES:
    case MIFARE_KEY_2K3DES:
    case MIFARE_KEY_AES128:
	new_key_length = 16;
	break;
    case MIFARE_KEY_3K3DES:
	new_key_length = 24;
	break;
    }

    memcpy(cmd + __cmd_n, new_key->data, new_key_length);

    if ((MIFARE_DESFIRE(tag)->authenticated_key_no & 0x0f) != (key_no & 0x0f)) {
	if (old_key) {
	    for (int n = 0; n < new_key_length; n++) {
		cmd[__cmd_n + n] ^= old_key->data[n];
	    }
	}
    }

    __cmd_n += new_key_length;

    if (new_key->type == MIFARE_KEY_AES128)
	cmd[__cmd_n++] = new_key->aes_version;

    if ((MIFARE_DESFIRE(tag)->authenticated_key_no & 0x0f) != (key_no & 0x0f)) {
	switch (MIFARE_DESFIRE(tag)->authentication_scheme) {
	case AS_LEGACY:
	    iso14443a_crc_append(cmd + 2, __cmd_n - 2);
	    __cmd_n += 2;
	    iso14443a_crc(new_key->data, new_key_length, cmd + __cmd_n);
	    __cmd_n += 2;
	    break;
	case AS_NEW:
	    desfire_crc32_append(cmd, __cmd_n);
	    __cmd_n += 4;

	    desfire_crc32(new_key->data, new_key_length, cmd + __cmd_n);
	    __cmd_n += 4;
	    break;
	}
    } else {
	switch (MIFARE_DESFIRE(tag)->authentication_scheme) {
	case AS_LEGACY:
	    iso14443a_crc_append(cmd + 2, __cmd_n - 2);
	    __cmd_n += 2;
	    break;
	case AS_NEW:
	    desfire_crc32_append(cmd, __cmd_n);
	    __cmd_n += 4;
	    break;
	}
    }

    uint8_t *p = mifare_cryto_preprocess_data(tag, cmd, &__cmd_n, 2, MDCM_ENCIPHERED | ENC_COMMAND | NO_CRC);

    if ((rc = MIFARE_DESFIRE_TRANSCEIVE(tag, p, __cmd_n, res, __res_size, &__res_n)) < 0)
	return rc;

    ssize_t sn = __res_n;
    p = mifare_cryto_postprocess_data(tag, res, &sn, MDCM_PLAIN | CMAC_COMMAND | CMAC_VERIFY);

    if (!p)
	return errno = EINVAL, -1;

    /*
     * If we changed the current authenticated key, we are not authenticated
     * anymore.
     */
    if (key_no == MIFARE_DESFIRE(tag)->authenticated_key_no) {
	free(MIFARE_DESFIRE(tag)->session_key);
	MIFARE_DESFIRE(tag)->session_key = NULL;
    }

    return 0;
}

/*
 * Retrieve version information for a given key.
 */
int
mifare_desfire_get_key_version(FreefareTag tag, uint8_t key_no, uint8_t *version)
{
    int rc;

    ASSERT_ACTIVE(tag);

    ASSERT_NOT_NULL(version);

    BUFFER_INIT(cmd, 2);
    BUFFER_APPEND(cmd, 0x64);
    BUFFER_APPEND(cmd, key_no);

    BUFFER_INIT(res, 2 + CMAC_LENGTH);

    uint8_t *p = mifare_cryto_preprocess_data(tag, cmd, &__cmd_n, 0, MDCM_PLAIN | CMAC_COMMAND);

    if ((rc = MIFARE_DESFIRE_TRANSCEIVE(tag, p, __cmd_n, res, __res_size, &__res_n)) < 0)
	return rc;

    ssize_t sn = __res_n;
    p = mifare_cryto_postprocess_data(tag, res, &sn, MDCM_PLAIN | CMAC_COMMAND | CMAC_VERIFY | MAC_VERIFY);

    if (!p)
	return errno = EINVAL, -1;

    *version = p[0];

    return 0;
}


static int
create_application(FreefareTag tag, MifareDESFireAID aid, uint8_t settings1, uint8_t settings2, int want_iso_application, int want_iso_file_identifiers, uint16_t iso_file_id, uint8_t *iso_file_name, size_t iso_file_name_len)
{
    int rc;

    ASSERT_ACTIVE(tag);

    BUFFER_INIT(cmd, 22);
    BUFFER_INIT(res, 1 + CMAC_LENGTH);

    if (want_iso_file_identifiers)
	settings2 |= 0x20;

    BUFFER_APPEND(cmd, 0xCA);
    BUFFER_APPEND_LE(cmd, aid->data, sizeof(aid->data), sizeof(aid->data));
    BUFFER_APPEND(cmd, settings1);
    BUFFER_APPEND(cmd, settings2);

    if (want_iso_application)
	BUFFER_APPEND_LE(cmd, iso_file_id, sizeof(iso_file_id), sizeof(iso_file_id));

    if (iso_file_name_len)
	BUFFER_APPEND_BYTES(cmd, iso_file_name, iso_file_name_len);

    uint8_t *p = mifare_cryto_preprocess_data(tag, cmd, &__cmd_n, 0, MDCM_PLAIN | CMAC_COMMAND);

    if ((rc = MIFARE_DESFIRE_TRANSCEIVE(tag, p, __cmd_n, res, __res_size, &__res_n)) < 0)
	return rc;

    ssize_t sn = __res_n;
    p = mifare_cryto_postprocess_data(tag, res, &sn, MDCM_PLAIN | CMAC_COMMAND | CMAC_VERIFY | MAC_VERIFY);

    if (!p)
	return errno = EINVAL, -1;

    return 0;
}

int
mifare_desfire_create_application(FreefareTag tag, MifareDESFireAID aid, uint8_t settings, uint8_t key_no)
{
    return create_application(tag, aid, settings, key_no, 0, 0, 0, NULL, 0);
}

int
mifare_desfire_create_application_iso(FreefareTag tag, MifareDESFireAID aid, uint8_t settings, uint8_t key_no, int want_iso_file_identifiers, uint16_t iso_file_id, uint8_t *iso_file_name, size_t iso_file_name_len)
{
    return create_application(tag, aid, settings, key_no, 1, want_iso_file_identifiers, iso_file_id, iso_file_name, iso_file_name_len);
}

int
mifare_desfire_create_application_3k3des(FreefareTag tag, MifareDESFireAID aid, uint8_t settings, uint8_t key_no)
{
    return create_application(tag, aid, settings, APPLICATION_CRYPTO_3K3DES | key_no, 0, 0, 0, NULL, 0);
}

int
mifare_desfire_create_application_3k3des_iso(FreefareTag tag, MifareDESFireAID aid, uint8_t settings, uint8_t key_no, int want_iso_file_identifiers, uint16_t iso_file_id, uint8_t *iso_file_name, size_t iso_file_name_len)
{
    return create_application(tag, aid, settings, APPLICATION_CRYPTO_3K3DES | key_no, 1, want_iso_file_identifiers, iso_file_id, iso_file_name, iso_file_name_len);
}

int
mifare_desfire_create_application_aes(FreefareTag tag, MifareDESFireAID aid, uint8_t settings, uint8_t key_no)
{
    return create_application(tag, aid, settings, APPLICATION_CRYPTO_AES | key_no, 0, 0, 0, NULL, 0);
}

int
mifare_desfire_create_application_aes_iso(FreefareTag tag, MifareDESFireAID aid, uint8_t settings, uint8_t key_no, int want_iso_file_identifiers, uint16_t iso_file_id, uint8_t *iso_file_name, size_t iso_file_name_len)
{
    return create_application(tag, aid, settings, APPLICATION_CRYPTO_AES | key_no, 1, want_iso_file_identifiers, iso_file_id, iso_file_name, iso_file_name_len);
}

int
mifare_desfire_delete_application(FreefareTag tag, MifareDESFireAID aid)
{
    int rc;

    ASSERT_ACTIVE(tag);

    BUFFER_INIT(cmd, 4 + CMAC_LENGTH);
    BUFFER_INIT(res, 1 + CMAC_LENGTH);

    BUFFER_APPEND(cmd, 0xDA);
    BUFFER_APPEND_LE(cmd, aid->data, sizeof(aid->data), sizeof(aid->data));

    uint8_t *p = mifare_cryto_preprocess_data(tag, cmd, &__cmd_n, 0, MDCM_PLAIN | CMAC_COMMAND);

    if ((rc = MIFARE_DESFIRE_TRANSCEIVE(tag, p, __cmd_n, res, __res_size, &__res_n)) < 0)
	return rc;

    ssize_t sn = __res_n;
    p = mifare_cryto_postprocess_data(tag, res, &sn, MDCM_PLAIN | CMAC_COMMAND | CMAC_VERIFY);

    if (!p)
	return errno = EINVAL, -1;

    /*
     * If we have deleted the current application, we are not authenticated
     * anymore.
     */
    if (MIFARE_DESFIRE(tag)->selected_application == (uint32_t)(aid->data[0] | aid->data[1] << 8 | aid->data[2] << 16)) {
	free(MIFARE_DESFIRE(tag)->session_key);
	MIFARE_DESFIRE(tag)->session_key = NULL;
	MIFARE_DESFIRE(tag)->selected_application = 0x000000;
    }

    return 0;
}

int
mifare_desfire_get_application_ids(FreefareTag tag, MifareDESFireAID *aids[], size_t *count)
{
    int rc;

    ASSERT_ACTIVE(tag);

    BUFFER_INIT(cmd, 1);
    BUFFER_INIT(res, MAX_RAPDU_SIZE);

    BUFFER_APPEND(cmd, 0x6A);

    uint8_t buffer[3 * MAX_APPLICATION_COUNT + CMAC_LENGTH + 1];
    *count = 0;

    uint8_t *p = mifare_cryto_preprocess_data(tag, cmd, &__cmd_n, 0, MDCM_PLAIN | CMAC_COMMAND);

    if ((rc = MIFARE_DESFIRE_TRANSCEIVE(tag, p, __cmd_n, res, __res_size, &__res_n)) < 0)
	return rc;

    // FIXME This code needs refactoring!
    memcpy(buffer, res, __res_n);

    if (res[__res_n - 1] == 0xAF) {
	off_t offset = __res_n - 1;
	p[0] = 0xAF;
	if ((rc = MIFARE_DESFIRE_TRANSCEIVE(tag, p, __cmd_n, res, __res_size, &__res_n)) < 0)
	    return rc;

	memcpy((uint8_t *)buffer + offset, res, __res_n);
	__res_n += offset;
    }

    ssize_t sn = __res_n;
    p = mifare_cryto_postprocess_data(tag, buffer, &sn, MDCM_PLAIN | CMAC_COMMAND | CMAC_VERIFY | MAC_VERIFY);

    if (!p)
	return errno = EINVAL, -1;

    *count = (sn - 1) / 3;

    if (!(*aids = malloc((*count + 1) * sizeof(MifareDESFireAID))))
	return -1;

    for (size_t i = 0; i < *count; i++) {
	if (!((*aids)[i] = memdup(p + 3 * i, 3))) {
	    while (i--) {
		free((*aids)[i]);
	    }
	    free(aids);
	    return -1;
	}
    }
    (*aids)[*count] = NULL;

    return 0;
}

int
mifare_desfire_get_df_names(FreefareTag tag, MifareDESFireDF *dfs[], size_t *count)
{
    int rc;

    ASSERT_ACTIVE(tag);

    *count = 0;
    *dfs = NULL;

    BUFFER_INIT(cmd, 1);
    BUFFER_INIT(res, 22 + CMAC_LENGTH);

    BUFFER_APPEND(cmd, 0x6D);

    uint8_t *p = mifare_cryto_preprocess_data(tag, cmd, &__cmd_n, 0, MDCM_PLAIN | CMAC_COMMAND);

    do {
	if ((rc = MIFARE_DESFIRE_TRANSCEIVE(tag, p, __cmd_n, res, __res_size, &__res_n)) < 0)
	    return rc;

	if (__res_n > 1) {
	    MifareDESFireDF *new_dfs;
	    if ((new_dfs = realloc(*dfs, sizeof(*new_dfs) * (*count + 1)))) {
		new_dfs[*count].aid = le24toh(res);
		new_dfs[*count].fid = le16toh(*(uint16_t *)(res + 3));
		memcpy(new_dfs[*count].df_name, res + 5, __res_n - 6);
		new_dfs[*count].df_name_len = __res_n - 6;
		*dfs = new_dfs;
		*count += 1;
	    }
	}

	p[0] = 0XAF;
    } while (res[__res_n - 1] == 0xAF);

    return 0;
}

void
mifare_desfire_free_application_ids(MifareDESFireAID aids[])
{
    for (int i = 0; aids[i]; i++)
	free(aids[i]);
    free(aids);
}

/*
 * Select the application specified by aid for further operation.  If aid is
 * NULL, the master application is selected (equivalent to aid = 0x00000).
 */
int
mifare_desfire_select_application(FreefareTag tag, MifareDESFireAID aid)
{
    int rc;

    ASSERT_ACTIVE(tag);

    struct mifare_desfire_aid null_aid = { .data = { 0x00, 0x00, 0x00 } };

    if (!aid) {
	aid = &null_aid;
    }

    BUFFER_INIT(cmd, 4 + CMAC_LENGTH);
    BUFFER_INIT(res, 1 + CMAC_LENGTH);

    BUFFER_APPEND(cmd, 0x5A);
    BUFFER_APPEND_LE(cmd, aid->data, sizeof(aid->data), sizeof(aid->data));

    uint8_t *p = mifare_cryto_preprocess_data(tag, cmd, &__cmd_n, 0, MDCM_PLAIN | CMAC_COMMAND);

    if ((rc = MIFARE_DESFIRE_TRANSCEIVE(tag, p, __cmd_n, res, __res_size, &__res_n)) < 0)
	return rc;

    ssize_t sn = __res_n;
    p = mifare_cryto_postprocess_data(tag, res, &sn, MDCM_PLAIN | CMAC_COMMAND);

    if (!p)
	return errno = EINVAL, -1;

    for (int n = 0; n < MAX_FILE_COUNT; n++)
	cached_file_settings_current[n] = false;

    free(MIFARE_DESFIRE(tag)->session_key);
    MIFARE_DESFIRE(tag)->session_key = NULL;

    MIFARE_DESFIRE(tag)->selected_application = aid->data[0] | aid->data[1] << 8 | aid->data[2] << 16;

    return 0;
}

int
mifare_desfire_format_picc(FreefareTag tag)
{
    int rc;

    ASSERT_ACTIVE(tag);
    ASSERT_AUTHENTICATED(tag);

    BUFFER_INIT(cmd, 1 + CMAC_LENGTH);
    BUFFER_INIT(res, 1 + CMAC_LENGTH);

    BUFFER_APPEND(cmd, 0xFC);

    uint8_t *p = mifare_cryto_preprocess_data(tag, cmd, &__cmd_n, 0, MDCM_PLAIN | CMAC_COMMAND);

    if ((rc = MIFARE_DESFIRE_TRANSCEIVE(tag, p, __cmd_n, res, __res_size, &__res_n)) < 0)
	return rc;

    ssize_t sn = __res_n;
    p = mifare_cryto_postprocess_data(tag, res, &sn, MDCM_PLAIN | CMAC_COMMAND | CMAC_VERIFY);

    if (!p)
	return errno = EINVAL, -1;

    free(MIFARE_DESFIRE(tag)->session_key);
    MIFARE_DESFIRE(tag)->session_key = NULL;
    MIFARE_DESFIRE(tag)->selected_application = 0x000000;

    return 0;
}

/*
 * Retrieve version information form the PICC.
 */
int
mifare_desfire_get_version(FreefareTag tag, struct mifare_desfire_version_info *version_info)
{
    int rc;

    ASSERT_ACTIVE(tag);

    ASSERT_NOT_NULL(version_info);

    BUFFER_INIT(cmd, 1);
    BUFFER_INIT(res, 15 + CMAC_LENGTH);  /* 8, 8, then 15 byte results */

    char buffer[28 + CMAC_LENGTH + 1];

    BUFFER_APPEND(cmd, 0x60);
    uint8_t *p = mifare_cryto_preprocess_data(tag, cmd, &__cmd_n, 0, MDCM_PLAIN | CMAC_COMMAND);

    if ((rc = MIFARE_DESFIRE_TRANSCEIVE(tag, p, __cmd_n, res, __res_size, &__res_n)) < 0)
	return rc;
    memcpy(&(version_info->hardware), res, 7);
    memcpy(buffer, res, 7);

    p[0] = 0xAF;
    if ((rc = MIFARE_DESFIRE_TRANSCEIVE(tag, p, __cmd_n, res, __res_size, &__res_n)) < 0)
	return rc;
    memcpy(&(version_info->software), res, 7);
    memcpy(buffer + 7, res, 7);

    if ((rc = MIFARE_DESFIRE_TRANSCEIVE(tag, p, __cmd_n, res, __res_size, &__res_n)) < 0)
	return rc;
    memcpy(&(version_info->uid), res, 14);
    memcpy(buffer + 14, res, __res_n);

    ssize_t sn = 28 + CMAC_LENGTH + 1;
    p = mifare_cryto_postprocess_data(tag, buffer, &sn, MDCM_PLAIN | CMAC_COMMAND | CMAC_VERIFY);

    if (!p)
	return errno = EINVAL, -1;

    return 0;
}

int
mifare_desfire_free_mem(FreefareTag tag, uint32_t *size)
{
    int rc;

    ASSERT_ACTIVE(tag);

    ASSERT_NOT_NULL(size);

    BUFFER_INIT(cmd, 1);
    BUFFER_INIT(res, 4 + CMAC_LENGTH);

    BUFFER_APPEND(cmd, 0x6E);

    uint8_t *p = mifare_cryto_preprocess_data(tag, cmd, &__cmd_n, 0, MDCM_PLAIN | CMAC_COMMAND);

    if ((rc = MIFARE_DESFIRE_TRANSCEIVE(tag, p, __cmd_n, res, __res_size, &__res_n)) < 0)
	return rc;

    ssize_t sn = __res_n;
    p = mifare_cryto_postprocess_data(tag, res, &sn, MDCM_PLAIN | CMAC_COMMAND | CMAC_VERIFY);

    if (!p)
	return errno = EINVAL, -1;

    *size = p[0] | (p[1] << 8) | (p[2] << 16);

    return 0;
}

int
mifare_desfire_set_configuration(FreefareTag tag, bool disable_format, bool enable_random_uid)
{
    int rc;

    ASSERT_ACTIVE(tag);

    BUFFER_INIT(cmd, 10);
    BUFFER_INIT(res, 1 + CMAC_LENGTH);

    BUFFER_APPEND(cmd, 0x5C);
    BUFFER_APPEND(cmd, 0x00);
    BUFFER_APPEND(cmd, (enable_random_uid ? 0x02 : 0x00) | (disable_format ? 0x01 : 0x00));

    uint8_t *p = mifare_cryto_preprocess_data(tag, cmd, &__cmd_n, 2, MDCM_ENCIPHERED | ENC_COMMAND);

    if ((rc = MIFARE_DESFIRE_TRANSCEIVE(tag, p, __cmd_n, res, __res_size, &__res_n)) < 0)
	return rc;

    ssize_t sn = __res_n;
    p = mifare_cryto_postprocess_data(tag, res, &sn, MDCM_PLAIN | CMAC_COMMAND | CMAC_VERIFY);

    if (!p)
	return errno = EINVAL, -1;

    return 0;
}

int
mifare_desfire_set_default_key(FreefareTag tag, MifareDESFireKey key)
{
    int rc;

    ASSERT_ACTIVE(tag);

    BUFFER_INIT(cmd, 34);
    BUFFER_INIT(res, 1 + CMAC_LENGTH);

    BUFFER_APPEND(cmd, 0x5C);
    BUFFER_APPEND(cmd, 0x01);
    size_t key_data_length;
    switch (key->type) {
    case MIFARE_KEY_DES:
    case MIFARE_KEY_2K3DES:
    case MIFARE_KEY_AES128:
	key_data_length = 16;
	break;
    case MIFARE_KEY_3K3DES:
	key_data_length = 24;
	break;
    }
    BUFFER_APPEND_BYTES(cmd, key->data, key_data_length);
    while (__cmd_n < 26)
	BUFFER_APPEND(cmd, 0x00);
    BUFFER_APPEND(cmd, mifare_desfire_key_get_version(key));

    uint8_t *p = mifare_cryto_preprocess_data(tag, cmd, &__cmd_n, 2, MDCM_ENCIPHERED | ENC_COMMAND);

    if ((rc = MIFARE_DESFIRE_TRANSCEIVE(tag, p, __cmd_n, res, __res_size, &__res_n)) < 0)
	return rc;

    ssize_t sn = __res_n;
    p = mifare_cryto_postprocess_data(tag, res, &sn, MDCM_PLAIN | CMAC_COMMAND | CMAC_VERIFY);

    if (!p)
	return errno = EINVAL, -1;

    return 0;
}

int
mifare_desfire_set_ats(FreefareTag tag, uint8_t *ats)
{
    int rc;

    ASSERT_ACTIVE(tag);

    BUFFER_INIT(cmd, 34);
    BUFFER_INIT(res, 1 + CMAC_LENGTH);

    BUFFER_APPEND(cmd, 0x5C);
    BUFFER_APPEND(cmd, 0x02);
    BUFFER_APPEND_BYTES(cmd, ats, *ats);
    switch (MIFARE_DESFIRE(tag)->authentication_scheme) {
    case AS_LEGACY:
	iso14443a_crc_append(cmd + 2, __cmd_n - 2);
	__cmd_n += 2;
	break;
    case AS_NEW:
	desfire_crc32_append(cmd, __cmd_n);
	__cmd_n += 4;
	break;
    }
    BUFFER_APPEND(cmd, 0x80);

    uint8_t *p = mifare_cryto_preprocess_data(tag, cmd, &__cmd_n, 2, MDCM_ENCIPHERED | NO_CRC | ENC_COMMAND);

    if ((rc = MIFARE_DESFIRE_TRANSCEIVE(tag, p, __cmd_n, res, __res_size, &__res_n)) < 0)
	return rc;

    ssize_t sn = __res_n;
    p = mifare_cryto_postprocess_data(tag, res, &sn, MDCM_PLAIN | CMAC_COMMAND | CMAC_VERIFY);

    if (!p)
	return errno = EINVAL, -1;

    return 0;
}

int
mifare_desfire_get_card_uid_raw(FreefareTag tag, uint8_t uid[])
{
    int rc;

    ASSERT_ACTIVE(tag);

    ASSERT_NOT_NULL(uid);

    BUFFER_INIT(cmd, 1);
    BUFFER_INIT(res, 17 + CMAC_LENGTH);

    BUFFER_APPEND(cmd, 0x51);

    uint8_t *p = mifare_cryto_preprocess_data(tag, cmd, &__cmd_n, 1, MDCM_PLAIN | CMAC_COMMAND);

    if ((rc = MIFARE_DESFIRE_TRANSCEIVE(tag, p, __cmd_n, res, __res_size, &__res_n)) < 0)
	return rc;

    ssize_t sn = __res_n;
    p = mifare_cryto_postprocess_data(tag, res, &sn, MDCM_ENCIPHERED);

    if (!p)
	return errno = EINVAL, -1;

    memcpy(uid, p, 7);

    return 0;
}

int
mifare_desfire_get_card_uid(FreefareTag tag, char **uid)
{
    uint8_t p[7];

    if (mifare_desfire_get_card_uid_raw(tag, p) < 0) {
	return -1;
    }

    if (!(*uid = malloc(2 * 7 + 1))) {
	return -1;
    }

    sprintf(*uid, "%02x%02x%02x%02x%02x%02x%02x",
	    p[0], p[1], p[2], p[3],
	    p[4], p[5], p[6]);

    return 0;
}


/* Application level commands */

int
mifare_desfire_get_file_ids(FreefareTag tag, uint8_t **files, size_t *count)
{
    int rc;

    ASSERT_ACTIVE(tag);

    BUFFER_INIT(cmd, 1 + CMAC_LENGTH);
    BUFFER_INIT(res, 16 + CMAC_LENGTH);

    BUFFER_APPEND(cmd, 0x6F);


    uint8_t *p = mifare_cryto_preprocess_data(tag, cmd, &__cmd_n, 0, MDCM_PLAIN | CMAC_COMMAND);

    if ((rc = MIFARE_DESFIRE_TRANSCEIVE(tag, p, __cmd_n, res, __res_size, &__res_n)) < 0)
	return rc;

    ssize_t sn = __res_n;
    p = mifare_cryto_postprocess_data(tag, res, &sn, MDCM_PLAIN | CMAC_COMMAND | CMAC_VERIFY);

    if (!p)
	return errno = EINVAL, -1;

    *count = sn - 1;

    if (!(*files = malloc(*count))) {
	errno = ENOMEM;
	return -1;
    }
    memcpy(*files, res, *count);

    return 0;
}

int
mifare_desfire_get_iso_file_ids(FreefareTag tag, uint16_t **files, size_t *count)
{
    int rc;

    ASSERT_ACTIVE(tag);

    BUFFER_INIT(cmd, 1);
    BUFFER_INIT(res, 2 * 27 + 1);

    BUFFER_APPEND(cmd, 0x61);

    uint8_t *data;
    if (!(data = malloc((27 + 5) * 2 + 8)))
	return -1;

    off_t offset = 0;

    uint8_t *p = mifare_cryto_preprocess_data(tag, cmd, &__cmd_n, 0, MDCM_PLAIN | CMAC_COMMAND);

    do {
	if ((rc = MIFARE_DESFIRE_TRANSCEIVE(tag, p, __cmd_n, res, __res_size, &__res_n)) < 0)
	    return rc;

	memcpy(data + offset, res, __res_n - 1);
	offset += __res_n - 1;

	p[0] = 0XAF;
    } while (res[__res_n - 1] == 0xAF);

    ssize_t sn = offset;
    p = mifare_cryto_postprocess_data(tag, data, &sn, MDCM_PLAIN | CMAC_COMMAND);

    if (!p) {
	free(data);
	return errno = EINVAL, -1;
    }

    *count = sn / 2;
    *files = malloc(sizeof(**files) * *count);
    if (!*files) {
	free(data);
	return -1;
    }

    for (size_t i = 0; i < *count; i++) {
	(*files)[i] = le16toh(*(uint16_t *)(p + (2 * i)));
    }

    free(data);

    return 0;
}

int
mifare_desfire_get_file_settings(FreefareTag tag, uint8_t file_no, struct mifare_desfire_file_settings *settings)
{
    int rc;

    ASSERT_ACTIVE(tag);

    if (cached_file_settings_current[file_no]) {
	*settings = cached_file_settings[file_no];
	return 0;
    }

    BUFFER_INIT(cmd, 2 + CMAC_LENGTH);
    BUFFER_INIT(res, 18 + CMAC_LENGTH);

    BUFFER_APPEND(cmd, 0xF5);
    BUFFER_APPEND(cmd, file_no);

    uint8_t *p = mifare_cryto_preprocess_data(tag, cmd, &__cmd_n, 0, MDCM_PLAIN | CMAC_COMMAND);

    if ((rc = MIFARE_DESFIRE_TRANSCEIVE(tag, p, __cmd_n, res, __res_size, &__res_n)) < 0)
	return rc;

    ssize_t sn = __res_n;
    p = mifare_cryto_postprocess_data(tag, res, &sn, MDCM_PLAIN | CMAC_COMMAND | CMAC_VERIFY);

    if (!p)
	return errno = EINVAL, -1;

    struct mifare_desfire_raw_file_settings raw_settings;
    memcpy(&raw_settings, p, sn - 1);

    settings->file_type = raw_settings.file_type;
    settings->communication_settings = raw_settings.communication_settings;
    settings->access_rights = le16toh(raw_settings.access_rights);

    switch (settings->file_type) {
    case MDFT_STANDARD_DATA_FILE:
    case MDFT_BACKUP_DATA_FILE:
	settings->settings.standard_file.file_size = le24toh(raw_settings.settings.standard_file.file_size);
	break;
    case MDFT_VALUE_FILE_WITH_BACKUP:
	settings->settings.value_file.lower_limit = le32toh(raw_settings.settings.value_file.lower_limit);
	settings->settings.value_file.upper_limit = le32toh(raw_settings.settings.value_file.upper_limit);
	settings->settings.value_file.limited_credit_value = le32toh(raw_settings.settings.value_file.limited_credit_value);
	settings->settings.value_file.limited_credit_enabled = raw_settings.settings.value_file.limited_credit_enabled;
	break;
    case MDFT_LINEAR_RECORD_FILE_WITH_BACKUP:
    case MDFT_CYCLIC_RECORD_FILE_WITH_BACKUP:
	settings->settings.linear_record_file.record_size = le24toh(raw_settings.settings.linear_record_file.record_size);
	settings->settings.linear_record_file.max_number_of_records = le24toh(raw_settings.settings.linear_record_file.max_number_of_records);
	settings->settings.linear_record_file.current_number_of_records = le24toh(raw_settings.settings.linear_record_file.current_number_of_records);
	break;
    }

    cached_file_settings[file_no] = *settings;
    cached_file_settings_current[file_no] = true;

    return 0;
}

int
mifare_desfire_change_file_settings(FreefareTag tag, uint8_t file_no, uint8_t communication_settings, uint16_t access_rights)
{
    int rc;

    ASSERT_ACTIVE(tag);

    struct mifare_desfire_file_settings settings;
    int res = mifare_desfire_get_file_settings(tag, file_no, &settings);
    if (res < 0)
	return res;

    cached_file_settings_current[file_no] = false;

    if (MDAR_CHANGE_AR(settings.access_rights) == MDAR_FREE) {
	BUFFER_INIT(cmd, 5 + CMAC_LENGTH);
	BUFFER_INIT(res, 1 + CMAC_LENGTH);

	BUFFER_APPEND(cmd, 0x5F);
	BUFFER_APPEND(cmd, file_no);
	BUFFER_APPEND(cmd, communication_settings);
	BUFFER_APPEND_LE(cmd, access_rights, 2, sizeof(uint16_t));

	uint8_t *p = mifare_cryto_preprocess_data(tag, cmd, &__cmd_n, 0, MDCM_PLAIN | CMAC_COMMAND);

	if ((rc = MIFARE_DESFIRE_TRANSCEIVE(tag, p, __cmd_n, res, __res_size, &__res_n)) < 0)
	    return rc;

	ssize_t sn = __res_n;
	p = mifare_cryto_postprocess_data(tag, res, &sn, MDCM_PLAIN | CMAC_COMMAND | CMAC_VERIFY);

	if (!p)
	    return errno = EINVAL, -1;
    } else {
	BUFFER_INIT(cmd, 10);
	BUFFER_INIT(res, 1 + CMAC_LENGTH);

	BUFFER_APPEND(cmd, 0x5F);
	BUFFER_APPEND(cmd, file_no);
	BUFFER_APPEND(cmd, communication_settings);
	BUFFER_APPEND_LE(cmd, access_rights, 2, sizeof(uint16_t));

	uint8_t *p = mifare_cryto_preprocess_data(tag, cmd, &__cmd_n, 2, MDCM_ENCIPHERED | ENC_COMMAND);

	if ((rc = MIFARE_DESFIRE_TRANSCEIVE(tag, p, __cmd_n, res, __res_size, &__res_n)) < 0)
	    return rc;

	ssize_t sn = __res_n;
	p = mifare_cryto_postprocess_data(tag, res, &sn, MDCM_PLAIN | CMAC_COMMAND | CMAC_VERIFY);

	if (!p)
	    return errno = EINVAL, -1;
    }

    return 0;
}

static int
create_file1(FreefareTag tag, uint8_t command, uint8_t file_no, int has_iso_file_id, uint16_t iso_file_id,  uint8_t communication_settings, uint16_t access_rights, uint32_t file_size)
{
    int rc;

    ASSERT_ACTIVE(tag);

    BUFFER_INIT(cmd, 10 + CMAC_LENGTH);
    BUFFER_INIT(res, 1 + CMAC_LENGTH);

    BUFFER_APPEND(cmd, command);
    BUFFER_APPEND(cmd, file_no);
    if (has_iso_file_id)
	BUFFER_APPEND_LE(cmd, iso_file_id, sizeof(iso_file_id), sizeof(iso_file_id));
    BUFFER_APPEND(cmd, communication_settings);
    BUFFER_APPEND_LE(cmd, access_rights, 2, sizeof(uint16_t));
    BUFFER_APPEND_LE(cmd, file_size, 3, sizeof(uint32_t));

    uint8_t *p = mifare_cryto_preprocess_data(tag, cmd, &__cmd_n, 0, MDCM_PLAIN | CMAC_COMMAND);

    if ((rc = MIFARE_DESFIRE_TRANSCEIVE(tag, p, __cmd_n, res, __res_size, &__res_n)) < 0)
	return rc;

    ssize_t sn = __res_n;
    p = mifare_cryto_postprocess_data(tag, res, &sn, MDCM_PLAIN | CMAC_COMMAND | CMAC_VERIFY);

    if (!p)
	return errno = EINVAL, -1;

    cached_file_settings_current[file_no] = false;

    return 0;
}

int
mifare_desfire_create_std_data_file(FreefareTag tag, uint8_t file_no, uint8_t communication_settings, uint16_t access_rights, uint32_t file_size)
{
    return create_file1(tag, 0xCD, file_no, 0, 0x0000, communication_settings, access_rights, file_size);
}

int
mifare_desfire_create_std_data_file_iso(FreefareTag tag, uint8_t file_no, uint8_t communication_settings, uint16_t access_rights, uint32_t file_size, uint16_t iso_file_id)
{
    return create_file1(tag, 0xCD, file_no, 1, iso_file_id, communication_settings, access_rights, file_size);
}

int
mifare_desfire_create_backup_data_file(FreefareTag tag, uint8_t file_no, uint8_t communication_settings, uint16_t access_rights, uint32_t file_size)
{
    return create_file1(tag, 0xCB, file_no, 0, 0x0000, communication_settings, access_rights, file_size);
}

int
mifare_desfire_create_backup_data_file_iso(FreefareTag tag, uint8_t file_no, uint8_t communication_settings, uint16_t access_rights, uint32_t file_size, uint16_t iso_file_id)
{
    return create_file1(tag, 0xCB, file_no, 1, iso_file_id, communication_settings, access_rights, file_size);
}

int
mifare_desfire_create_value_file(FreefareTag tag, uint8_t file_no, uint8_t communication_settings, uint16_t access_rights, int32_t lower_limit, int32_t upper_limit, int32_t value, uint8_t limited_credit_enable)
{
    int rc;

    ASSERT_ACTIVE(tag);

    BUFFER_INIT(cmd, 18 + CMAC_LENGTH);
    BUFFER_INIT(res, 1 + CMAC_LENGTH);

    BUFFER_APPEND(cmd, 0xCC);
    BUFFER_APPEND(cmd, file_no);
    BUFFER_APPEND(cmd, communication_settings);
    BUFFER_APPEND_LE(cmd, access_rights, 2, sizeof(uint16_t));
    BUFFER_APPEND_LE(cmd, lower_limit, 4, sizeof(int32_t));
    BUFFER_APPEND_LE(cmd, upper_limit, 4, sizeof(int32_t));
    BUFFER_APPEND_LE(cmd, value, 4, sizeof(int32_t));
    BUFFER_APPEND(cmd, limited_credit_enable);

    uint8_t *p = mifare_cryto_preprocess_data(tag, cmd, &__cmd_n, 0, MDCM_PLAIN | CMAC_COMMAND);

    if ((rc = MIFARE_DESFIRE_TRANSCEIVE(tag, p, __cmd_n, res, __res_size, &__res_n)) < 0)
	return rc;

    ssize_t sn = __res_n;
    p = mifare_cryto_postprocess_data(tag, res, &sn, MDCM_PLAIN | CMAC_COMMAND | CMAC_VERIFY);

    if (!p)
	return errno = EINVAL, -1;

    cached_file_settings_current[file_no] = false;

    return 0;
}

static int
create_file2(FreefareTag tag, uint8_t command, uint8_t file_no, int has_iso_file_id, uint16_t iso_file_id, uint8_t communication_settings, uint16_t access_rights, uint32_t record_size, uint32_t max_number_of_records)
{
    int rc;

    ASSERT_ACTIVE(tag);

    BUFFER_INIT(cmd, 11 + CMAC_LENGTH);
    BUFFER_INIT(res, 1 + CMAC_LENGTH);

    BUFFER_APPEND(cmd, command);
    BUFFER_APPEND(cmd, file_no);
    if (has_iso_file_id)
	BUFFER_APPEND_LE(cmd, iso_file_id, sizeof(iso_file_id), sizeof(iso_file_id));
    BUFFER_APPEND(cmd, communication_settings);
    BUFFER_APPEND_LE(cmd, access_rights, 2, sizeof(uint16_t));
    BUFFER_APPEND_LE(cmd, record_size, 3, sizeof(uint32_t));
    BUFFER_APPEND_LE(cmd, max_number_of_records, 3, sizeof(uint32_t));

    uint8_t *p = mifare_cryto_preprocess_data(tag, cmd, &__cmd_n, 0, MDCM_PLAIN | CMAC_COMMAND);

    if ((rc = MIFARE_DESFIRE_TRANSCEIVE(tag, p, __cmd_n, res, __res_size, &__res_n)) < 0)
	return rc;

    ssize_t sn = __res_n;
    p = mifare_cryto_postprocess_data(tag, res, &sn, MDCM_PLAIN | CMAC_COMMAND | CMAC_VERIFY);

    if (!p)
	return errno = EINVAL, -1;

    cached_file_settings_current[file_no] = false;

    return 0;
}

int
mifare_desfire_create_linear_record_file(FreefareTag tag, uint8_t file_no, uint8_t communication_settings, uint16_t access_rights, uint32_t record_size, uint32_t max_number_of_records)
{
    return create_file2(tag, 0xC1, file_no, 0, 0x0000, communication_settings, access_rights, record_size, max_number_of_records);
}

int
mifare_desfire_create_linear_record_file_iso(FreefareTag tag, uint8_t file_no, uint8_t communication_settings, uint16_t access_rights, uint32_t record_size, uint32_t max_number_of_records, uint16_t iso_file_id)
{
    return create_file2(tag, 0xC1, file_no, 1, iso_file_id, communication_settings, access_rights, record_size, max_number_of_records);
}

int
mifare_desfire_create_cyclic_record_file(FreefareTag tag, uint8_t file_no, uint8_t communication_settings, uint16_t access_rights, uint32_t record_size, uint32_t max_number_of_records)
{
    return create_file2(tag, 0xC0, file_no, 0, 0x000, communication_settings, access_rights, record_size, max_number_of_records);
}

int
mifare_desfire_create_cyclic_record_file_iso(FreefareTag tag, uint8_t file_no, uint8_t communication_settings, uint16_t access_rights, uint32_t record_size, uint32_t max_number_of_records, uint16_t iso_file_id)
{
    return create_file2(tag, 0xC0, file_no, 1, iso_file_id, communication_settings, access_rights, record_size, max_number_of_records);
}

int
mifare_desfire_delete_file(FreefareTag tag, uint8_t file_no)
{
    int rc;

    ASSERT_ACTIVE(tag);

    BUFFER_INIT(cmd, 2 + CMAC_LENGTH);
    BUFFER_INIT(res, 1 + CMAC_LENGTH);

    BUFFER_APPEND(cmd, 0xDF);
    BUFFER_APPEND(cmd, file_no);

    uint8_t *p = mifare_cryto_preprocess_data(tag, cmd, &__cmd_n, 0, MDCM_PLAIN | CMAC_COMMAND);

    if ((rc = MIFARE_DESFIRE_TRANSCEIVE(tag, p, __cmd_n, res, __res_size, &__res_n)) < 0)
	return rc;

    ssize_t sn = __res_n;
    p = mifare_cryto_postprocess_data(tag, res, &sn, MDCM_PLAIN | CMAC_COMMAND | CMAC_VERIFY);

    if (!p)
	return errno = EINVAL, -1;

    return 0;
}

/*
 * Data manipulation commands.
 */

static ssize_t
read_data(FreefareTag tag, uint8_t command, uint8_t file_no, off_t offset, size_t length, void *data, int cs)
{
    int rc;

    size_t bytes_received = 0;

    ASSERT_ACTIVE(tag);
    ASSERT_CS(cs);

    BUFFER_INIT(cmd, 8);
    BUFFER_INIT(res, MAX_RAPDU_SIZE);

    BUFFER_APPEND(cmd, command);
    BUFFER_APPEND(cmd, file_no);
    BUFFER_APPEND_LE(cmd, offset, 3, sizeof(off_t));
    BUFFER_APPEND_LE(cmd, length, 3, sizeof(size_t));

    /*
     * In case no length is provided (whole file is to be read) get the file's
     * length in order to allocate a large enought buffer for crypto
     * post-processing.
     */
    int record_size = 1;

    struct mifare_desfire_file_settings settings;
    int r = mifare_desfire_get_file_settings(tag, file_no, &settings);
    if (r < 0)
	return r;

    switch (settings.file_type) {
    case MDFT_STANDARD_DATA_FILE:
    case MDFT_BACKUP_DATA_FILE:
    case MDFT_VALUE_FILE_WITH_BACKUP:
	/* NOOP */
	break;
    case MDFT_LINEAR_RECORD_FILE_WITH_BACKUP:
    case MDFT_CYCLIC_RECORD_FILE_WITH_BACKUP:
	// length indicates a number of records, we need the record size to
	// allocate enougth memory.
	record_size = settings.settings.linear_record_file.record_size;
	break;
    }
    if (!length) {
	switch (settings.file_type) {
	case MDFT_STANDARD_DATA_FILE:
	case MDFT_BACKUP_DATA_FILE:
	    length = settings.settings.standard_file.file_size;
	    break;
	case MDFT_VALUE_FILE_WITH_BACKUP:
	    abort();
	    break;
	case MDFT_LINEAR_RECORD_FILE_WITH_BACKUP:
	case MDFT_CYCLIC_RECORD_FILE_WITH_BACKUP:
	    length = settings.settings.linear_record_file.current_number_of_records;
	    break;
	}
    }

    uint8_t ocs = cs;
    if ((MIFARE_DESFIRE(tag)->session_key) && (cs | MDCM_MACED)) {
	switch (MIFARE_DESFIRE(tag)->authentication_scheme) {
	case AS_LEGACY:
	    break;
	case AS_NEW:
	    cs = MDCM_PLAIN;
	    break;
	}
    }
    uint8_t *p = mifare_cryto_preprocess_data(tag, cmd, &__cmd_n, 8, MDCM_PLAIN | CMAC_COMMAND);
    cs = ocs;

    /*
     * Depending on the communication settings, we might read more bytes than
     * the actual data length (a MAC or padding padding might follow).  This
     * can be a problem if the destination buffer is long enouth for the data
     * but the MAC / padding overflows.
     *
     * Create a temporary read buffer to collect all read data, post-process it
     * through the cryptography code and copy the actual data to the
     * destination buffer.
     */
    uint8_t *read_buffer = malloc(enciphered_data_length(tag, length * record_size, 0) + 1);

    do {
	if ((rc = MIFARE_DESFIRE_TRANSCEIVE(tag, p, __cmd_n, res, __res_size, &__res_n)) < 0) {
	    free(read_buffer);
	    return rc;
	}

	size_t frame_bytes = BUFFER_SIZE(res) - 1;
	memcpy(read_buffer + bytes_received, res, frame_bytes);
	bytes_received += frame_bytes;

	p[0] = 0xAF;
	__cmd_n = 1;
    } while (0xAF == res[__res_n - 1]);

    read_buffer[bytes_received++] = 0x00;

    ssize_t sr = bytes_received;
    p = mifare_cryto_postprocess_data(tag, read_buffer, &sr, cs | CMAC_COMMAND | CMAC_VERIFY | MAC_VERIFY);

    if (sr > 0)
	memcpy(data, read_buffer, sr - 1);

    free(read_buffer);

    if (!p)
	return errno = EINVAL, -1;

    return (sr <= 0) ? sr : sr - 1;
}

ssize_t
mifare_desfire_read_data(FreefareTag tag, uint8_t file_no, off_t offset, size_t length, void *data)
{
    return mifare_desfire_read_data_ex(tag, file_no, offset, length, data, madame_soleil_get_read_communication_settings(tag, file_no));
}

ssize_t
mifare_desfire_read_data_ex(FreefareTag tag, uint8_t file_no, off_t offset, size_t length, void *data, int cs)
{
    return read_data(tag, 0xBD, file_no, offset, length, data, cs);
}

static ssize_t
write_data(FreefareTag tag, uint8_t command, uint8_t file_no, off_t offset, size_t length, const void *data, int cs)
{
    int rc;

    size_t bytes_left;
    size_t bytes_send = 0;

    ASSERT_ACTIVE(tag);
    ASSERT_CS(cs);

    BUFFER_INIT(cmd, 8 + length + CMAC_LENGTH);
    BUFFER_INIT(res, 1 + CMAC_LENGTH);

    BUFFER_APPEND(cmd, command);
    BUFFER_APPEND(cmd, file_no);
    BUFFER_APPEND_LE(cmd, offset, 3, sizeof(off_t));
    BUFFER_APPEND_LE(cmd, length, 3, sizeof(size_t));
    BUFFER_APPEND_BYTES(cmd, data, length);

    uint8_t *p = mifare_cryto_preprocess_data(tag, cmd, &__cmd_n, 8, cs | MAC_COMMAND | CMAC_COMMAND | ENC_COMMAND);
    size_t overhead_size = __cmd_n - length; // (CRC | padding) + headers

    BUFFER_INIT(d, MAX_CAPDU_SIZE);
    bytes_left = __d_size;

    while (bytes_send < __cmd_n) {
	size_t frame_bytes = MIN(bytes_left, __cmd_n - bytes_send);
	BUFFER_APPEND_BYTES(d, p + bytes_send, frame_bytes);

	if ((rc = MIFARE_DESFIRE_TRANSCEIVE(tag, d, __d_n, res, __res_size, &__res_n)) < 0)
	    return rc;

	bytes_send += frame_bytes;

	if (0x00 == res[__res_n - 1])
	    break;

	// PICC returned 0xAF and expects more data
	BUFFER_CLEAR(d);
	bytes_left = __d_size;
	BUFFER_APPEND(d, 0xAF);
	bytes_left--;
    }

    ssize_t sn = __res_n;
    p = mifare_cryto_postprocess_data(tag, res, &sn, MDCM_PLAIN | CMAC_COMMAND | CMAC_VERIFY);

    if (!p)
	return errno = EINVAL, -1;

    if (0x00 == p[__res_n - 1]) {
	// Remove header length
	bytes_send -= overhead_size;
    } else {
	// 0xAF (additionnal Frame) failure can happen here (wrong crypto method).
	MIFARE_DESFIRE(tag)->last_picc_error = p[__res_n - 1];
	bytes_send = -1;
    }

    cached_file_settings_current[file_no] = false;

    return bytes_send;
}

ssize_t
mifare_desfire_write_data(FreefareTag tag, uint8_t file_no, off_t offset, size_t length, const void *data)
{
    return mifare_desfire_write_data_ex(tag, file_no, offset, length, data, madame_soleil_get_write_communication_settings(tag, file_no));
}

ssize_t
mifare_desfire_write_data_ex(FreefareTag tag, uint8_t file_no, off_t offset, size_t length, const void *data, int cs)
{
    return write_data(tag, 0x3D, file_no, offset, length, data, cs);
}

int
mifare_desfire_get_value(FreefareTag tag, uint8_t file_no, int32_t *value)
{
    return mifare_desfire_get_value_ex(tag, file_no, value, madame_soleil_get_read_communication_settings(tag, file_no));
}
int
mifare_desfire_get_value_ex(FreefareTag tag, uint8_t file_no, int32_t *value, int cs)
{
    int rc;

    if (!value)
	return errno = EINVAL, -1;

    ASSERT_ACTIVE(tag);
    ASSERT_CS(cs);

    BUFFER_INIT(cmd, 2 + CMAC_LENGTH);
    BUFFER_INIT(res, 9 + CMAC_LENGTH);

    BUFFER_APPEND(cmd, 0x6C);
    BUFFER_APPEND(cmd, file_no);

    uint8_t *p = mifare_cryto_preprocess_data(tag, cmd, &__cmd_n, 0, MDCM_PLAIN | CMAC_COMMAND);

    if ((rc = MIFARE_DESFIRE_TRANSCEIVE(tag, p, __cmd_n, res, __res_size, &__res_n)) < 0)
	return rc;

    ssize_t sn = __res_n;
    p = mifare_cryto_postprocess_data(tag, res, &sn, cs | CMAC_COMMAND | CMAC_VERIFY | MAC_VERIFY);

    if (!p)
	return errno = EINVAL, -1;

    *value = le32toh(*(int32_t *)(p));

    return 0;
}

int
mifare_desfire_credit(FreefareTag tag, uint8_t file_no, int32_t amount)
{
    return mifare_desfire_credit_ex(tag, file_no, amount, madame_soleil_get_write_communication_settings(tag, file_no));
}

int
mifare_desfire_credit_ex(FreefareTag tag, uint8_t file_no, int32_t amount, int cs)
{
    int rc;

    ASSERT_ACTIVE(tag);
    ASSERT_CS(cs);

    BUFFER_INIT(cmd, 10 + CMAC_LENGTH);
    BUFFER_INIT(res, 1 + CMAC_LENGTH);

    BUFFER_APPEND(cmd, 0x0C);
    BUFFER_APPEND(cmd, file_no);
    BUFFER_APPEND_LE(cmd, amount, 4, sizeof(int32_t));
    uint8_t *p = mifare_cryto_preprocess_data(tag, cmd, &__cmd_n, 2, cs | MAC_COMMAND | CMAC_COMMAND | ENC_COMMAND);

    if ((rc = MIFARE_DESFIRE_TRANSCEIVE(tag, p, __cmd_n, res, __res_size, &__res_n)) < 0)
	return rc;

    ssize_t sn = __res_n;
    p = mifare_cryto_postprocess_data(tag, res, &sn, MDCM_PLAIN | CMAC_COMMAND | CMAC_VERIFY);

    if (!p)
	return errno = EINVAL, -1;

    cached_file_settings_current[file_no] = false;

    return 0;
}

int
mifare_desfire_debit(FreefareTag tag, uint8_t file_no, int32_t amount)
{
    return mifare_desfire_debit_ex(tag, file_no, amount, madame_soleil_get_write_communication_settings(tag, file_no));
}
int
mifare_desfire_debit_ex(FreefareTag tag, uint8_t file_no, int32_t amount, int cs)
{
    int rc;

    ASSERT_ACTIVE(tag);
    ASSERT_CS(cs);

    BUFFER_INIT(cmd, 10 + CMAC_LENGTH);
    BUFFER_INIT(res, 1 + CMAC_LENGTH);

    BUFFER_APPEND(cmd, 0xDC);
    BUFFER_APPEND(cmd, file_no);
    BUFFER_APPEND_LE(cmd, amount, 4, sizeof(int32_t));
    uint8_t *p = mifare_cryto_preprocess_data(tag, cmd, &__cmd_n, 2, cs | MAC_COMMAND | CMAC_COMMAND | ENC_COMMAND);

    if ((rc = MIFARE_DESFIRE_TRANSCEIVE(tag, p, __cmd_n, res, __res_size, &__res_n)) < 0)
	return rc;

    ssize_t sn = __res_n;
    p = mifare_cryto_postprocess_data(tag, res, &sn, MDCM_PLAIN | CMAC_COMMAND | CMAC_VERIFY);

    if (!p)
	return errno = EINVAL, -1;

    cached_file_settings_current[file_no] = false;

    return 0;
}

int
mifare_desfire_limited_credit(FreefareTag tag, uint8_t file_no, int32_t amount)
{
    return mifare_desfire_limited_credit_ex(tag, file_no, amount, madame_soleil_get_write_communication_settings(tag, file_no));
}
int
mifare_desfire_limited_credit_ex(FreefareTag tag, uint8_t file_no, int32_t amount, int cs)
{
    int rc;

    ASSERT_ACTIVE(tag);
    ASSERT_CS(cs);

    BUFFER_INIT(cmd, 10 + CMAC_LENGTH);
    BUFFER_INIT(res, 1 + CMAC_LENGTH);

    BUFFER_APPEND(cmd, 0x1C);
    BUFFER_APPEND(cmd, file_no);
    BUFFER_APPEND_LE(cmd, amount, 4, sizeof(int32_t));
    uint8_t *p = mifare_cryto_preprocess_data(tag, cmd, &__cmd_n, 2, cs | MAC_COMMAND | CMAC_COMMAND | ENC_COMMAND);

    if ((rc = MIFARE_DESFIRE_TRANSCEIVE(tag, p, __cmd_n, res, __res_size, &__res_n)) < 0)
	return rc;

    ssize_t sn = __res_n;
    p = mifare_cryto_postprocess_data(tag, res, &sn, MDCM_PLAIN | CMAC_COMMAND | CMAC_VERIFY);

    if (!p)
	return errno = EINVAL, -1;

    cached_file_settings_current[file_no] = false;

    return 0;
}

ssize_t
mifare_desfire_write_record(FreefareTag tag, uint8_t file_no, off_t offset, size_t length, void *data)
{
    return mifare_desfire_write_record_ex(tag, file_no, offset, length, data, madame_soleil_get_write_communication_settings(tag, file_no));
}
ssize_t
mifare_desfire_write_record_ex(FreefareTag tag, uint8_t file_no, off_t offset, size_t length, void *data, int cs)
{
    return write_data(tag, 0x3B, file_no, offset, length, data, cs);
}

ssize_t
mifare_desfire_read_records(FreefareTag tag, uint8_t file_no, off_t offset, size_t length, void *data)
{
    return mifare_desfire_read_records_ex(tag, file_no, offset, length, data, madame_soleil_get_read_communication_settings(tag, file_no));
}

ssize_t
mifare_desfire_read_records_ex(FreefareTag tag, uint8_t file_no, off_t offset, size_t length, void *data, int cs)
{
    return read_data(tag, 0xBB, file_no, offset, length, data, cs);
}

int
mifare_desfire_clear_record_file(FreefareTag tag, uint8_t file_no)
{
    int rc;

    ASSERT_ACTIVE(tag);

    BUFFER_INIT(cmd, 2 + CMAC_LENGTH);
    BUFFER_INIT(res, 1 + CMAC_LENGTH);

    BUFFER_APPEND(cmd, 0xEB);
    BUFFER_APPEND(cmd, file_no);

    uint8_t *p = mifare_cryto_preprocess_data(tag, cmd, &__cmd_n, 0, MDCM_PLAIN | CMAC_COMMAND);

    if ((rc = MIFARE_DESFIRE_TRANSCEIVE(tag, p, __cmd_n, res, __res_size, &__res_n)) < 0)
	return rc;

    ssize_t sn = __res_n;
    p = mifare_cryto_postprocess_data(tag, res, &sn, MDCM_PLAIN | CMAC_COMMAND | CMAC_VERIFY);

    if (!p)
	return errno = EINVAL, -1;

    cached_file_settings_current[file_no] = false;

    return 0;
}

int
mifare_desfire_commit_transaction(FreefareTag tag)
{
    int rc;

    ASSERT_ACTIVE(tag);

    BUFFER_INIT(cmd, 1 + CMAC_LENGTH);
    BUFFER_INIT(res, 1 + CMAC_LENGTH);

    BUFFER_APPEND(cmd, 0xC7);

    uint8_t *p = mifare_cryto_preprocess_data(tag, cmd, &__cmd_n, 0, MDCM_PLAIN | CMAC_COMMAND);

    if ((rc = MIFARE_DESFIRE_TRANSCEIVE(tag, p, __cmd_n, res, __res_size, &__res_n)) < 0)
	return rc;

    ssize_t sn = __res_n;
    p = mifare_cryto_postprocess_data(tag, res, &sn, MDCM_PLAIN | CMAC_COMMAND | CMAC_VERIFY);

    if (!p)
	return errno = EINVAL, -1;

    return 0;
}

int
mifare_desfire_abort_transaction(FreefareTag tag)
{
    int rc;

    ASSERT_ACTIVE(tag);

    BUFFER_INIT(cmd, 1 + CMAC_LENGTH);
    BUFFER_INIT(res, 1 + CMAC_LENGTH);

    BUFFER_APPEND(cmd, 0xA7);

    uint8_t *p = mifare_cryto_preprocess_data(tag, cmd, &__cmd_n, 0, MDCM_PLAIN | CMAC_COMMAND);

    if ((rc = MIFARE_DESFIRE_TRANSCEIVE(tag, p, __cmd_n, res, __res_size, &__res_n)) < 0)
	return rc;

    ssize_t sn = __res_n;
    p = mifare_cryto_postprocess_data(tag, res, &sn, MDCM_PLAIN | CMAC_COMMAND | CMAC_VERIFY);

    if (!p)
	return errno = EINVAL, -1;

    return 0;
}

