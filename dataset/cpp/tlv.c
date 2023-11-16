/*
 * This implementation was written based on information provided by the
 * following documents:
 *
 * NFC Forum - Type 1 Tag Operation Specification
 *   NFCForum-TS-Type-1-Tag_1.0
 *   2007-07-09
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

#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include <freefare.h>
#include "freefare_internal.h"

#define TLV_TERMINATOR 0xFE

size_t		 tlv_record_length(const uint8_t *stream, size_t *field_length_size, size_t *field_value_size);
uint8_t		*tlv_next(uint8_t *stream);
size_t		 tlv_sequence_length(uint8_t *stream);

/*
 * TLV (Type Length Value) Manipulation Functions.
 */

/*
 * Encode data stream into TLV.
 */
uint8_t *
tlv_encode(const uint8_t type, const uint8_t *istream, uint16_t isize, size_t *osize)
{
    uint8_t *res;
    off_t n = 0;

    if (osize)
	*osize = 0;

    if (isize == 0xffff) /* RFU */
	return NULL;

    if ((res = malloc(1 + ((isize > 254) ? 3 : 1) + isize + 1))) {
	/* type + size + payload + terminator */
	res[n++] = type;

	if (isize > 254) {
	    res[n++] = 0xff;
	    uint16_t size_be = htobe16(isize);
	    memcpy(res + n, &size_be, sizeof(uint16_t));
	    n += 2;
	} else {
	    res[n++] = (uint8_t)isize;
	}

	memcpy(res + n, istream, isize);

	n += isize;
	res[n++] = TLV_TERMINATOR;

	if (osize)
	    *osize = n;
    }
    return res;
}

/*
 * Decode TLV from data stream.
 */
uint8_t *
tlv_decode(const uint8_t *istream, uint8_t *type, uint16_t *size)
{
    size_t fls = 0;
    size_t fvs = 0;
    uint8_t *res = NULL;

    if (type)
	*type = istream[0];

    tlv_record_length(istream, &fls, &fvs);

    if (size) {
	*size = fvs;
    }

    if ((res = malloc(fvs))) {
	memcpy(res, istream + 1 + fls, fvs);
    }
    return res;
}

/*
 * Length of a TLV field
 */
size_t
tlv_record_length(const uint8_t *stream, size_t *field_length_size, size_t *field_value_size)
{
    size_t fls = 0;
    size_t fvs = 0;

    switch (stream[0]) {
    case 0x00:
    case 0xfe:
	break;
    case 0x01:
    case 0x02:
    case 0x03:
    default: // FIXME Not supported.
	if (stream[1] == 0xff) {
	    uint16_t be_size;
	    memcpy(&be_size, stream + 2, sizeof(uint16_t));
	    fls = 3;
	    fvs = be16toh(be_size);
	} else {
	    fls = 1;
	    fvs = stream[1];
	}
	break;
    }

    if (field_length_size)
	*field_length_size = fls;

    if (field_value_size)
	*field_value_size = fvs;

    return 1 + fls + fvs;
}

/*
 * Get a pointer to the next record in the provided TLV sequence.
 *             | 0x03 | 0x02 | 0xbe | 0xef | 0x00 | 0x00 | 0xfe |
 * First call  ---'                           |      |
 * Second call -------------------------------'      |
 * Third call  --------------------------------------'
 * Fourth call NULL
 */
uint8_t *
tlv_next(uint8_t *stream)
{
    uint8_t *res = NULL;
    if (stream[0] != TLV_TERMINATOR)
	res = stream + tlv_record_length(stream, NULL, NULL);

    return res;
}

/*
 * Full-length of all TLV fields.
 */
size_t
tlv_sequence_length(uint8_t *stream)
{
    size_t res = 0;

    do {
	res += tlv_record_length(stream, NULL, NULL);
    } while ((stream = tlv_next(stream)));

    return res;
}


/*
 * Append two TLV.  Acts like realloc(3).
 */
uint8_t *
tlv_append(uint8_t *a, uint8_t *b)
{
    size_t a_size = tlv_sequence_length(a);
    size_t b_size = tlv_sequence_length(b);
    size_t new_size = a_size + b_size - 1;

    if ((a = realloc(a, new_size))) {
	memcpy(a + a_size - 1, b, b_size);
    }

    return a;
}
