#include <err.h>
#include <stdlib.h>

#include <nfc/nfc.h>

#include <freefare.h>


int
main(int argc, char *argv[])
{
    int error = EXIT_SUCCESS;
    nfc_device *device = NULL;
    FreefareTag *tags = NULL;

    if (argc > 1)
	errx(EXIT_FAILURE, "usage: %s", argv[0]);

    nfc_connstring devices[8];
    size_t device_count;

    nfc_context *context;
    nfc_init(&context);
    if (context == NULL)
	errx(EXIT_FAILURE, "Unable to init libnfc (malloc)");

    device_count = nfc_list_devices(context, devices, sizeof(devices) / sizeof(*devices));
    if (device_count <= 0)
	errx(EXIT_FAILURE, "No NFC device found");

    for (size_t d = 0; d < device_count; d++) {
	if (!(device = nfc_open(context, devices[d]))) {
	    warnx("nfc_open() failed.");
	    error = EXIT_FAILURE;
	    continue;
	}

	if (!(tags = freefare_get_tags(device))) {
	    nfc_close(device);
	    errx(EXIT_FAILURE, "Error listing tags.");
	}

	for (int i = 0; (!error) && tags[i]; i++) {
	    switch (freefare_get_tag_type(tags[i])) {
	    case NTAG_21x:
		break;
	    default:
		continue;
	    }

	    char *tag_uid = freefare_get_tag_uid(tags[i]);
	    printf("Tag with UID %s is a %s\n", tag_uid, freefare_get_tag_friendly_name(tags[i]));
	    FreefareTag tag = tags[i];
	    int res;
	    if (ntag21x_connect(tag) < 0)
		errx(EXIT_FAILURE, "Error connecting to tag.");

	    uint8_t pwd[4] = {0xff, 0xff, 0xff, 0xff};
	    uint8_t pack[2] = {0xaa, 0xaa};

	    NTAG21xKey key;
	    key = ntag21x_key_new(pwd, pack); // Creating key

	    uint8_t auth0 = 0x00; // Buffer for auth0 byte
	    uint8_t authlim = 0x00;
	    switch (true) {
	    case true:
		/*
		   Get information about tag
		   MUST do, because here we are recognizing tag subtype (NTAG213,NTAG215,NTAG216), and gathering all parameters
		   */
		res = ntag21x_get_info(tag);
		if (res < 0) {
		    printf("Error getting info from tag\n");
		    break;
		}
		// Get auth byte from tag
		res = ntag21x_get_auth(tag, &auth0);
		if (res < 0) {
		    printf("Error getting auth0 byte from tag\n");
		    break;
		}
		printf("Old auth0: %#02x\n", auth0);
		res = ntag21x_get_authentication_limit(tag, &authlim);
		if (res < 0) {
		    printf("Error getting auth0 byte from tag\n");
		    break;
		}
		printf("Authlim: %#02x\n", authlim);
		// Check if auth is required to set pwd and pack
		if (auth0 < ntag21x_get_last_page(tag) - 2) { // Check if last 2 pages are protected
		    printf("Error: pwd and PACK sections are protected with unknown password\n");
		    break;
		}
		// Set key
		res = ntag21x_set_key(tag, key);
		if (res < 0) {
		    printf("Error setting key tag\n");
		    break;
		}
		// Protect last 6 pages !! It can be hacked if you don't protect last 4 pages where auth0 byte is located
		res = ntag21x_set_auth(tag, ntag21x_get_last_page(tag) - 5);
		if (res < 0) {
		    printf("Error setting auth0 byte \n");
		    break;
		}
		// Enable read & write pwd protection (default: write only protection)
		res = ntag21x_access_enable(tag, NTAG_PROT);
		if (res < 0) {
		    printf("Error setting access byte \n");
		    break;
		}
		// Get auth byte from tag
		res = ntag21x_get_auth(tag, &auth0);
		if (res < 0) {
		    printf("Error getting auth0 byte from tag\n");
		    break;
		}
		printf("New auth0: %#02x\n", auth0);
	    }
	    ntag21x_disconnect(tag);
	    ntag21x_key_free(key); // Delete key
	    free(tag_uid);
	}
	freefare_free_tags(tags);
	nfc_close(device);
    }
    nfc_exit(context);
    exit(error);
}
