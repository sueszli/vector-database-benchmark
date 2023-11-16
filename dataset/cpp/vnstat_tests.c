#include "common.h"
#include "cfg.h"
#include "vnstat_tests.h"
#include "common_tests.h"
#include "dbsql_tests.h"
#include "database_tests.h"
#include "config_tests.h"
#include "ifinfo_tests.h"
#include "misc_tests.h"
#include "daemon_tests.h"
#include "datacache_tests.h"
#include "fs_tests.h"
#include "id_tests.h"
#include "iflist_tests.h"
#include "cli_tests.h"
#include "parseargs_tests.h"
#if defined(HAVE_IMAGE)
#include "image_tests.h"
#endif

int output_suppressed = 0;

int main(void)
{
	Suite *s;
	SRunner *sr;
	int number_failed = 0;

	verify_fork_status();
	s = test_suite();
	sr = srunner_create(s);
	srunner_set_log(sr, "test.log");
	srunner_set_xml(sr, "test.xml");
	srunner_run_all(sr, CK_NORMAL);
	number_failed = srunner_ntests_failed(sr);
	srunner_free(sr);

	if (number_failed == 0) {
		remove_directory(TESTDIR);
	}

	return number_failed;
}

Suite *test_suite(void)
{
	Suite *s = suite_create("vnStat");

	add_common_tests(s);
	add_dbsql_tests(s);
	add_database_tests(s);
	add_config_tests(s);
	add_ifinfo_tests(s);
	add_misc_tests(s);
	add_daemon_tests(s);
	add_datacache_tests(s);
	add_fs_tests(s);
	add_id_tests(s);
	add_iflist_tests(s);
	add_cli_tests(s);
	add_parseargs_tests(s);
#if defined(HAVE_IMAGE)
	add_image_tests(s);
#endif

	return s;
}

/* tests can't reliably be executed if check doesn't have forking enabled */
/* and only SRunner knows that so a dummy runner needs to be created */
void verify_fork_status(void)
{
	Suite *s = suite_create("fork status check");
	SRunner *sr = srunner_create(s);
	if (srunner_fork_status(sr) == CK_NOFORK) {
		printf("Error: Tests require Check to have fork mode enabled.\n");
		exit(1);
	}
	srunner_free(sr);
}

void setup(void) {
	defaultcfg();
	debug = 0;
	disableprinte = 0;
	stderrprinte = 0;
}

void teardown(void) {
	return;
}

void suppress_output(void)
{
	if (!output_suppressed) {
		fclose(stdout);
		output_suppressed = 1;
	}
}

int pipe_output(void)
{
	int out_pipe[2];

	if (pipe(out_pipe) != 0) {
		ck_abort_msg("error \"%s\" while creating pipe", strerror(errno));
	}

	if (out_pipe[1] != STDOUT_FILENO) {
		dup2(out_pipe[1], STDOUT_FILENO);
		close(out_pipe[1]);
	}

	output_suppressed = 1;
	return out_pipe[0];
}

void disable_logprints(void)
{
	noexit = 2;
	cfg.uselogging = 0;
	disableprinte = 1;
}

int clean_testdbdir(void)
{
	struct stat statbuf;

	create_testdir();

	if (stat(TESTDBDIR, &statbuf) != 0) {
		if (errno == ENOENT) {
			if (mkdir(TESTDBDIR, 0755) == 0) {
				return 1;
			}
		}
		ck_abort_msg("error \"%s\" while creating directory \"%s\"", strerror(errno), TESTDBDIR);
	}

	if (!remove_directory(TESTDBDIR)) {
		ck_abort_msg("error \"%s\" while removing directory \"%s\", please remove it manually", strerror(errno), TESTDBDIR);
	}

	if (mkdir(TESTDBDIR, 0755) != 0) {
		ck_abort_msg("error \"%s\" while creating directory \"%s\"", strerror(errno), TESTDBDIR);
	}

	return 1;
}

int create_testdir(void)
{
	struct stat statbuf;

	if (stat(TESTDIR, &statbuf) != 0) {
		if (errno == ENOENT) {
			if (mkdir(TESTDIR, 0755) == 0) {
				return 1;
			}
		}
		ck_abort_msg("error \"%s\" while creating directory \"%s\"", strerror(errno), TESTDIR);
	}

	return 1;
}

int create_directory(const char *directory)
{
	struct stat statbuf;

	if (stat(directory, &statbuf) != 0) {
		if (errno == ENOENT) {
			if (mkdir(directory, 0755) != 0) {
				ck_abort_msg("error \"%s\" while creating directory \"%s\"", strerror(errno), directory);
			}
		} else {
			ck_abort_msg("error \"%s\" while creating directory \"%s\"", strerror(errno), directory);
		}
	}

	return 1;
}

int remove_directory(const char *directory)
{
	DIR *dir = NULL;
	struct dirent *di = NULL;
	char entryname[512];

	if ((dir = opendir(directory)) == NULL) {
		if (errno == ENOENT) {
			return 1;
		} else {
			return 0;
		}
	}

	while ((di = readdir(dir))) {
		switch (di->d_type) {
			case DT_LNK:
			case DT_REG:
				snprintf(entryname, 512, "%s/%s", directory, di->d_name);
				if (unlink(entryname) != 0) {
					closedir(dir);
					return 0;
				}
				break;
			case DT_DIR:
				if (strcmp(di->d_name, ".") == 0 || strcmp(di->d_name, "..") == 0) {
					continue;
				}
				snprintf(entryname, 512, "%s/%s", directory, di->d_name);
				if (!remove_directory(entryname)) {
					closedir(dir);
					return 0;
				}
				break;
			case DT_UNKNOWN:
				if (strcmp(di->d_name, ".") == 0 || strcmp(di->d_name, "..") == 0) {
					continue;
				}
				snprintf(entryname, 512, "%s/%s", directory, di->d_name);
				if (unlink(entryname) != 0) {
					if (errno == EISDIR) {
						if (!remove_directory(entryname)) {
							closedir(dir);
							return 0;
						}
					} else {
						closedir(dir);
						return 0;
					}
				}
				break;
			default:
				continue;
		}
	}
	closedir(dir);
	if (rmdir(directory) != 0) {
		return 0;
	}

	return 1;
}

int create_zerosize_dbfile(const char *iface)
{
	FILE *fp;
	char filename[512];

	snprintf(filename, 512, "%s/%s", TESTDBDIR, iface);
	if ((fp = fopen(filename, "w")) == NULL) {
		ck_abort_msg("error \"%s\" while opening file \"%s\" for writing", strerror(errno), filename);
	}
	fclose(fp);

	return 1;
}

int check_dbfile_exists(const char *iface, const int minsize)
{
	struct stat statbuf;
	char filename[512];

	snprintf(filename, 512, "%s/%s", TESTDBDIR, iface);
	if (stat(filename, &statbuf) != 0) {
		if (errno == ENOENT) {
			return 0;
		}
		ck_abort_msg("error \"%s\" while inspecting file \"%s\"", strerror(errno), filename);
	}

	if (statbuf.st_size < minsize) {
		ck_abort_msg("file \"%s\" is smaller (%d) then given minimum %d", filename, (int)statbuf.st_size, minsize);
	}

	return 1;
}

int fake_proc_net_dev(const char *mode, const char *iface, const int rx, const int tx, const int rxp, const int txp)
{
	FILE *devfp;
	char filename[512];

	if (strcmp(mode, "w") != 0 && strcmp(mode, "a") != 0) {
		ck_abort_msg("error: only w and a modes are supported");
	}

	create_testdir();
	create_directory(TESTPROCDIR);

	snprintf(filename, 512, "%s/dev", TESTPROCDIR);
	if ((devfp = fopen(filename, mode)) == NULL) {
		ck_abort_msg("error \"%s\" while opening file \"%s\" for writing", strerror(errno), filename);
	}

	if (strcmp(mode, "w") == 0) {
		fprintf(devfp, "Inter-|   Receive                                                |  Transmit\n");
		fprintf(devfp, " face |bytes    packets errs drop fifo frame compressed multicast|bytes    packets errs drop fifo colls carrier compressed\n");
		fprintf(devfp, "    lo:       0       0    0    0    0     0          0         0        0       0    0    0    0     0       0          0\n");
	}
	fprintf(devfp, "%6s: %7d %7d    0    0    0     0          0         0  %7d %7d    0    0    0     0       0          0\n", iface, rx, rxp, tx, txp);

	fclose(devfp);

	return 1;
}

int fake_sys_class_net(const char *iface, const int rx, const int tx, const int rxp, const int txp, const int speed)
{
	FILE *devfp;
	char dirname[512];
	char filename[512];

	create_testdir();
	create_directory(TESTSYSCLASSNETDIR);

	snprintf(dirname, 512, "%s/%s", TESTSYSCLASSNETDIR, iface);
	create_directory(dirname);

	snprintf(dirname, 512, "%s/%s/statistics", TESTSYSCLASSNETDIR, iface);
	create_directory(dirname);

	if (speed != 0) {
		snprintf(filename, 512, "%s/%s/speed", TESTSYSCLASSNETDIR, iface);
		if ((devfp = fopen(filename, "w")) == NULL) {
			ck_abort_msg("error \"%s\" while opening file \"%s\" for writing", strerror(errno), filename);
		}
		fprintf(devfp, "%d\n", speed);
		fclose(devfp);
	}

	snprintf(filename, 512, "%s/%s/statistics/rx_bytes", TESTSYSCLASSNETDIR, iface);
	if ((devfp = fopen(filename, "w")) == NULL) {
		ck_abort_msg("error \"%s\" while opening file \"%s\" for writing", strerror(errno), filename);
	}
	fprintf(devfp, "%d\n", rx);
	fclose(devfp);

	snprintf(filename, 512, "%s/%s/statistics/tx_bytes", TESTSYSCLASSNETDIR, iface);
	if ((devfp = fopen(filename, "w")) == NULL) {
		ck_abort_msg("error \"%s\" while opening file \"%s\" for writing", strerror(errno), filename);
	}
	fprintf(devfp, "%d\n", tx);
	fclose(devfp);

	snprintf(filename, 512, "%s/%s/statistics/rx_packets", TESTSYSCLASSNETDIR, iface);
	if ((devfp = fopen(filename, "w")) == NULL) {
		ck_abort_msg("error \"%s\" while opening file \"%s\" for writing", strerror(errno), filename);
	}
	fprintf(devfp, "%d\n", rxp);
	fclose(devfp);

	snprintf(filename, 512, "%s/%s/statistics/tx_packets", TESTSYSCLASSNETDIR, iface);
	if ((devfp = fopen(filename, "w")) == NULL) {
		ck_abort_msg("error \"%s\" while opening file \"%s\" for writing", strerror(errno), filename);
	}
	fprintf(devfp, "%d\n", txp);
	fclose(devfp);

	return 1;
}

uint64_t get_timestamp(const int year, const int month, const int day, const int hour, const int minute)
{
	struct tm stm;

	memset(&stm, 0, sizeof(struct tm));
	stm.tm_year = year - 1900;
	stm.tm_mon = month - 1;
	stm.tm_mday = day;
	stm.tm_hour = hour;
	stm.tm_min = minute;
	stm.tm_isdst = -1;

	if (cfg.useutc) {
		return (uint64_t)timegm(&stm);
	} else {
		return (uint64_t)mktime(&stm);
	}
}
