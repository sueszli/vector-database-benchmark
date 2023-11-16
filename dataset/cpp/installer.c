/*
 * (C) Copyright 2013-2023
 * Stefano Babic <stefano.babic@swupdate.org>
 *
 * SPDX-License-Identifier:     GPL-2.0-only
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <stdint.h>
#include <unistd.h>
#include <string.h>
#include <getopt.h>
#include <errno.h>
#include <ctype.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/mount.h>

#include "generated/autoconf.h"
#include "bsdqueue.h"
#include "globals.h"
#include "util.h"
#include "swupdate.h"
#include "installer.h"
#include "handler.h"
#include "cpiohdr.h"
#include "parsers.h"
#include "bootloader.h"
#include "progress.h"
#include "pctl.h"
#include "swupdate_vars.h"

/*
 * function returns:
 * 0 = do not skip the file, it must be installed
 * 1 = skip the file
 * 2 = install directly (stream to the handler)
 * -1= error found
 */
swupdate_file_t check_if_required(struct imglist *list, struct filehdr *pfdh,
				const char *destdir,
				struct img_type **pimg)
{
	swupdate_file_t skip = SKIP_FILE;
	struct img_type *img;

	/*
	 * Check that not more than one image want to be streamed
	 */
	int install_direct = 0;

	LIST_FOREACH(img, list, next) {
		if (strcmp(pfdh->filename, img->fname) == 0) {
			skip = COPY_FILE;
			img->provided = 1;
			img->size = (unsigned int)pfdh->size;

			if (snprintf(img->extract_file,
				     sizeof(img->extract_file), "%s%s",
				     destdir, pfdh->filename) >= (int)sizeof(img->extract_file)) {
				ERROR("Path too long: %s%s", destdir, pfdh->filename);
				return -EBADF;
			}
			/*
			 *  Streaming is possible to only one handler
			 *  If more img requires the same file,
			 *  sw-description contains an error
			 */
			if (install_direct) {
				ERROR("sw-description: stream to several handlers unsupported");
				return -EINVAL;
			}

			if (img->install_directly) {
				skip = INSTALL_FROM_STREAM;
				install_direct++;
			}

			*pimg = img;
		}
	}

	return skip;
}


/*
 * Extract all scripts from a list from the image
 * and save them on the filesystem to be executed later
 */
static int extract_scripts(struct imglist *head)
{
	struct img_type *script;
	int fdout;
	int ret = 0;
	const char* tmpdir_scripts = get_tmpdirscripts();

	LIST_FOREACH(script, head, next) {
		int fdin;
		char *tmpfile;
		unsigned long offset = 0;
		uint32_t checksum;

		if (!script->fname[0] && (script->provided == 0)) {
			TRACE("No script provided for script of type %s",
				script->type);
			continue;
		}
		if (script->provided == 0) {
			ERROR("Required script %s not found in image",
				script->fname);
			return -1;
		}

		snprintf(script->extract_file, sizeof(script->extract_file), "%s%s",
			 tmpdir_scripts , script->fname);

		fdout = openfileoutput(script->extract_file);
		if (fdout < 0)
			return fdout;


		if (asprintf(&tmpfile, "%s%s", get_tmpdir(), script->fname) ==
			ENOMEM_ASPRINTF) {
			ERROR("Path too long: %s%s", get_tmpdir(), script->fname);
			close(fdout);
			return -ENOMEM;
		}

		fdin = open(tmpfile, O_RDONLY);
		free(tmpfile);
		if (fdin < 0) {
			ERROR("Extracted script not found in %s: %s %d",
				get_tmpdir(), script->extract_file, errno);
			close(fdout);
			return -ENOENT;
		}

		ret = copyfile(fdin, &fdout, script->size, &offset, 0, 0,
				script->compressed,
				&checksum,
				script->sha256,
				script->is_encrypted,
				script->ivt_ascii,
				NULL);
		close(fdin);
		close(fdout);

		if (ret < 0)
			return ret;
	}
	return 0;
}

static int prepare_var_script(struct dict *dict, const char *script)
{
	int fd;
	struct dict_entry *bootvar;
	char buf[MAX_BOOT_SCRIPT_LINE_LENGTH];

	fd = openfileoutput(script);
	if (fd < 0) {
		ERROR("Temporary file %s cannot be opened for writing", script);
		return -1;
	}

	LIST_FOREACH(bootvar, dict, next) {
		char *key = dict_entry_get_key(bootvar);
		char *value = dict_entry_get_value(bootvar);

		if (!key || !value)
			continue;
		snprintf(buf, sizeof(buf), "%s=%s\n", key, value);
		if (write(fd, buf, strlen(buf)) != (ssize_t)strlen(buf)) {
			  TRACE("Error saving temporary bootloader environment file");
			  close(fd);
			  return -1;
		}
	}
	close(fd);

	return 0;
}

static int update_bootloader_env(struct swupdate_cfg *cfg, const char *script)
{
	int ret = 0;

	ret = prepare_var_script(&cfg->bootloader, script);
	if (ret)
		return ret;

	if ((ret = bootloader_apply_list(script)) < 0) {
		ERROR("Bootloader-specific error %d updating its environment", ret);
	}

	return ret;
}

static int update_swupdate_vars(struct swupdate_cfg *cfg, const char *script)
{
	int ret = 0;

	ret = prepare_var_script(&cfg->vars, script);
	if (ret)
		return ret;

	if ((ret = swupdate_vars_apply_list(script, cfg->namespace_for_vars)) < 0) {
		ERROR("Bootloader-specific error %d updating its environment", ret);
	}
	return ret;
}

static int run_prepost_scripts(struct imglist *list, script_fn type)
{
	int ret;
	struct img_type *img;
	struct installer_handler *hnd;

	/* Scripts must be run before installing images */
	LIST_FOREACH(img, list, next) {
		if (!img->is_script)
			continue;
		hnd = find_handler(img);
		if (hnd) {
			struct script_handler_data data = {
				.scriptfn = type,
				.data = hnd->data
			};

			swupdate_progress_inc_step(img->fname, hnd->desc);
			swupdate_progress_update(0);
			ret = hnd->installer(img, &data);
			swupdate_progress_update(100);
			swupdate_progress_step_completed();
			if (ret)
				return ret;
		}
	}

	return 0;
}

int install_single_image(struct img_type *img, bool dry_run)
{
	struct installer_handler *hnd;
	int ret;

	/*
	 * in case of dry run, replace the handler
	 * with a dummy doing nothing
	 */
	if (dry_run)
		strcpy(img->type, "dummy");

	hnd = find_handler(img);
	if (!hnd) {
		TRACE("Image Type %s not supported", img->type);
		return -1;
	}
	TRACE("Found installer for stream %s %s", img->fname, hnd->desc);

	swupdate_progress_inc_step(img->fname, hnd->desc);

	/* TODO : check callback to push results / progress */
	ret = hnd->installer(img, hnd->data);
	if (ret != 0) {
		TRACE("Installer for %s not successful !",
			hnd->desc);
	}

	swupdate_progress_step_completed();

	return ret;
}

/*
 * streamfd: file descriptor if it is required to extract
 *           images from the stream (update from file)
 * extract : boolean, true to enable extraction
 */

int install_images(struct swupdate_cfg *sw)
{
	int ret;
	struct img_type *img, *tmp;
	char *filename;
	struct stat buf;
	const char* TMPDIR = get_tmpdir();
	bool dry_run = sw->parms.dry_run;
	bool dropimg;

	/* Extract all scripts, preinstall scripts must be run now */
	const char* tmpdir_scripts = get_tmpdirscripts();
	ret = extract_scripts(&sw->scripts);
	ret |= extract_scripts(&sw->bootscripts);
	if (ret) {
		ERROR("extracting script to %s failed", tmpdir_scripts);
		return ret;
	}

	/* Scripts must be run before installing images */
	if (!dry_run) {
		ret = run_prepost_scripts(&sw->scripts, PREINSTALL);
		if (ret) {
			ERROR("execute preinstall scripts failed");
			return ret;
		}
	}

	LIST_FOREACH_SAFE(img, &sw->images, next, tmp) {

		dropimg = false;

		/*
		 *  If image is flagged to be installed from stream
		 *  it  was already installed by loading the
		 *  .swu image and it is skipped here.
		 *  This does not make sense when installed from file,
		 *  because images are seekd (no streaming)
		 */
		if (img->install_directly)
			continue;

		if (asprintf(&filename, "%s%s", TMPDIR, img->fname) ==
				ENOMEM_ASPRINTF) {
				ERROR("Path too long: %s%s", TMPDIR, img->fname);
				return -1;
		}

		ret = stat(filename, &buf);
		if (ret) {
			TRACE("%s not found or wrong", filename);
			free(filename);
			return -1;
		}
		img->size = buf.st_size;
		img->fdin = open(filename, O_RDONLY);
		free(filename);
		if (img->fdin < 0) {
			ERROR("Image %s cannot be opened",
			img->fname);
			return -1;
		}

		if ((strlen(img->path) > 0) &&
			(strlen(img->extract_file) > 0) &&
			(strncmp(img->path, img->extract_file, sizeof(img->path)) == 0)){
			struct img_type *tmpimg;
			WARN("Temporary and final location for %s is identical, skip "
			     "processing.", img->path);
			LIST_REMOVE(img, next);
			LIST_FOREACH(tmpimg, &sw->images, next) {
				if (strncmp(tmpimg->fname, img->fname, sizeof(img->fname)) == 0) {
					WARN("%s will be removed, it's referenced more "
					     "than once.", img->path);
					break;
				}
			}
			dropimg = true;
			ret = 0;
		} else {
			ret = install_single_image(img, dry_run);
		}

		close(img->fdin);

		if (dropimg)
			free_image(img);

		if (ret)
			return ret;
	}

	/*
	 * Skip scripts in dry-run mode
	 */
	if (dry_run) {
		return ret;
	}

	ret = run_prepost_scripts(&sw->scripts, POSTINSTALL);
	if (ret) {
		ERROR("execute postinstall scripts failed");
		return ret;
	}

	char* script = alloca(strlen(TMPDIR)+strlen(BOOT_SCRIPT_SUFFIX)+1);
	sprintf(script, "%s%s", TMPDIR, BOOT_SCRIPT_SUFFIX);

	if (!LIST_EMPTY(&sw->vars)) {
		ret = update_swupdate_vars(sw, script);
		if (ret) {
			return ret;
		}
	}

	if (!LIST_EMPTY(&sw->bootloader)) {
		ret = update_bootloader_env(sw, script);
		if (ret) {
			return ret;
		}
	}

	ret |= run_prepost_scripts(&sw->bootscripts, POSTINSTALL);

	return ret;
}

static void remove_sw_file(char __attribute__ ((__unused__)) *fname)
{
#ifndef CONFIG_NOCLEANUP
	/* yes, "best effort", the files need not necessarily exist */
	unlink(fname);
#endif
}

static void cleaup_img_entry(struct img_type *img)
{
	char *fn;
	const char *tmp[] = { get_tmpdirscripts(), get_tmpdir() };

	if (img->fname[0]) {
		for (unsigned int i = 0; i < ARRAY_SIZE(tmp); i++) {
			if (asprintf(&fn, "%s%s", tmp[i], img->fname) == ENOMEM_ASPRINTF) {
				ERROR("Path too long: %s%s", tmp[i], img->fname);
			} else {
				remove_sw_file(fn);
				free(fn);
			}
		}
	}
}

void free_image(struct img_type *img) {
	dict_drop_db(&img->properties);
	free(img);
}

void cleanup_files(struct swupdate_cfg *software) {
	char *fn;
	struct img_type *img;
	struct img_type *img_tmp;
	struct hw_type *hw;
	struct hw_type *hw_tmp;
	const char* TMPDIR = get_tmpdir();
	struct imglist *list[] = {&software->scripts, &software->bootscripts};

	LIST_FOREACH_SAFE(img, &software->images, next, img_tmp) {
		if (img->fname[0]) {
			if (asprintf(&fn, "%s%s", TMPDIR,
				     img->fname) == ENOMEM_ASPRINTF) {
				ERROR("Path too long: %s%s", TMPDIR, img->fname);
			}
			remove_sw_file(fn);
			free(fn);
		}
		LIST_REMOVE(img, next);
		free_image(img);
	}

	for (unsigned int count = 0; count < ARRAY_SIZE(list); count++) {
		LIST_FOREACH_SAFE(img, list[count], next, img_tmp) {
			cleaup_img_entry(img);

			LIST_REMOVE(img, next);
			free_image(img);
		}
	}

	/*
	 * drop environment databases
	 */
	dict_drop_db(&software->bootloader);
	dict_drop_db(&software->vars);

	if (asprintf(&fn, "%s%s", TMPDIR, BOOT_SCRIPT_SUFFIX) != ENOMEM_ASPRINTF) {
		remove_sw_file(fn);
		free(fn);
	}

	LIST_FOREACH_SAFE(hw, &software->hardware, next, hw_tmp) {
		LIST_REMOVE(hw, next);
		free(hw);
	}
	if (asprintf(&fn, "%s%s", TMPDIR, SW_DESCRIPTION_FILENAME) != ENOMEM_ASPRINTF) {
		remove_sw_file(fn);
		free(fn);
	}
#ifdef CONFIG_SIGNED_IMAGES
	if (asprintf(&fn, "%s%s.sig", TMPDIR, SW_DESCRIPTION_FILENAME) != ENOMEM_ASPRINTF) {
		remove_sw_file(fn);
		free(fn);
	}
#endif
}

int preupdatecmd(struct swupdate_cfg *swcfg)
{
	if (swcfg) {
		if (swcfg->parms.dry_run) {
			DEBUG("Dry run, skipping Pre-update command");
		} else {
			DEBUG("Running Pre-update command");
			return run_system_cmd(swcfg->preupdatecmd);
		}
	}

	return 0;
}

int postupdate(struct swupdate_cfg *swcfg, const char *info)
{
	swupdate_progress_done(info);

	if (swcfg) {
		if (swcfg->parms.dry_run) {
			DEBUG("Dry run, skipping Post-update command");
		} else {
			DEBUG("Running Post-update command");
			return run_system_cmd(swcfg->postupdatecmd);
		}

	}

	return 0;
}
