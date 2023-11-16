/***
  This file is part of systemd.

  Copyright 2015 Lennart Poettering

  systemd is free software; you can redistribute it and/or modify it
  under the terms of the GNU Lesser General Public License as published by
  the Free Software Foundation; either version 2.1 of the License, or
  (at your option) any later version.

  systemd is distributed in the hope that it will be useful, but
  WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
  Lesser General Public License for more details.

  You should have received a copy of the GNU Lesser General Public License
  along with systemd; If not, see <http://www.gnu.org/licenses/>.
***/

#include <sys/prctl.h>

#include "btrfs-util.h"
#include "capability.h"
#include "copy.h"
#include "import-common.h"
#include "import-job.h"
#include "strv.h"
#include "util.h"

#define FILENAME_ESCAPE "/.#\"\'"

int
import_find_old_etags(const char *url, const char *image_root, int dt,
	const char *prefix, const char *suffix, char ***etags)
{
	_cleanup_free_ char *escaped_url = NULL;
	_cleanup_closedir_ DIR *d = NULL;
	_cleanup_strv_free_ char **l = NULL;
	struct dirent *de;
	int r;

	assert(url);
	assert(etags);

	if (!image_root)
		image_root = "/var/lib/machines";

	escaped_url = xescape(url, FILENAME_ESCAPE);
	if (!escaped_url)
		return -ENOMEM;

	d = opendir(image_root);
	if (!d) {
		if (errno == ENOENT) {
			*etags = NULL;
			return 0;
		}

		return -errno;
	}

	FOREACH_DIRENT_ALL (de, d, return -errno) {
		const char *a, *b;
		char *u;

		if (de->d_type != DT_UNKNOWN && de->d_type != dt)
			continue;

		if (prefix) {
			a = startswith(de->d_name, prefix);
			if (!a)
				continue;
		} else
			a = de->d_name;

		a = startswith(a, escaped_url);
		if (!a)
			continue;

		a = startswith(a, ".");
		if (!a)
			continue;

		if (suffix) {
			b = endswith(de->d_name, suffix);
			if (!b)
				continue;
		} else
			b = strchr(de->d_name, 0);

		if (a >= b)
			continue;

		u = cunescape_length(a, b - a);
		if (!u)
			return -ENOMEM;

		if (!http_etag_is_valid(u)) {
			free(u);
			continue;
		}

		r = strv_consume(&l, u);
		if (r < 0)
			return r;
	}

	*etags = l;
	l = NULL;

	return 0;
}

int
import_make_local_copy(const char *final, const char *image_root,
	const char *local, bool force_local)
{
	const char *p;
	int r;

	assert(final);
	assert(local);

	if (!image_root)
		image_root = "/var/lib/machines";

	p = strjoina(image_root, "/", local);

	if (force_local) {
		(void)btrfs_subvol_remove(p);
		(void)rm_rf_dangerous(p, false, true, false);
	}

	r = btrfs_subvol_snapshot(final, p, false, false);
	if (r == -ENOTTY) {
		r = copy_tree(final, p, false);
		if (r < 0)
			return log_error_errno(r, "Failed to copy image: %m");
	} else if (r < 0)
		return log_error_errno(r, "Failed to create local image: %m");

	log_info("Created new local image '%s'.", local);

	return 0;
}

int
import_make_read_only_fd(int fd)
{
	int r;

	assert(fd >= 0);

	/* First, let's make this a read-only subvolume if it refers
         * to a subvolume */
	r = btrfs_subvol_set_read_only_fd(fd, true);
	if (r == -ENOTTY || r == -ENOTDIR || r == -EINVAL) {
		struct stat st;

		/* This doesn't refer to a subvolume, or the file
                 * system isn't even btrfs. In that, case fall back to
                 * chmod()ing */

		r = fstat(fd, &st);
		if (r < 0)
			return log_error_errno(errno,
				"Failed to stat temporary image: %m");

		/* Drop "w" flag */
		if (fchmod(fd, st.st_mode & 07555) < 0)
			return log_error_errno(errno,
				"Failed to chmod() final image: %m");

		return 0;

	} else if (r < 0)
		return log_error_errno(r,
			"Failed to make subvolume read-only: %m");

	return 0;
}

int
import_make_read_only(const char *path)
{
	_cleanup_close_ int fd = 1;

	fd = open(path, O_RDONLY | O_NOCTTY | O_CLOEXEC);
	if (fd < 0)
		return log_error_errno(errno, "Failed to open %s: %m", path);

	return import_make_read_only_fd(fd);
}

int
import_make_path(const char *url, const char *etag, const char *image_root,
	const char *prefix, const char *suffix, char **ret)
{
	_cleanup_free_ char *escaped_url = NULL;
	char *path;

	assert(url);
	assert(ret);

	if (!image_root)
		image_root = "/var/lib/machines";

	escaped_url = xescape(url, FILENAME_ESCAPE);
	if (!escaped_url)
		return -ENOMEM;

	if (etag) {
		_cleanup_free_ char *escaped_etag = NULL;

		escaped_etag = xescape(etag, FILENAME_ESCAPE);
		if (!escaped_etag)
			return -ENOMEM;

		path = strjoin(image_root, "/", strempty(prefix), escaped_url,
			".", escaped_etag, strempty(suffix), NULL);
	} else
		path = strjoin(image_root, "/", strempty(prefix), escaped_url,
			strempty(suffix), NULL);
	if (!path)
		return -ENOMEM;

	*ret = path;
	return 0;
}

int
import_make_verification_jobs(ImportJob **ret_checksum_job,
	ImportJob **ret_signature_job, ImportVerify verify, const char *url,
	CurlGlue *glue, ImportJobFinished on_finished, void *userdata)
{
	_cleanup_(import_job_unrefp) ImportJob *checksum_job = NULL,
					       *signature_job = NULL;
	int r;

	assert(ret_checksum_job);
	assert(ret_signature_job);
	assert(verify >= 0);
	assert(verify < _IMPORT_VERIFY_MAX);
	assert(url);
	assert(glue);

	if (verify != IMPORT_VERIFY_NO) {
		_cleanup_free_ char *checksum_url = NULL;

		/* Queue job for the SHA256SUMS file for the image */
		r = import_url_change_last_component(url, "SHA256SUMS",
			&checksum_url);
		if (r < 0)
			return r;

		r = import_job_new(&checksum_job, checksum_url, glue, userdata);
		if (r < 0)
			return r;

		checksum_job->on_finished = on_finished;
		checksum_job->uncompressed_max = checksum_job->compressed_max =
			1ULL * 1024ULL * 1024ULL;
	}

	if (verify == IMPORT_VERIFY_SIGNATURE) {
		_cleanup_free_ char *signature_url = NULL;

		/* Queue job for the SHA256SUMS.gpg file for the image. */
		r = import_url_change_last_component(url, "SHA256SUMS.gpg",
			&signature_url);
		if (r < 0)
			return r;

		r = import_job_new(&signature_job, signature_url, glue,
			userdata);
		if (r < 0)
			return r;

		signature_job->on_finished = on_finished;
		signature_job->uncompressed_max =
			signature_job->compressed_max =
				1ULL * 1024ULL * 1024ULL;
	}

	*ret_checksum_job = checksum_job;
	*ret_signature_job = signature_job;

	checksum_job = signature_job = NULL;

	return 0;
}

int
import_verify(ImportJob *main_job, ImportJob *checksum_job,
	ImportJob *signature_job)
{
	_cleanup_close_pair_ int gpg_pipe[2] = { -1, -1 };
	_cleanup_free_ char *fn = NULL;
	_cleanup_close_ int sig_file = -1;
	const char *p, *line;
	char sig_file_path[] = "/tmp/sigXXXXXX",
	     gpg_home[] = "/tmp/gpghomeXXXXXX";
	_cleanup_sigkill_wait_ pid_t pid = 0;
	bool gpg_home_created = false;
	int r;

	assert(main_job);
	assert(main_job->state == IMPORT_JOB_DONE);

	if (!checksum_job)
		return 0;

	assert(main_job->calc_checksum);
	assert(main_job->checksum);
	assert(checksum_job->state == IMPORT_JOB_DONE);

	if (!checksum_job->payload || checksum_job->payload_size <= 0) {
		log_error("Checksum is empty, cannot verify.");
		return -EBADMSG;
	}

	r = import_url_last_component(main_job->url, &fn);
	if (r < 0)
		return log_oom();

	if (!filename_is_valid(fn)) {
		log_error(
			"Cannot verify checksum, could not determine valid server-side file name.");
		return -EBADMSG;
	}

	line = strjoina(main_job->checksum, " *", fn, "\n");

	p = memmem(checksum_job->payload, checksum_job->payload_size, line,
		strlen(line));

	if (!p || (p != (char *)checksum_job->payload && p[-1] != '\n')) {
		log_error(
			"Checksum did not check out, payload has been tempered with.");
		return -EBADMSG;
	}

	log_info("SHA256 checksum of %s is valid.", main_job->url);

	if (!signature_job)
		return 0;

	assert(signature_job->state == IMPORT_JOB_DONE);

	if (!signature_job->payload || signature_job->payload_size <= 0) {
		log_error("Signature is empty, cannot verify.");
		return -EBADMSG;
	}

	r = pipe2(gpg_pipe, O_CLOEXEC);
	if (r < 0)
		return log_error_errno(errno,
			"Failed to create pipe for gpg: %m");

	sig_file = mkostemp(sig_file_path, O_RDWR);
	if (sig_file < 0)
		return log_error_errno(errno,
			"Failed to create temporary file: %m");

	r = loop_write(sig_file, signature_job->payload,
		signature_job->payload_size, false);
	if (r < 0) {
		log_error_errno(r, "Failed to write to temporary file: %m");
		goto finish;
	}

	if (!mkdtemp(gpg_home)) {
		r = log_error_errno(errno,
			"Failed to create tempory home for gpg: %m");
		goto finish;
	}

	gpg_home_created = true;

	pid = fork();
	if (pid < 0)
		return log_error_errno(errno, "Failed to fork off gpg: %m");
	if (pid == 0) {
		const char *cmd[] = {
			"gpg", "--no-options", "--no-default-keyring",
			"--no-auto-key-locate", "--no-auto-check-trustdb",
			"--batch", "--trust-model=always",
			NULL, /* --homedir=  */
			NULL, /* --keyring= */
			NULL, /* --verify */
			NULL, /* signature file */
			NULL, /* dash */
			NULL /* trailing NULL */
		};
		unsigned k = ELEMENTSOF(cmd) - 6;
		int null_fd;

		/* Child */

		reset_all_signal_handlers();
		reset_signal_mask();
		assert_se(prctl(PR_SET_PDEATHSIG, SIGTERM) == 0);

		gpg_pipe[1] = safe_close(gpg_pipe[1]);

		if (dup2(gpg_pipe[0], STDIN_FILENO) != STDIN_FILENO) {
			log_error_errno(errno, "Failed to dup2() fd: %m");
			_exit(EXIT_FAILURE);
		}

		if (gpg_pipe[0] != STDIN_FILENO)
			gpg_pipe[0] = safe_close(gpg_pipe[0]);

		null_fd = open("/dev/null", O_WRONLY | O_NOCTTY);
		if (null_fd < 0) {
			log_error_errno(errno, "Failed to open /dev/null: %m");
			_exit(EXIT_FAILURE);
		}

		if (dup2(null_fd, STDOUT_FILENO) != STDOUT_FILENO) {
			log_error_errno(errno, "Failed to dup2() fd: %m");
			_exit(EXIT_FAILURE);
		}

		if (null_fd != STDOUT_FILENO)
			null_fd = safe_close(null_fd);

		cmd[k++] = strjoina("--homedir=", gpg_home);

		/* We add the user keyring only to the command line
                 * arguments, if it's around since gpg fails
                 * otherwise. */
		if (access(USER_KEYRING_PATH, F_OK) >= 0)
			cmd[k++] = "--keyring=" USER_KEYRING_PATH;
		else
			cmd[k++] = "--keyring=" VENDOR_KEYRING_PATH;

		cmd[k++] = "--verify";
		cmd[k++] = sig_file_path;
		cmd[k++] = "-";
		cmd[k++] = NULL;

		fd_cloexec(STDIN_FILENO, false);
		fd_cloexec(STDOUT_FILENO, false);
		fd_cloexec(STDERR_FILENO, false);

		execvp("gpg2", (char *const *)cmd);
		execvp("gpg", (char *const *)cmd);
		log_error_errno(errno, "Failed to execute gpg: %m");
		_exit(EXIT_FAILURE);
	}

	gpg_pipe[0] = safe_close(gpg_pipe[0]);

	r = loop_write(gpg_pipe[1], checksum_job->payload,
		checksum_job->payload_size, false);
	if (r < 0) {
		log_error_errno(r, "Failed to write to pipe: %m");
		goto finish;
	}

	gpg_pipe[1] = safe_close(gpg_pipe[1]);

	r = wait_for_terminate_and_warn("gpg", pid, true);
	pid = 0;
	if (r < 0)
		goto finish;
	if (r > 0) {
		log_error("Signature verification failed.");
		r = -EBADMSG;
	} else {
		log_info("Signature verification succeeded.");
		r = 0;
	}

finish:
	(void)unlink(sig_file_path);

	if (gpg_home_created)
		rm_rf_dangerous(gpg_home, false, true, false);

	return r;
}

int
import_fork_tar(const char *path, pid_t *ret)
{
	_cleanup_close_pair_ int pipefd[2] = { -1, -1 };
	pid_t pid;
	int r;

	assert(path);
	assert(ret);

	if (pipe2(pipefd, O_CLOEXEC) < 0)
		return log_error_errno(errno,
			"Failed to create pipe for tar: %m");

	pid = fork();
	if (pid < 0)
		return log_error_errno(errno, "Failed to fork off tar: %m");

	if (pid == 0) {
		int null_fd;
		uint64_t retain = (1ULL << CAP_CHOWN) | (1ULL << CAP_FOWNER) |
			(1ULL << CAP_FSETID) | (1ULL << CAP_MKNOD) |
			(1ULL << CAP_SETFCAP) | (1ULL << CAP_DAC_OVERRIDE);

		/* Child */

		reset_all_signal_handlers();
		reset_signal_mask();
		assert_se(prctl(PR_SET_PDEATHSIG, SIGTERM) == 0);

		pipefd[1] = safe_close(pipefd[1]);

		if (dup2(pipefd[0], STDIN_FILENO) != STDIN_FILENO) {
			log_error_errno(errno, "Failed to dup2() fd: %m");
			_exit(EXIT_FAILURE);
		}

		if (pipefd[0] != STDIN_FILENO)
			pipefd[0] = safe_close(pipefd[0]);

		null_fd = open("/dev/null", O_WRONLY | O_NOCTTY);
		if (null_fd < 0) {
			log_error_errno(errno, "Failed to open /dev/null: %m");
			_exit(EXIT_FAILURE);
		}

		if (dup2(null_fd, STDOUT_FILENO) != STDOUT_FILENO) {
			log_error_errno(errno, "Failed to dup2() fd: %m");
			_exit(EXIT_FAILURE);
		}

		if (null_fd != STDOUT_FILENO)
			null_fd = safe_close(null_fd);

		fd_cloexec(STDIN_FILENO, false);
		fd_cloexec(STDOUT_FILENO, false);
		fd_cloexec(STDERR_FILENO, false);

		if (unshare(CLONE_NEWNET) < 0)
			log_error_errno(errno,
				"Failed to lock tar into network namespace, ignoring: %m");

		r = capability_bounding_set_drop(retain, true);
		if (r < 0)
			log_error_errno(r,
				"Failed to drop capabilities, ignoring: %m");

		execlp("tar", "tar", "--numeric-owner", "-C", path, "-px",
			NULL);
		log_error_errno(errno, "Failed to execute tar: %m");
		_exit(EXIT_FAILURE);
	}

	pipefd[0] = safe_close(pipefd[0]);
	r = pipefd[1];
	pipefd[1] = -1;

	*ret = pid;

	return r;
}
