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

#include "import-job.h"
#include "bsdxattr.h"
#include "strv.h"

ImportJob *
import_job_unref(ImportJob *j)
{
	if (!j)
		return NULL;

	curl_glue_remove_and_free(j->glue, j->curl);
	curl_slist_free_all(j->request_header);

	safe_close(j->disk_fd);

	if (j->compressed == IMPORT_JOB_XZ)
		lzma_end(&j->xz);
	else if (j->compressed == IMPORT_JOB_GZIP)
		inflateEnd(&j->gzip);
	else if (j->compressed == IMPORT_JOB_BZIP2)
		BZ2_bzDecompressEnd(&j->bzip2);

	if (j->checksum_context)
		gcry_md_close(j->checksum_context);

	free(j->url);
	free(j->etag);
	strv_free(j->old_etags);
	free(j->payload);
	free(j->checksum);

	free(j);

	return NULL;
}

static void
import_job_finish(ImportJob *j, int ret)
{
	assert(j);

	if (j->state == IMPORT_JOB_DONE || j->state == IMPORT_JOB_FAILED)
		return;

	if (ret == 0) {
		j->state = IMPORT_JOB_DONE;
		j->progress_percent = 100;
		log_info("Download of %s complete.", j->url);
	} else {
		j->state = IMPORT_JOB_FAILED;
		j->error = ret;
	}

	if (j->on_finished)
		j->on_finished(j);
}

void
import_job_curl_on_finished(CurlGlue *g, CURL *curl, CURLcode result)
{
	ImportJob *j = NULL;
	CURLcode code;
	long status;
	int r;

	if (curl_easy_getinfo(curl, CURLINFO_PRIVATE, (char **)&j) != CURLE_OK)
		return;

	if (!j || j->state == IMPORT_JOB_DONE || j->state == IMPORT_JOB_FAILED)
		return;

	if (result != CURLE_OK) {
		log_error("Transfer failed: %s", curl_easy_strerror(result));
		r = -EIO;
		goto finish;
	}

	code = curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &status);
	if (code != CURLE_OK) {
		log_error("Failed to retrieve response code: %s",
			curl_easy_strerror(code));
		r = -EIO;
		goto finish;
	} else if (status == 304) {
		log_info("Image already downloaded. Skipping download.");
		j->etag_exists = true;
		r = 0;
		goto finish;
	} else if (status >= 300) {
		log_error("HTTP request to %s failed with code %li.", j->url,
			status);
		r = -EIO;
		goto finish;
	} else if (status < 200) {
		log_error(
			"HTTP request to %s finished with unexpected code %li.",
			j->url, status);
		r = -EIO;
		goto finish;
	}

	if (j->state != IMPORT_JOB_RUNNING) {
		log_error("Premature connection termination.");
		r = -EIO;
		goto finish;
	}

	if (j->content_length != (uint64_t)-1 &&
		j->content_length != j->written_compressed) {
		log_error("Download truncated.");
		r = -EIO;
		goto finish;
	}

	if (j->checksum_context) {
		uint8_t *k;

		k = gcry_md_read(j->checksum_context, GCRY_MD_SHA256);
		if (!k) {
			log_error("Failed to get checksum.");
			r = -EIO;
			goto finish;
		}

		j->checksum = hexmem(k, gcry_md_get_algo_dlen(GCRY_MD_SHA256));
		if (!j->checksum) {
			r = log_oom();
			goto finish;
		}

		log_debug("SHA256 of %s is %s.", j->url, j->checksum);
	}

	if (j->disk_fd >= 0 && j->allow_sparse) {
		/* Make sure the file size is right, in case the file was
                 * sparse and we just seeked for the last part */

		if (ftruncate(j->disk_fd, j->written_uncompressed) < 0) {
			log_error_errno(errno, "Failed to truncate file: %m");
			r = -errno;
			goto finish;
		}

		if (j->etag)
			(void)fsetxattr(j->disk_fd, "user.source_etag", j->etag,
				strlen(j->etag), 0);
		if (j->url)
			(void)fsetxattr(j->disk_fd, "user.source_url", j->url,
				strlen(j->url), 0);

		if (j->mtime != 0) {
			struct timespec ut[2];

			timespec_store(&ut[0], j->mtime);
			ut[1] = ut[0];
			(void)futimens(j->disk_fd, ut);

			(void)fd_setcrtime(j->disk_fd, j->mtime);
		}
	}

	r = 0;

finish:
	import_job_finish(j, r);
}

static int
import_job_write_uncompressed(ImportJob *j, void *p, size_t sz)
{
	ssize_t n;

	assert(j);
	assert(p);

	if (sz <= 0)
		return 0;

	if (j->written_uncompressed + sz < j->written_uncompressed) {
		log_error("File too large, overflow");
		return -EOVERFLOW;
	}

	if (j->written_uncompressed + sz > j->uncompressed_max) {
		log_error("File overly large, refusing");
		return -EFBIG;
	}

	if (j->disk_fd >= 0) {
		if (j->allow_sparse)
			n = sparse_write(j->disk_fd, p, sz, 64);
		else
			n = write(j->disk_fd, p, sz);
		if (n < 0) {
			log_error_errno(errno, "Failed to write file: %m");
			return -errno;
		}
		if ((size_t)n < sz) {
			log_error("Short write");
			return -EIO;
		}
	} else {
		if (!GREEDY_REALLOC(j->payload, j->payload_allocated,
			    j->payload_size + sz))
			return log_oom();

		memcpy(j->payload + j->payload_size, p, sz);
		j->payload_size += sz;
	}

	j->written_uncompressed += sz;

	return 0;
}

static int
import_job_write_compressed(ImportJob *j, void *p, size_t sz)
{
	int r;

	assert(j);
	assert(p);

	if (sz <= 0)
		return 0;

	if (j->written_compressed + sz < j->written_compressed) {
		log_error("File too large, overflow");
		return -EOVERFLOW;
	}

	if (j->written_compressed + sz > j->compressed_max) {
		log_error("File overly large, refusing.");
		return -EFBIG;
	}

	if (j->content_length != (uint64_t)-1 &&
		j->written_compressed + sz > j->content_length) {
		log_error("Content length incorrect.");
		return -EFBIG;
	}

	if (j->checksum_context)
		gcry_md_write(j->checksum_context, p, sz);

	switch (j->compressed) {
	case IMPORT_JOB_UNCOMPRESSED:
		r = import_job_write_uncompressed(j, p, sz);
		if (r < 0)
			return r;

		break;

	case IMPORT_JOB_XZ:
		j->xz.next_in = p;
		j->xz.avail_in = sz;

		while (j->xz.avail_in > 0) {
			uint8_t buffer[16 * 1024];
			lzma_ret lzr;

			j->xz.next_out = buffer;
			j->xz.avail_out = sizeof(buffer);

			lzr = lzma_code(&j->xz, LZMA_RUN);
			if (lzr != LZMA_OK && lzr != LZMA_STREAM_END) {
				log_error("Decompression error.");
				return -EIO;
			}

			r = import_job_write_uncompressed(j, buffer,
				sizeof(buffer) - j->xz.avail_out);
			if (r < 0)
				return r;
		}

		break;

	case IMPORT_JOB_GZIP:
		j->gzip.next_in = p;
		j->gzip.avail_in = sz;

		while (j->gzip.avail_in > 0) {
			uint8_t buffer[16 * 1024];

			j->gzip.next_out = buffer;
			j->gzip.avail_out = sizeof(buffer);

			r = inflate(&j->gzip, Z_NO_FLUSH);
			if (r != Z_OK && r != Z_STREAM_END) {
				log_error("Decompression error.");
				return -EIO;
			}

			r = import_job_write_uncompressed(j, buffer,
				sizeof(buffer) - j->gzip.avail_out);
			if (r < 0)
				return r;
		}

		break;

	case IMPORT_JOB_BZIP2:
		j->bzip2.next_in = p;
		j->bzip2.avail_in = sz;

		while (j->bzip2.avail_in > 0) {
			uint8_t buffer[16 * 1024];

			j->bzip2.next_out = (char *)buffer;
			j->bzip2.avail_out = sizeof(buffer);

			r = BZ2_bzDecompress(&j->bzip2);
			if (r != BZ_OK && r != BZ_STREAM_END) {
				log_error("Decompression error.");
				return -EIO;
			}

			r = import_job_write_uncompressed(j, buffer,
				sizeof(buffer) - j->bzip2.avail_out);
			if (r < 0)
				return r;
		}

		break;

	default:
		assert_not_reached("Unknown compression");
	}

	j->written_compressed += sz;

	return 0;
}

static int
import_job_open_disk(ImportJob *j)
{
	int r;

	assert(j);

	if (j->on_open_disk) {
		r = j->on_open_disk(j);
		if (r < 0)
			return r;
	}

	if (j->disk_fd >= 0) {
		/* Check if we can do sparse files */

		if (lseek(j->disk_fd, SEEK_SET, 0) == 0)
			j->allow_sparse = true;
		else {
			if (errno != ESPIPE)
				return log_error_errno(errno,
					"Failed to seek on file descriptor: %m");

			j->allow_sparse = false;
		}
	}

	if (j->calc_checksum) {
		if (gcry_md_open(&j->checksum_context, GCRY_MD_SHA256, 0) !=
			0) {
			log_error("Failed to initialize hash context.");
			return -EIO;
		}
	}

	return 0;
}

static int
import_job_detect_compression(ImportJob *j)
{
	static const uint8_t xz_signature[] = { 0xfd, '7', 'z', 'X', 'Z',
		0x00 };
	static const uint8_t gzip_signature[] = { 0x1f, 0x8b };
	static const uint8_t bzip2_signature[] = { 'B', 'Z', 'h' };

	_cleanup_free_ uint8_t *stub = NULL;
	size_t stub_size;

	int r;

	assert(j);

	if (j->payload_size < MAX3(sizeof(xz_signature), sizeof(gzip_signature),
				      sizeof(bzip2_signature)))
		return 0;

	if (memcmp(j->payload, xz_signature, sizeof(xz_signature)) == 0)
		j->compressed = IMPORT_JOB_XZ;
	else if (memcmp(j->payload, gzip_signature, sizeof(gzip_signature)) ==
		0)
		j->compressed = IMPORT_JOB_GZIP;
	else if (memcmp(j->payload, bzip2_signature, sizeof(bzip2_signature)) ==
		0)
		j->compressed = IMPORT_JOB_BZIP2;
	else
		j->compressed = IMPORT_JOB_UNCOMPRESSED;

	log_debug("Stream is XZ compressed: %s",
		yes_no(j->compressed == IMPORT_JOB_XZ));
	log_debug("Stream is GZIP compressed: %s",
		yes_no(j->compressed == IMPORT_JOB_GZIP));
	log_debug("Stream is BZIP2 compressed: %s",
		yes_no(j->compressed == IMPORT_JOB_BZIP2));

	if (j->compressed == IMPORT_JOB_XZ) {
		lzma_ret xzr;

		xzr = lzma_stream_decoder(&j->xz, UINT64_MAX,
			LZMA_TELL_UNSUPPORTED_CHECK);
		if (xzr != LZMA_OK) {
			log_error("Failed to initialize XZ decoder.");
			return -EIO;
		}
	}
	if (j->compressed == IMPORT_JOB_GZIP) {
		r = inflateInit2(&j->gzip, 15 + 16);
		if (r != Z_OK) {
			log_error("Failed to initialize gzip decoder.");
			return -EIO;
		}
	}
	if (j->compressed == IMPORT_JOB_BZIP2) {
		r = BZ2_bzDecompressInit(&j->bzip2, 0, 0);
		if (r != BZ_OK) {
			log_error("Failed to initialize bzip2 decoder.");
			return -EIO;
		}
	}

	r = import_job_open_disk(j);
	if (r < 0)
		return r;

	/* Now, take the payload we read so far, and decompress it */
	stub = j->payload;
	stub_size = j->payload_size;

	j->payload = NULL;
	j->payload_size = 0;
	j->payload_allocated = 0;

	j->state = IMPORT_JOB_RUNNING;

	r = import_job_write_compressed(j, stub, stub_size);
	if (r < 0)
		return r;

	return 0;
}

static size_t
import_job_write_callback(void *contents, size_t size, size_t nmemb,
	void *userdata)
{
	ImportJob *j = userdata;
	size_t sz = size * nmemb;
	int r;

	assert(contents);
	assert(j);

	switch (j->state) {
	case IMPORT_JOB_ANALYZING:
		/* Let's first check what it actually is */

		if (!GREEDY_REALLOC(j->payload, j->payload_allocated,
			    j->payload_size + sz)) {
			r = log_oom();
			goto fail;
		}

		memcpy(j->payload + j->payload_size, contents, sz);
		j->payload_size += sz;

		r = import_job_detect_compression(j);
		if (r < 0)
			goto fail;

		break;

	case IMPORT_JOB_RUNNING:

		r = import_job_write_compressed(j, contents, sz);
		if (r < 0)
			goto fail;

		break;

	case IMPORT_JOB_DONE:
	case IMPORT_JOB_FAILED:
		r = -ESTALE;
		goto fail;

	default:
		assert_not_reached("Impossible state.");
	}

	return sz;

fail:
	import_job_finish(j, r);
	return 0;
}

static size_t
import_job_header_callback(void *contents, size_t size, size_t nmemb,
	void *userdata)
{
	ImportJob *j = userdata;
	size_t sz = size * nmemb;
	_cleanup_free_ char *length = NULL, *last_modified = NULL;
	char *etag;
	int r;

	assert(contents);
	assert(j);

	if (j->state == IMPORT_JOB_DONE || j->state == IMPORT_JOB_FAILED) {
		r = -ESTALE;
		goto fail;
	}

	assert(j->state == IMPORT_JOB_ANALYZING);

	r = curl_header_strdup(contents, sz, "ETag:", &etag);
	if (r < 0) {
		log_oom();
		goto fail;
	}
	if (r > 0) {
		free(j->etag);
		j->etag = etag;

		if (strv_contains(j->old_etags, j->etag)) {
			log_info(
				"Image already downloaded. Skipping download.");
			j->etag_exists = true;
			import_job_finish(j, 0);
			return sz;
		}

		return sz;
	}

	r = curl_header_strdup(contents, sz, "Content-Length:", &length);
	if (r < 0) {
		log_oom();
		goto fail;
	}
	if (r > 0) {
		(void)safe_atou64(length, &j->content_length);

		if (j->content_length != (uint64_t)-1) {
			char bytes[FORMAT_BYTES_MAX];

			if (j->content_length > j->compressed_max) {
				log_error("Content too large.");
				r = -EFBIG;
				goto fail;
			}

			log_info("Downloading %s for %s.",
				format_bytes(bytes, sizeof(bytes),
					j->content_length),
				j->url);
		}

		return sz;
	}

	r = curl_header_strdup(contents, sz, "Last-Modified:", &last_modified);
	if (r < 0) {
		log_oom();
		goto fail;
	}
	if (r > 0) {
		(void)curl_parse_http_time(last_modified, &j->mtime);
		return sz;
	}

	if (j->on_header) {
		r = j->on_header(j, contents, sz);
		if (r < 0)
			goto fail;
	}

	return sz;

fail:
	import_job_finish(j, r);
	return 0;
}

static int
import_job_progress_callback(void *userdata, double dltotal, double dlnow,
	double ultotal, double ulnow)
{
	ImportJob *j = userdata;
	unsigned percent;
	usec_t n;

	assert(j);

	if (dltotal <= 0)
		return 0;

	percent = ((100 * dlnow) / dltotal);
	n = now(CLOCK_MONOTONIC);

	if (n > j->last_status_usec + USEC_PER_SEC &&
		percent != j->progress_percent && dlnow < dltotal) {
		char buf[FORMAT_TIMESPAN_MAX];

		if (n - j->start_usec > USEC_PER_SEC && dlnow > 0) {
			char y[FORMAT_BYTES_MAX];
			usec_t left, done;

			done = n - j->start_usec;
			left = (usec_t)(((double)done * (double)dltotal) /
				       dlnow) -
				done;

			log_info("Got %u%% of %s. %s left at %s/s.", percent,
				j->url,
				format_timespan(buf, sizeof(buf), left,
					USEC_PER_SEC),
				format_bytes(y, sizeof(y),
					(uint64_t)((double)dlnow /
						((double)done /
							(double)USEC_PER_SEC))));
		} else
			log_info("Got %u%% of %s.", percent, j->url);

		j->progress_percent = percent;
		j->last_status_usec = n;

		if (j->on_progress)
			j->on_progress(j);
	}

	return 0;
}

int
import_job_new(ImportJob **ret, const char *url, CurlGlue *glue, void *userdata)
{
	_cleanup_(import_job_unrefp) ImportJob *j = NULL;

	assert(url);
	assert(glue);
	assert(ret);

	j = new0(ImportJob, 1);
	if (!j)
		return -ENOMEM;

	j->state = IMPORT_JOB_INIT;
	j->disk_fd = -1;
	j->userdata = userdata;
	j->glue = glue;
	j->content_length = (uint64_t)-1;
	j->start_usec = now(CLOCK_MONOTONIC);
	j->compressed_max = j->uncompressed_max =
		8LLU * 1024LLU * 1024LLU * 1024LLU; /* 8GB */

	j->url = strdup(url);
	if (!j->url)
		return -ENOMEM;

	*ret = j;
	j = NULL;

	return 0;
}

int
import_job_begin(ImportJob *j)
{
	int r;

	assert(j);

	if (j->state != IMPORT_JOB_INIT)
		return -EBUSY;

	r = curl_glue_make(&j->curl, j->url, j);
	if (r < 0)
		return r;

	if (!strv_isempty(j->old_etags)) {
		_cleanup_free_ char *cc = NULL, *hdr = NULL;

		cc = strv_join(j->old_etags, ", ");
		if (!cc)
			return -ENOMEM;

		hdr = strappend("If-None-Match: ", cc);
		if (!hdr)
			return -ENOMEM;

		if (!j->request_header) {
			j->request_header = curl_slist_new(hdr, NULL);
			if (!j->request_header)
				return -ENOMEM;
		} else {
			struct curl_slist *l;

			l = curl_sIWLIST_APPEND(j->request_header, hdr);
			if (!l)
				return -ENOMEM;

			j->request_header = l;
		}
	}

	if (j->request_header) {
		if (curl_easy_setopt(j->curl, CURLOPT_HTTPHEADER,
			    j->request_header) != CURLE_OK)
			return -EIO;
	}

	if (curl_easy_setopt(j->curl, CURLOPT_WRITEFUNCTION,
		    import_job_write_callback) != CURLE_OK)
		return -EIO;

	if (curl_easy_setopt(j->curl, CURLOPT_WRITEDATA, j) != CURLE_OK)
		return -EIO;

	if (curl_easy_setopt(j->curl, CURLOPT_HEADERFUNCTION,
		    import_job_header_callback) != CURLE_OK)
		return -EIO;

	if (curl_easy_setopt(j->curl, CURLOPT_HEADERDATA, j) != CURLE_OK)
		return -EIO;

	if (curl_easy_setopt(j->curl, CURLOPT_PROGRESSFUNCTION,
		    import_job_progress_callback) != CURLE_OK)
		return -EIO;

	if (curl_easy_setopt(j->curl, CURLOPT_PROGRESSDATA, j) != CURLE_OK)
		return -EIO;

	if (curl_easy_setopt(j->curl, CURLOPT_NOPROGRESS, 0) != CURLE_OK)
		return -EIO;

	r = curl_glue_add(j->glue, j->curl);
	if (r < 0)
		return r;

	j->state = IMPORT_JOB_ANALYZING;

	return 0;
}
