/* -*- Mode: C; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*- */

#include <monkey/mk_lib.h>

#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
#include <unistd.h>

#define API_ADDR   "127.0.0.1"
#define API_PORT   "8080"

/* Main context set as global so the signal handler can use it */
mk_ctx_t *ctx;

void cb_main(mk_request_t *request, void *data)
{
    (void) data;

    mk_http_status(request, 200);
    mk_http_header(request, "X-Monkey", 8, "OK", 2);

    mk_http_send(request, ":)\n", 3, NULL);
    mk_http_done(request);
}

void cb_test_chunks(mk_request_t *request, void *data)
{
    int i = 0;
    int len;
    char tmp[32];
    (void) data;

    mk_http_status(request, 200);
    mk_http_header(request, "X-Monkey", 8, "OK", 2);

    for (i = 0; i < 1000; i++) {
        len = snprintf(tmp, sizeof(tmp) -1, "test-chunk %6i\n ", i);
        mk_http_send(request, tmp, len, NULL);
    }
    mk_http_done(request);
}

void cb_test_big_chunk(mk_request_t *request, void *data)
{
    size_t chunk_size = 1024000000;
    char *chunk;
    (void) data;

    mk_http_status(request, 200);
    mk_http_header(request, "X-Monkey", 8, "OK", 2);

    chunk = calloc(1, chunk_size);
    mk_http_send(request, chunk, chunk_size, NULL);
    free(chunk);
    mk_http_done(request);
}


static void signal_handler(int signal)
{
    write(STDERR_FILENO, "[engine] caught signal\n", 23);

    switch (signal) {
    case SIGTERM:
    case SIGINT:
        mk_stop(ctx);
        mk_destroy(ctx);
        _exit(EXIT_SUCCESS);
    default:
        break;
    }
}

static void signal_init()
{
    signal(SIGINT,  &signal_handler);
    signal(SIGTERM, &signal_handler);
}

int main()
{
    int vid;

    signal_init();

    ctx = mk_create();
    if (!ctx) {
        return -1;
    }

    mk_config_set(ctx,
                  "Listen", API_PORT,
                  NULL);

    vid = mk_vhost_create(ctx, NULL);
    mk_vhost_set(ctx, vid,
                 "Name", "monotop",
                 NULL);
    mk_vhost_handler(ctx, vid, "/test_chunks", cb_test_chunks, NULL);
    mk_vhost_handler(ctx, vid, "/test_big_chunk", cb_test_big_chunk, NULL);
    mk_vhost_handler(ctx, vid, "/", cb_main, NULL);

    mk_info("Service: http://%s:%s/test_chunks",  API_ADDR, API_PORT);
    mk_start(ctx);

    sleep(3600);

    mk_stop(ctx);
    mk_destroy(ctx);

    return 0;
}
