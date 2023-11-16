
/*
 * machinarium.
 *
 * Cooperative multitasking engine.
 */

#include <machinarium.h>

static int ops = 0;

static void benchmark_reader(void *arg)
{
	machine_channel_t *channel = arg;
	while (machine_active()) {
		machine_msg_t *msg;
		msg = machine_channel_read(channel, UINT32_MAX);
		if (msg)
			machine_msg_free(msg);
		ops++;
	}
}

static void benchmark_writer(void *arg)
{
	machine_channel_t *channel = arg;
	while (machine_active()) {
		machine_msg_t *msg;
		msg = machine_msg_create(0);
		machine_channel_write(channel, msg);
		ops++;
		machine_sleep(0);
	}
}

static void benchmark_runner(void *arg)
{
	printf("benchmark started.\n");

	machine_channel_t *channel;
	channel = machine_channel_create(0);

	int r = machine_coroutine_create(benchmark_reader, channel);
	int w = machine_coroutine_create(benchmark_writer, channel);

	machine_sleep(1000);
	machine_stop_current();
	machine_cancel(r);
	machine_join(r);
	machine_join(w);

	machine_channel_free(channel);

	printf("done.\n");
	printf("channel operations %d in 1 sec.\n", ops);
}

int main(int argc, char *argv[])
{
	machinarium_init();
	int id = machine_create("benchmark_channel", benchmark_runner, NULL);
	machine_wait(id);
	machinarium_free();
	return 0;
}
