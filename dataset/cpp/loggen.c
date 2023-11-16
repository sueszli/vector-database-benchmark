/*
 * Copyright (c) 2007-2018 Balabit
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
 *
 * As an additional exemption you are allowed to compile & link against the
 * OpenSSL libraries as published by the OpenSSL project. See the file
 * COPYING for details.
 *
 */

#include "compat/compat.h"
#include "compat/glib.h"
#include "loggen_plugin.h"
#include "loggen_helper.h"
#include "file_reader.h"
#include "logline_generator.h"
#include "reloc.h"

#include <stdio.h>
#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>
#include <signal.h>
#include <string.h>
#include <gmodule.h>
#include <errno.h>

#ifdef __APPLE__
#ifdef G_MODULE_SUFFIX
#undef G_MODULE_SUFFIX
#endif
#define G_MODULE_SUFFIX "dylib"
#endif

static PluginOption global_plugin_option =
{
  .message_length = 256,
  .interval = 10,
  .number_of_messages = 0,
  .permanent = 0,
  .active_connections = 1,
  .idle_connections = 0,
  .use_ipv6 = 0,
  .target = NULL,
  .port = NULL,
  .rate = 1000,
  .reconnect = 0,
  .proxied = FALSE,
  .proxy_version = 1,
  .proxy_src_ip = NULL,
  .proxy_dst_ip = NULL,
  .proxy_src_port = NULL,
  .proxy_dst_port = NULL,
};

static char *sdata_value = NULL;
static int noframing = 0;
static int syslog_proto = 0;
static int quiet = 0;
static int csv = 0;
static int debug = 0;
static unsigned long sent_messages_num = 0;
static int read_from_file = 0;
static gint64 raw_message_length = 0;
static gint64 *thread_stat_count = NULL;
static gint64 *thread_stat_count_last = NULL;

static GMutex message_counter_lock;

static gboolean
_process_proxied_arg(const gchar *option_name,
                     const gchar *value,
                     gpointer data,
                     GError **error)
{
  global_plugin_option.proxied = TRUE;
  global_plugin_option.proxy_version = atoi(value ? : "1");
  return global_plugin_option.proxy_version == 1 || global_plugin_option.proxy_version == 2;
}

static GOptionEntry loggen_options[] =
{
  { "rate", 'r', 0, G_OPTION_ARG_INT, &global_plugin_option.rate, "Number of messages to generate per second", "<msg/sec/active connection>" },
  { "size", 's', 0, G_OPTION_ARG_INT, &global_plugin_option.message_length, "Specify the size of the syslog message", "<size>" },
  { "interval", 'I', 0, G_OPTION_ARG_INT, &global_plugin_option.interval, "Number of seconds to run the test for", "<sec>" },
  { "permanent", 'T', 0, G_OPTION_ARG_NONE, &global_plugin_option.permanent, "Send logs without time limit", NULL},
  { "syslog-proto", 'P', 0, G_OPTION_ARG_NONE, &syslog_proto, "Use the new syslog-protocol message format (see also framing)", NULL },
  { "proxied", 'H', G_OPTION_FLAG_OPTIONAL_ARG, G_OPTION_ARG_CALLBACK, _process_proxied_arg, "Generate PROXY protocol header", "<protocol version 1 or 2>" },
  { "proxy-src-ip", 0, 0, G_OPTION_ARG_STRING, &global_plugin_option.proxy_src_ip, "Source IP for the PROXY protocol header", "<ip address>" },
  { "proxy-dst-ip", 0, 0, G_OPTION_ARG_STRING, &global_plugin_option.proxy_dst_ip, "Destination IP for the PROXY protocol header", "<ip address>" },
  { "proxy-src-port", 0, 0, G_OPTION_ARG_STRING, &global_plugin_option.proxy_src_port, "Source port for the PROXY protocol header", "<port>" },
  { "proxy-dst-port", 0, 0, G_OPTION_ARG_STRING, &global_plugin_option.proxy_dst_port, "Destination port for the PROXY protocol header", "<port>" },
  { "sdata", 'p', 0, G_OPTION_ARG_STRING, &sdata_value, "Send the given sdata (e.g. \"[test name=\\\"value\\\"]\") in case of syslog-proto", NULL },
  { "no-framing", 'F', G_OPTION_ARG_NONE, G_OPTION_ARG_NONE, &noframing, "Don't use syslog-protocol style framing, even if syslog-proto is set", NULL },
  { "active-connections", 0, 0, G_OPTION_ARG_INT, &global_plugin_option.active_connections, "Number of active connections to the server (default = 1)", "<number>" },
  { "idle-connections", 0, 0, G_OPTION_ARG_INT, &global_plugin_option.idle_connections, "Number of inactive connections to the server (default = 0)", "<number>" },
  { "ipv6",    '6', 0, G_OPTION_ARG_NONE, &global_plugin_option.use_ipv6, "Use AF_INET6 sockets instead of AF_INET (can use both IPv4 & IPv6)", NULL },
  { "csv", 'C', 0, G_OPTION_ARG_NONE, &csv, "Produce CSV output", NULL },
  { "number", 'n', 0, G_OPTION_ARG_INT, &global_plugin_option.number_of_messages, "Number of messages to generate", "<number>" },
  { "quiet", 'Q', 0, G_OPTION_ARG_NONE, &quiet, "Don't print the msg/sec data", NULL },
  { "debug", 0, 0, G_OPTION_ARG_NONE, &debug, "Enable loggen debug messages", NULL },
  { "reconnect", 0, 0, G_OPTION_ARG_NONE, &global_plugin_option.reconnect, "Attempt to reconnect when destination connections are lost", NULL},
  { NULL }
};

/* This is the callback function called by plugins when
 * they need a new log line */
int
generate_message(char *buffer, int buffer_size, ThreadData *thread_context, unsigned long seq)
{
  int str_len;

  if (global_plugin_option.proxied && !thread_context->proxy_header_sent)
    {
      str_len = generate_proxy_header(buffer, buffer_size, thread_context->index,
                                      global_plugin_option.proxy_version,
                                      global_plugin_option.proxy_src_ip, global_plugin_option.proxy_dst_ip,
                                      global_plugin_option.proxy_src_port, global_plugin_option.proxy_dst_port);
      thread_context->proxy_header_sent = TRUE;
      DEBUG("Generated PROXY protocol v%d header; len=%d\n", global_plugin_option.proxy_version, str_len);
      return str_len;
    }

  if (read_from_file)
    str_len = read_next_message_from_file(buffer, buffer_size, syslog_proto, thread_context->index);
  else
    str_len = generate_log_line(buffer, buffer_size, syslog_proto, thread_context->index, seq);

  if (str_len < 0)
    return -1;

  g_mutex_lock(&message_counter_lock);
  sent_messages_num++;
  raw_message_length += str_len;

  if (thread_stat_count && csv)
    thread_stat_count[thread_context->index]+=1;

  g_mutex_unlock(&message_counter_lock);

  return str_len;
}

static
gboolean is_plugin_already_loaded(GPtrArray *plugin_array, const gchar *name)
{
  for (int i=0; i < plugin_array->len; i++)
    {
      PluginInfo *loaded_plugin = g_ptr_array_index(plugin_array, i);
      if (!loaded_plugin)
        continue;

      if (strcmp(name, loaded_plugin->name) == 0)
        {
          return TRUE;
        }
    }
  return FALSE;
}

/* receives a filename and returns a PluginInfo or NULL */
static PluginInfo *
load_plugin_info_with_fname(const gchar *plugin_path, const gchar *fname)
{
  if (strlen(fname) <= strlen(LOGGEN_PLUGIN_LIB_PREFIX) || !g_str_has_suffix(fname, G_MODULE_SUFFIX)
      || !g_str_has_prefix(fname, LOGGEN_PLUGIN_LIB_PREFIX))
    return NULL;

  gchar *full_lib_path = g_build_filename(plugin_path, fname, NULL);
  GModule *module = g_module_open(full_lib_path, G_MODULE_BIND_LAZY);
  g_free(full_lib_path);

  if (!module)
    {
      ERROR("error opening plugin module %s (%s)\n", fname, g_module_error());
      return NULL;
    }


  /* libloggen_name_of_the_plugin.{so,dll,dylib} */
  const gchar *fname_start = fname + strlen(LOGGEN_PLUGIN_LIB_PREFIX);
  const gchar *fname_end = strrchr(fname_start, '_');
  if (!fname_end)
    {
      ERROR("error opening plugin module %s, module filename does not fit the pattern\n", fname);
      return NULL;
    }
  const gint fname_len = fname_end - fname_start;

  gchar plugin_name[LOGGEN_PLUGIN_NAME_MAXSIZE + 1];
  g_snprintf(plugin_name, LOGGEN_PLUGIN_NAME_MAXSIZE, "%.*s_%s", fname_len, fname_start, LOGGEN_PLUGIN_INFO);

  /* get plugin info from lib file */
  PluginInfo *plugin;
  if (!g_module_symbol(module, plugin_name, (gpointer *) &plugin))
    {
      DEBUG("%s isn't a plugin for loggen. skip it. (%s)\n", fname, g_module_error());
      g_module_close(module);
      return NULL;
    }
  return plugin;
}

/* return value means the number of successfully loaded plugins */
static int
enumerate_plugins(const gchar *plugin_path, GPtrArray *plugin_array, GOptionContext *ctx)
{
  const gchar *fname;

  GDir *dir = g_dir_open(plugin_path, 0, NULL);
  if (!dir)
    {
      ERROR("unable to open plugin directory %s (err=%s)\n", plugin_path, strerror(errno));
      ERROR("hint: you can use the %s environmental variable to specify plugin path\n", "SYSLOGNG_PREFIX");
      return 0;
    }

  DEBUG("search for plugins in directory %s\n", plugin_path);

  /* add common options to help context: */
  g_option_context_add_main_entries(ctx, loggen_options, 0);

  while ((fname = g_dir_read_name(dir)))
    {
      PluginInfo *plugin = load_plugin_info_with_fname(plugin_path, fname);
      if(!plugin)
        continue;

      if (is_plugin_already_loaded(plugin_array, plugin->name))
        {
          DEBUG("plugin %s was already loaded. skip it\n", plugin->name);
          continue;
        }

      if (plugin->set_generate_message)
        plugin->set_generate_message(generate_message);
      else
        ERROR("plugin (%s) doesn't have set_generate_message function\n", plugin->name);

      g_ptr_array_add(plugin_array, (gpointer) plugin);

      /* create sub group for plugin specific parameters: */
      GOptionGroup *group = g_option_group_new(plugin->name, plugin->name, "Show options", NULL, NULL);
      g_option_group_add_entries(group, plugin->get_options_list());
      g_option_context_add_group(ctx, group);

      DEBUG("%s in %s is a loggen plugin\n", plugin->name, fname);
    }

  g_dir_close(dir);

  if (plugin_array->len == 0)
    {
      ERROR("no loggen plugin found in %s\n", plugin_path);
      ERROR("hint: you can use the %s environmental variable to specify plugin path\n", "SYSLOGNG_PREFIX");
    }

  DEBUG("%d plugin successfully loaded\n", plugin_array->len);
  return plugin_array->len;
}

static void
stop_plugins(GPtrArray *plugin_array)
{
  if (!plugin_array)
    return;

  for (int i=0; i < plugin_array->len; i++)
    {
      PluginInfo *plugin = g_ptr_array_index(plugin_array, i);
      if (!plugin)
        continue;

      DEBUG("stop plugin (%s:%d)\n", plugin->name, i);
      if (plugin->stop_plugin)
        plugin->stop_plugin((gpointer)&global_plugin_option);
    }

  DEBUG("all plugins have been stopped\n");
}

static void
init_logline_generator(GPtrArray *plugin_array)
{
  if (!plugin_array)
    return;

  gboolean require_framing = FALSE;
  for (int i=0; i < plugin_array->len; i++)
    {
      PluginInfo *plugin = g_ptr_array_index(plugin_array, i);
      if (!plugin)
        continue;

      /* check if any active plugin requires framing */
      if (plugin->require_framing && plugin->is_plugin_activated())
        {
          require_framing = TRUE;
          break;
        }
    }

  int framing;
  if (syslog_proto && !noframing)
    framing = 1;
  else if (!syslog_proto && require_framing && !noframing)
    framing = 1;
  else
    framing = 0;

  prepare_log_line_template(
    syslog_proto,
    framing,
    global_plugin_option.message_length,
    sdata_value);
}

static void
init_csv_statistics(void)
{
  /* message counter for csv output */
  thread_stat_count = (gint64 *) g_malloc0(global_plugin_option.active_connections * sizeof(gint64));
  thread_stat_count_last = (gint64 *) g_malloc0(global_plugin_option.active_connections * sizeof(gint64));
  if (csv)
    {
      /* print CSV header and initial line about time zero */
      printf("ThreadId;Time;Rate;Count\n");
      for (int j=0; j < global_plugin_option.active_connections; j++)
        {
          fprintf(stderr, "%d;%lu.%06lu;%.2lf;%lu\n", j, (long) 0, (long) 0, (double) 0, (long)0);
        }
    }
}

static int
start_plugins(GPtrArray *plugin_array)
{
  if (!plugin_array)
    {
      ERROR("invalid reference for plugin_array\n");
      return 0;
    }

  /* check plugins to see how many is activated by command line parameters */
  int number_of_active_plugins = 0;
  for (int i=0; i < plugin_array->len; i++)
    {
      PluginInfo *plugin = g_ptr_array_index(plugin_array, i);
      if (!plugin)
        continue;

      if (plugin->is_plugin_activated())
        number_of_active_plugins++;
    }

  if (number_of_active_plugins != 1)
    {
      ERROR("%d plugins activated. You should activate exactly one plugin at a time.\nDid you forget to add -S ?\nSee \"loggen --help-all\" for available plugin options\n",
            number_of_active_plugins);
      return 0;
    }

  for (int i=0; i < plugin_array->len; i++)
    {
      PluginInfo *plugin = g_ptr_array_index(plugin_array, i);
      if (!plugin)
        continue;

      if (plugin->start_plugin && plugin->is_plugin_activated())
        {
          if (!plugin->start_plugin((gpointer)&global_plugin_option))
            return 0;

          break;
        }
    }

  return number_of_active_plugins;
}

void
print_statistic(struct timeval *start_time)
{
  gint64 count;
  static gint64 last_count = 0;
  static struct timeval last_ts_format;

  struct timeval now;
  gettimeofday(&now, NULL);

  if (!quiet && !csv)
    {
      guint64 diff_usec = time_val_diff_in_usec(&now, &last_ts_format);
      if (diff_usec > 0)
        {
          g_mutex_lock(&message_counter_lock);
          count = sent_messages_num;
          g_mutex_unlock(&message_counter_lock);

          if (count > last_count && last_count > 0)
            {
              fprintf(stderr, "count=%"G_GINT64_FORMAT", rate = %.2lf msg/sec\n",
                      count,
                      ((double) (count - last_count) * USEC_PER_SEC) / diff_usec);
            }
          last_count = count;
        }
    }

  if (thread_stat_count && thread_stat_count_last && csv)
    {
      struct timeval diff_tv;
      time_val_diff_in_timeval(&diff_tv, &now, start_time);
      guint64 diff_usec = time_val_diff_in_usec(&now, &last_ts_format);

      for (int j=0; j < global_plugin_option.active_connections; j++)
        {
          g_mutex_lock(&message_counter_lock);
          double msg_count_diff = ((double) (thread_stat_count[j]-thread_stat_count_last[j]) * USEC_PER_SEC) / diff_usec;
          thread_stat_count_last[j] = thread_stat_count[j];
          count = thread_stat_count[j];
          g_mutex_unlock(&message_counter_lock);

          fprintf(stderr, "%d;%lu.%06lu;%.2lf;%"G_GINT64_FORMAT"\n",
                  j,
                  (long) diff_tv.tv_sec,
                  (long) diff_tv.tv_usec,
                  msg_count_diff,
                  count
                 );
        }
    }
  last_ts_format = now;
}

void wait_all_plugin_to_finish(GPtrArray *plugin_array)
{
  if (!plugin_array)
    return;

  struct timeval start_time;
  gettimeofday(&start_time, NULL);

  for (int i=0; i < plugin_array->len; i++)
    {
      PluginInfo *plugin = g_ptr_array_index(plugin_array, i);
      if (!plugin)
        continue;

      while (plugin->get_thread_count() > 0)
        {
          g_usleep(500*1000);
          print_statistic(&start_time);
        }
    }

  /* print final statistic: */
  print_statistic(&start_time);
  unsigned long count = sent_messages_num;
  struct timeval now;
  gettimeofday(&now, NULL);
  double total_runtime_sec = time_val_diff_in_sec(&now, &start_time);
  if (total_runtime_sec > 0 && count > 0)
    fprintf(stderr,
            "average rate = %.2lf msg/sec, count=%ld, time=%g, (average) msg size=%"G_GINT64_FORMAT", bandwidth=%.2f kB/sec\n",
            (double)count/total_runtime_sec,
            count,
            total_runtime_sec,
            (gint64)raw_message_length/count,
            (double)raw_message_length/(total_runtime_sec*1024) );
  else
    fprintf(stderr, "Total runtime = %g, count = %ld\n", total_runtime_sec, count);
}

static void
signal_callback_handler(int signum)
{
  ERROR("Send error Broken pipe, results may be skewed. %d\n", signum);
}

static void
rate_change_handler(int signum)
{
  switch(signum)
    {
    case SIGUSR1:
      global_plugin_option.rate *= 2;
      break;
    case SIGUSR2:
    {
      int proposed_new_rate = global_plugin_option.rate / 2;
      global_plugin_option.rate = proposed_new_rate > 0 ? proposed_new_rate: 1;
      break;
    }
    default:
      break;
    }
}

static void
setup_rate_change_signals(void)
{
  struct sigaction sa;
  sa.sa_handler = rate_change_handler;
  sigemptyset(&sa.sa_mask);
  sa.sa_flags = SA_RESTART;

  sigaction(SIGUSR1, &sa, NULL);
  sigaction(SIGUSR2, &sa, NULL);
}

int
main(int argc, char *argv[])
{

  GPtrArray *plugin_array = g_ptr_array_new();
  GOptionContext *ctx = g_option_context_new(" target port");

  signal(SIGPIPE, signal_callback_handler);
  setup_rate_change_signals();

  const gchar *plugin_path = get_installation_path_for(SYSLOG_NG_PATH_LOGGENPLUGINDIR);
  enumerate_plugins(plugin_path, plugin_array, ctx);
  reloc_deinit();

  /* create sub group for file reader functions */
  GOptionGroup *group = g_option_group_new("file-reader", "file-reader", "Show options", NULL, NULL);
  g_option_group_add_entries(group, get_file_reader_options());
  g_option_context_add_group(ctx, group);

  GError *error = NULL;
  if (!g_option_context_parse(ctx, &argc, &argv, &error))
    {
      ERROR("option parsing failed: %s\n", error->message);
      g_ptr_array_free(plugin_array, TRUE);
      if (error)
        g_error_free(error);
      return 1;
    }

  /* debug option defined by --debug command line option */
  set_debug_level(debug);

  if (argc>=3)
    {
      global_plugin_option.target = g_strdup(argv[1]);
      global_plugin_option.port = g_strdup(argv[2]);
    }
  else if (argc>=2)
    {
      global_plugin_option.target = g_strdup(argv[1]);
      global_plugin_option.port = NULL;
    }
  else
    {
      global_plugin_option.target = NULL;
      global_plugin_option.port = NULL;
      DEBUG("no port and address specified");
    }

  DEBUG("target=%s port=%s\n", global_plugin_option.target, global_plugin_option.port);

  if (global_plugin_option.message_length > MAX_MESSAGE_LENGTH)
    {
      ERROR("warning: defined message length (%d) is too big. truncated to (%d)\n", global_plugin_option.message_length,
            MAX_MESSAGE_LENGTH);
      global_plugin_option.message_length = MAX_MESSAGE_LENGTH;
    }

  read_from_file = init_file_reader(global_plugin_option.active_connections);
  if (read_from_file < 0)
    {
      ERROR("error while opening input file. exit.\n");
      return 1;
    }

  g_mutex_init(&message_counter_lock);

  init_logline_generator(plugin_array);
  init_csv_statistics();

  if (start_plugins(plugin_array) > 0)
    {
      wait_all_plugin_to_finish(plugin_array);
      stop_plugins(plugin_array);
    }

  close_file_reader(global_plugin_option.active_connections);

  g_mutex_clear(&message_counter_lock);
  g_free((gpointer)global_plugin_option.target);
  g_free((gpointer)global_plugin_option.port);
  g_option_context_free(ctx);
  g_ptr_array_free(plugin_array, TRUE);
  g_free(thread_stat_count_last);
  g_free(thread_stat_count);
  return 0;
}
