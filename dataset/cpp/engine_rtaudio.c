// SPDX-FileCopyrightText: © 2020-2023 Alexandros Theodotou <alex@zrythm.org>
// SPDX-License-Identifier: LicenseRef-ZrythmLicense

#include "zrythm-config.h"

#ifdef HAVE_RTAUDIO

#  include "dsp/engine.h"
#  include "dsp/engine_rtaudio.h"
#  include "project.h"
#  include "settings/settings.h"
#  include "utils/dsp.h"
#  include "utils/string.h"
#  include "zrythm.h"

#  include <gtk/gtk.h>

#  include <rtaudio_c.h>

static rtaudio_api_t
get_api_from_audio_backend (AudioBackend backend)
{
  const rtaudio_api_t * apis = rtaudio_compiled_api ();
  unsigned int          num_apis = rtaudio_get_num_compiled_apis ();
  for (unsigned int i = 0; i < num_apis; i++)
    {
      if (
        backend == AUDIO_BACKEND_ALSA_RTAUDIO
        && apis[i] == RTAUDIO_API_LINUX_ALSA)
        {
          return apis[i];
        }
      if (
        backend == AUDIO_BACKEND_JACK_RTAUDIO
        && apis[i] == RTAUDIO_API_UNIX_JACK)
        {
          return apis[i];
        }
      if (
        backend == AUDIO_BACKEND_PULSEAUDIO_RTAUDIO
        && apis[i] == RTAUDIO_API_LINUX_PULSE)
        {
          return apis[i];
        }
      if (
        backend == AUDIO_BACKEND_COREAUDIO_RTAUDIO
        && apis[i] == RTAUDIO_API_MACOSX_CORE)
        {
          return apis[i];
        }
      if (
        backend == AUDIO_BACKEND_WASAPI_RTAUDIO
        && apis[i] == RTAUDIO_API_WINDOWS_WASAPI)
        {
          return apis[i];
        }
      if (
        backend == AUDIO_BACKEND_ASIO_RTAUDIO
        && apis[i] == RTAUDIO_API_WINDOWS_ASIO)
        {
          return apis[i];
        }
    }

  return RTAUDIO_API_DUMMY;
}

static int
audio_cb (
  float *                 out_buf,
  float *                 in_buf,
  unsigned int            nframes,
  double                  stream_time,
  rtaudio_stream_status_t status,
  AudioEngine *           self)
{
  if (status != 0)
    {
      /* xrun */
      //g_warning ("XRUN in RtAudio");
    }

  if (!engine_get_run (self))
    return 0;

  nframes_t num_frames = (nframes_t) nframes;
  engine_process (self, num_frames);

  dsp_fill (out_buf, DENORMAL_PREVENTION_VAL, (size_t) (nframes * 2));
  for (nframes_t i = 0; i < num_frames; i++)
    {
      out_buf[i * 2] = self->monitor_out->l->buf[i];
      out_buf[i * 2 + 1] = self->monitor_out->r->buf[i];
    }

  return 0;
}

static void
error_cb (rtaudio_error_t err, const char * msg)
{
  g_critical ("RtAudio error: %s", msg);
}

static bool engine_rtaudio_first_run = true;

rtaudio_t
engine_rtaudio_create_rtaudio (AudioEngine * self, AudioBackend backend)
{
  rtaudio_t rtaudio;

  if (engine_rtaudio_first_run)
    {
      /* print compiled APIs */
      const rtaudio_api_t * apis = rtaudio_compiled_api ();
      unsigned int          num_apis = rtaudio_get_num_compiled_apis ();
      for (unsigned int i = 0; i < num_apis; i++)
        {
          g_message ("RtAudio API found: %s", rtaudio_api_name (apis[i]));
        }
      engine_rtaudio_first_run = false;
    }

  rtaudio_api_t api = get_api_from_audio_backend (backend);
  if (api == RTAUDIO_API_DUMMY)
    {
      g_warning ("RtAudio API for %s not enabled", audio_backend_str[backend]);
      return NULL;
    }

  g_debug ("calling rtaudio_create...");
  rtaudio = rtaudio_create (api);

  if (rtaudio_error (rtaudio))
    {
      g_critical ("RtAudio: %s", rtaudio_error (rtaudio));
      return NULL;
    }

  g_debug ("rtaudio_create() successful");

  return rtaudio;
}

static void
print_dev_info (rtaudio_device_info_t * nfo, char * buf)
{
  sprintf (
    buf,
    "name: %s\n"
#  if RTAUDIO_MAJOR_VER == 5
    "probed: %d\n"
#  endif
    "output channels: %u\n"
    "input channels: %u\n"
    "duplex channels: %u\n"
    "is default output: %d\n"
    "is default input: %d\n"
    "formats: TODO\n"
    "preferred sample rate: %u\n"
    "sample rates: TODO",
    nfo->name,
#  if RTAUDIO_MAJOR_VER == 5
    nfo->probed,
#  endif
    nfo->output_channels, nfo->input_channels, nfo->duplex_channels,
    nfo->is_default_output, nfo->is_default_input, nfo->preferred_sample_rate);
}

/**
 * Set up the engine.
 */
int
engine_rtaudio_setup (AudioEngine * self)
{
  g_message ("Setting up RtAudio %s...", rtaudio_version ());

  self->rtaudio = engine_rtaudio_create_rtaudio (self, self->audio_backend);
  if (!self->rtaudio)
    {
      return -1;
    }

  int dev_count = rtaudio_device_count (self->rtaudio);
  g_return_val_if_fail (dev_count >= 0, -1);
  if (dev_count == 0)
    {
      g_warning ("No devices found");
      return -1;
    }

  /* check if selected device is found in the list
   * of devices and get the id of the output device
   * to open */
  char * out_device =
    g_settings_get_string (S_P_GENERAL_ENGINE, "rtaudio-audio-device-name");
  unsigned int out_device_id = UINT_MAX;
  for (unsigned int i = 0; i < (unsigned int) dev_count; i++)
    {
      rtaudio_device_info_t dev_nfo = rtaudio_get_device_info (self->rtaudio, i);
      char dev_nfo_str[800];
      print_dev_info (&dev_nfo, dev_nfo_str);
      g_message ("RtAudio device %d: %s", i, dev_nfo_str);
      if (
        string_is_equal (dev_nfo.name, out_device)
        && dev_nfo.output_channels > 0)
        {
          g_message ("found device at index %d", i);
          out_device_id = i;
        }
    }
  if (out_device_id == UINT_MAX)
    {
      g_message ("selected device not found, using default");
      out_device_id = rtaudio_get_default_output_device (self->rtaudio);
      rtaudio_device_info_t dev_nfo =
        rtaudio_get_device_info (self->rtaudio, out_device_id);
      out_device = g_strdup (dev_nfo.name);
    }

  /* prepare params */
  struct rtaudio_stream_parameters out_stream_params = {
    .device_id = out_device_id,
    .num_channels = 2,
    .first_channel = 0,
  };
  struct rtaudio_stream_options stream_opts = {
    .flags = RTAUDIO_FLAGS_SCHEDULE_REALTIME,
    .num_buffers = 2,
    .priority = 99,
    .name = "Zrythm",
  };

  unsigned int samplerate = (unsigned int) engine_samplerate_enum_to_int (
    (AudioEngineSamplerate) g_settings_get_enum (
      S_P_GENERAL_ENGINE, "sample-rate"));
  unsigned int buffer_size = (unsigned int) engine_buffer_size_enum_to_int (
    (AudioEngineBufferSize) g_settings_get_enum (
      S_P_GENERAL_ENGINE, "buffer-size"));
  g_message (
    "Attempting to open device [%s] with sample "
    "rate %u and buffer size %d",
    out_device, samplerate, buffer_size);

  int ret = rtaudio_open_stream (
    self->rtaudio, &out_stream_params, NULL, RTAUDIO_FORMAT_FLOAT32, samplerate,
    &buffer_size, (rtaudio_cb_t) audio_cb, self, &stream_opts,
    (rtaudio_error_cb_t) error_cb);
  if (ret)
    {
      g_warning (
        "An error occurred opening the RtAudio "
        "stream: %s",
        rtaudio_error (self->rtaudio));
      return -1;
    }
  bool is_open = rtaudio_is_stream_open (self->rtaudio);
  if (!is_open)
    {
      g_warning ("RtAudio stream failed to open");
      return -1;
    }
  self->block_length = buffer_size;
  self->sample_rate = (sample_rate_t) samplerate;

  g_message ("RtAudio set up");

  return 0;
}

void
engine_rtaudio_activate (AudioEngine * self, bool activate)
{
  if (activate)
    {
      g_message ("%s: activating...", __func__);
      rtaudio_start_stream (self->rtaudio);
    }
  else
    {
      g_message ("%s: deactivating...", __func__);
      rtaudio_stop_stream (self->rtaudio);
    }

  g_message ("%s: done", __func__);
}

/**
 * Tests if the backend is working properly.
 *
 * Returns 0 if ok, non-null if has errors.
 *
 * If win is not null, it displays error messages
 * to it.
 */
int
engine_rtaudio_test (GtkWindow * win)
{
  return 0;
}

/**
 * Returns a list of names inside \ref names that
 * must be free'd.
 *
 * @param input 1 for input, 0 for output.
 */
void
engine_rtaudio_get_device_names (
  AudioEngine * self,
  AudioBackend  backend,
  int           input,
  char **       names,
  int *         num_names)
{
  rtaudio_t rtaudio = engine_rtaudio_create_rtaudio (self, backend);
  if (!rtaudio)
    {
      g_warning ("failed to create rtaudio instance");
      return;
    }
  int num_devs = rtaudio_device_count (rtaudio);
  g_return_if_fail (num_devs >= 0);
  *num_names = 0;
  for (unsigned int i = 0; i < (unsigned int) num_devs; i++)
    {
      rtaudio_device_info_t dev_nfo = rtaudio_get_device_info (rtaudio, i);
      if (input && (dev_nfo.input_channels > 0 || dev_nfo.duplex_channels > 0))
        {
          names[(*num_names)++] = g_strdup (dev_nfo.name);
        }
      else if (
        !input && (dev_nfo.output_channels > 0 || dev_nfo.duplex_channels > 0))
        {
          names[(*num_names)++] = g_strdup (dev_nfo.name);
        }
      else
        {
          continue;
        }
      g_message ("RtAudio device %d: %s", i, names[*num_names - 1]);
    }
  rtaudio_destroy (rtaudio);
}

/**
 * Closes the engine.
 */
void
engine_rtaudio_tear_down (AudioEngine * self)
{
  rtaudio_close_stream (self->rtaudio);
}

#endif // HAVE_RTAUDIO
