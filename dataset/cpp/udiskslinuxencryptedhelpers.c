/* -*- mode: C; c-file-style: "gnu"; indent-tabs-mode: nil; -*-
 *
 * Copyright (C) 2017 Red Hat, Inc.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
 *
 * Author: Vratislav Podzimek <vpodzime@redhat.com>
 *
 */

#include <glib.h>
#include <blockdev/crypto.h>

#include "udisksthreadedjob.h"
#include "udiskslinuxencryptedhelpers.h"

gboolean luks_format_job_func (UDisksThreadedJob  *job,
                      GCancellable       *cancellable,
                      gpointer            user_data,
                      GError            **error)
{
  BDCryptoLUKSVersion luks_version;
  CryptoJobData *data = (CryptoJobData*) user_data;
  BDCryptoKeyslotContext *context = NULL;
  gboolean ret = FALSE;

  if (g_strcmp0 (data->type, "luks1") == 0)
    luks_version = BD_CRYPTO_LUKS_VERSION_LUKS1;
  else if ((g_strcmp0 (data->type, "luks2") == 0))
    luks_version = BD_CRYPTO_LUKS_VERSION_LUKS2;
  else
    {
      g_set_error (error,
                   UDISKS_ERROR,
                   UDISKS_ERROR_FAILED,
                   "Unknown or unsupported encryption type specified: '%s'",
                   data->type);
      return FALSE;
    }

  context = bd_crypto_keyslot_context_new_passphrase ((const guint8 *) data->passphrase->str,
                                                      data->passphrase->len, error);
  if (!context)
    return FALSE;

  /* device, cipher, key_size, context, min_entropy, luks_version, extra, error */
  ret = bd_crypto_luks_format (data->device, NULL, 0, context, 0, luks_version, NULL, error);
  bd_crypto_keyslot_context_free (context);
  return ret;
}

gboolean luks_open_job_func (UDisksThreadedJob  *job,
                    GCancellable       *cancellable,
                    gpointer            user_data,
                    GError            **error)
{
  CryptoJobData *data = (CryptoJobData*) user_data;
  BDCryptoKeyslotContext *context = NULL;
  gboolean ret = FALSE;

  context = bd_crypto_keyslot_context_new_passphrase ((const guint8 *) data->passphrase->str,
                                                      data->passphrase->len, error);
  if (!context)
    return FALSE;

  /* device, name, context, read_only, error */
  ret = bd_crypto_luks_open (data->device, data->map_name, context, data->read_only, error);
  bd_crypto_keyslot_context_free (context);
  return ret;
}

gboolean luks_close_job_func (UDisksThreadedJob  *job,
                    GCancellable       *cancellable,
                    gpointer            user_data,
                    GError            **error)
{
  CryptoJobData *data = (CryptoJobData*) user_data;
  return bd_crypto_luks_close (data->map_name, error);
}

gboolean luks_change_key_job_func (UDisksThreadedJob  *job,
                          GCancellable       *cancellable,
                          gpointer            user_data,
                          GError            **error)
{
  CryptoJobData *data = (CryptoJobData*) user_data;
  BDCryptoKeyslotContext *context = NULL;
  BDCryptoKeyslotContext *ncontext = NULL;
  gboolean ret = FALSE;

  context = bd_crypto_keyslot_context_new_passphrase ((const guint8 *) data->passphrase->str,
                                                      data->passphrase->len, error);
  if (!context)
    return FALSE;
  ncontext = bd_crypto_keyslot_context_new_passphrase ((const guint8 *) data->new_passphrase->str,
                                                       data->new_passphrase->len, error);
  if (!context)
    {
      bd_crypto_keyslot_context_free (context);
      return FALSE;
    }

  ret = bd_crypto_luks_change_key (data->device, context, ncontext, error);
  bd_crypto_keyslot_context_free (context);
  bd_crypto_keyslot_context_free (ncontext);
  return ret;
}

gboolean tcrypt_open_job_func (UDisksThreadedJob  *job,
                               GCancellable       *cancellable,
                               gpointer            user_data,
                               GError            **error)
{
  CryptoJobData *data = (CryptoJobData*) user_data;
  BDCryptoKeyslotContext *context = NULL;
  gboolean ret = FALSE;

  /* We always use the veracrypt option, because it can unlock both VeraCrypt and legacy TrueCrypt volumes */
  gboolean  veracrypt = TRUE;

  /* passphrase can be empty for veracrypt with keyfiles */
  if (data->passphrase->len > 0)
    {
      context = bd_crypto_keyslot_context_new_passphrase ((const guint8 *) data->passphrase->str,
                                                          data->passphrase->len, error);
      if (!context)
        return FALSE;
    }

  ret = bd_crypto_tc_open (data->device, data->map_name, context,
                           data->keyfiles, data->hidden, data->system, veracrypt, data->pim,
                           data->read_only, error);
  bd_crypto_keyslot_context_free (context);
  return ret;
}

gboolean tcrypt_close_job_func (UDisksThreadedJob  *job,
                                GCancellable       *cancellable,
                                gpointer            user_data,
                                GError            **error)
{
  CryptoJobData *data = (CryptoJobData*) user_data;
  return bd_crypto_tc_close (data->map_name, error);
}

gboolean bitlk_open_job_func (UDisksThreadedJob  *job,
                              GCancellable       *cancellable,
                              gpointer            user_data,
                              GError            **error)
{
  CryptoJobData *data = (CryptoJobData*) user_data;
  BDCryptoKeyslotContext *context = NULL;
  gboolean ret = FALSE;

  context = bd_crypto_keyslot_context_new_passphrase ((const guint8 *) data->passphrase->str,
                                                      data->passphrase->len, error);
  if (!context)
    return FALSE;

  ret = bd_crypto_bitlk_open (data->device, data->map_name, context, data->read_only, error);
  bd_crypto_keyslot_context_free (context);
  return ret;
}

gboolean bitlk_close_job_func (UDisksThreadedJob  *job,
                               GCancellable       *cancellable,
                               gpointer            user_data,
                               GError            **error)
{
  CryptoJobData *data = (CryptoJobData*) user_data;
  return bd_crypto_bitlk_close (data->map_name, error);
}
