/* -*- mode: C; c-file-style: "gnu"; indent-tabs-mode: nil; -*-
 *
 * Copyright (C) 2015 Peter Hatina <phatina@redhat.com>
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
 */

#include "config.h"

#include <fcntl.h>
#include <errno.h>
#include <string.h>

#include <glib/gi18n-lib.h>

#include <libiscsi.h>

#include <src/udisksdaemon.h>
#include <src/udisksdaemonutil.h>
#include <src/udiskslogging.h>
#include <src/udisksmodulemanager.h>

#include "udisksiscsiutil.h"
#include "udiskslinuxmanageriscsiinitiator.h"

/**
 * SECTION:udiskslinuxmanageriscsiinitiator
 * @title: UDisksLinuxManagerISCSIInitiator
 * @short_description: Linux implementation of
 * #UDisksLinuxManagerISCSIInitiator
 *
 * This type provides an implementation of the
 * #UDisksLinuxManagerISCSIInitiator interface on Linux.
 */

/**
 * UDisksLinuxManagerISCSIInitiator:
 *
 * The #UDisksLinuxManagerISCSIInitiator structure contains only private data
 * and should only be accessed using the provided API.
 */
struct _UDisksLinuxManagerISCSIInitiator {
  UDisksManagerISCSIInitiatorSkeleton parent_instance;

  UDisksLinuxModuleISCSI *module;
  GMutex initiator_config_mutex;  /* We use separate mutex for configuration
                                     file because libiscsi doesn't provide us
                                     any API for this. */
};

struct _UDisksLinuxManagerISCSIInitiatorClass {
  UDisksManagerISCSIInitiatorSkeletonClass parent_class;
};

enum
{
  PROP_0,
  PROP_MODULE
};

static void udisks_linux_manager_iscsi_initiator_iface_init (UDisksManagerISCSIInitiatorIface *iface);

G_DEFINE_TYPE_WITH_CODE (UDisksLinuxManagerISCSIInitiator, udisks_linux_manager_iscsi_initiator,
                         UDISKS_TYPE_MANAGER_ISCSI_INITIATOR_SKELETON,
                         G_IMPLEMENT_INTERFACE (UDISKS_TYPE_MANAGER_ISCSI_INITIATOR,
                                                udisks_linux_manager_iscsi_initiator_iface_init));

#define INITIATOR_FILENAME "/etc/iscsi/initiatorname.iscsi"
#define INITIATOR_NAME_KEY "InitiatorName"

/* ---------------------------------------------------------------------------------------------------- */

static void
udisks_linux_manager_iscsi_initiator_get_property (GObject *object, guint property_id,
                                                   GValue *value, GParamSpec *pspec)
{
  UDisksLinuxManagerISCSIInitiator *manager = UDISKS_LINUX_MANAGER_ISCSI_INITIATOR (object);

  switch (property_id)
    {
    case PROP_MODULE:
      g_value_set_object (value, udisks_linux_manager_iscsi_initiator_get_module (manager));
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, property_id, pspec);
      break;
    }
}

static void
udisks_linux_manager_iscsi_initiator_set_property (GObject *object, guint property_id,
                                                   const GValue *value, GParamSpec *pspec)
{
  UDisksLinuxManagerISCSIInitiator *manager = UDISKS_LINUX_MANAGER_ISCSI_INITIATOR (object);

  switch (property_id)
    {
    case PROP_MODULE:
      g_assert (manager->module == NULL);
      manager->module = g_value_dup_object (value);
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, property_id, pspec);
      break;
    }
}

static void
udisks_linux_manager_iscsi_initiator_finalize (GObject *object)
{
  if (G_OBJECT_CLASS (udisks_linux_manager_iscsi_initiator_parent_class))
    G_OBJECT_CLASS (udisks_linux_manager_iscsi_initiator_parent_class)->finalize (object);
}

static void
udisks_linux_manager_iscsi_initiator_class_init (UDisksLinuxManagerISCSIInitiatorClass *klass)
{
  GObjectClass *gobject_class = G_OBJECT_CLASS (klass);

  gobject_class->get_property = udisks_linux_manager_iscsi_initiator_get_property;
  gobject_class->set_property = udisks_linux_manager_iscsi_initiator_set_property;
  gobject_class->finalize = udisks_linux_manager_iscsi_initiator_finalize;

  /**
   * UDisksLinuxManager:module:
   *
   * The #UDisksLinuxModuleISCSI for the object.
   */
  g_object_class_install_property (gobject_class,
                                   PROP_MODULE,
                                   g_param_spec_object ("module",
                                                        "Module",
                                                        "The module for the object",
                                                        UDISKS_TYPE_LINUX_MODULE_ISCSI,
                                                        G_PARAM_READABLE |
                                                        G_PARAM_WRITABLE |
                                                        G_PARAM_CONSTRUCT_ONLY |
                                                        G_PARAM_STATIC_STRINGS));
}

static void
udisks_linux_manager_iscsi_initiator_init (UDisksLinuxManagerISCSIInitiator *manager)
{
  g_dbus_interface_skeleton_set_flags (G_DBUS_INTERFACE_SKELETON (manager),
                                       G_DBUS_INTERFACE_SKELETON_FLAGS_HANDLE_METHOD_INVOCATIONS_IN_THREAD);
#ifdef HAVE_LIBISCSI_GET_SESSION_INFOS
  udisks_manager_iscsi_initiator_set_sessions_supported (UDISKS_MANAGER_ISCSI_INITIATOR (manager),
                                                         TRUE);
#endif
}

/**
 * udisks_linux_manager_iscsi_initiator_new:
 * @module: A #UDisksLinuxModuleISCSI.
 *
 * Creates a new #UDisksLinuxManagerISCSIInitiator instance.
 *
 * Returns: A new #UDisksLinuxManagerISCSIInitiator. Free with g_object_unref().
 */
UDisksLinuxManagerISCSIInitiator *
udisks_linux_manager_iscsi_initiator_new (UDisksLinuxModuleISCSI *module)
{
  g_return_val_if_fail (UDISKS_IS_LINUX_MODULE_ISCSI (module), NULL);
  return UDISKS_LINUX_MANAGER_ISCSI_INITIATOR (g_object_new (UDISKS_TYPE_LINUX_MANAGER_ISCSI_INITIATOR,
                                                             "module", module,
                                                             NULL));
}

/**
 * udisks_linux_manager_iscsi_initiator_get_module:
 * @manager: A #UDisksLinuxManagerISCSIInitiator.
 *
 * Gets the module used by @manager.
 *
 * Returns: A #UDisksLinuxModuleISCSI. Do not free, the object is owned by @manager.
 */
UDisksLinuxModuleISCSI *
udisks_linux_manager_iscsi_initiator_get_module (UDisksLinuxManagerISCSIInitiator *manager)
{
  g_return_val_if_fail (UDISKS_IS_LINUX_MANAGER_ISCSI_INITIATOR (manager), NULL);
  return manager->module;
}

/* ---------------------------------------------------------------------------------------------------- */

static gboolean
handle_get_firmware_initiator_name (UDisksManagerISCSIInitiator *object,
                                    GDBusMethodInvocation       *invocation)
{
  gchar initiator_name[LIBISCSI_VALUE_MAXLEN];
  gint rval;

  rval = libiscsi_get_firmware_initiator_name (initiator_name);
  if (rval == 0)
    {
      udisks_manager_iscsi_initiator_complete_get_firmware_initiator_name (object,
                                                                           invocation,
                                                                           initiator_name);
    }
  else
    {
      g_dbus_method_invocation_return_error (invocation,
                                             UDISKS_ERROR,
                                             UDISKS_ERROR_ISCSI_NO_FIRMWARE,
                                             "No firmware found");
    }

  /* Indicate that we handled the method invocation */
  return TRUE;
}


#define FAKE_GROUP_NAME "general"       /* NOTE: shows up in some error messages */

static gchar *
_get_initiator_name (GError **error)
{
  gchar *initiator_name;
  gchar *contents = NULL;
  gchar *contents_group;
  GKeyFile *key_file;

  if (! g_file_get_contents (INITIATOR_FILENAME, &contents, NULL, error))
    {
      g_prefix_error (error, "Error reading iSCSI initiator name from '%s': ", INITIATOR_FILENAME);
      return NULL;
    }

  /* key-value pairs have to belong to some group, let's create a fake one just to satisfy the parser needs */
  contents_group = g_strconcat ("[" FAKE_GROUP_NAME "]\n", contents, NULL);
  g_free (contents);

  key_file = g_key_file_new ();
  if (! g_key_file_load_from_data (key_file, contents_group, -1, G_KEY_FILE_NONE, error))
    {
      g_prefix_error (error, "Error reading iSCSI initiator name from '%s': ", INITIATOR_FILENAME);
      g_key_file_free (key_file);
      g_free (contents_group);
      return NULL;
    }

  initiator_name = g_key_file_get_string (key_file, FAKE_GROUP_NAME, INITIATOR_NAME_KEY, error);
  if (initiator_name == NULL)
    g_prefix_error (error, "Error reading iSCSI initiator name from '%s': ", INITIATOR_FILENAME);

  g_key_file_free (key_file);
  g_free (contents_group);

  /* trim the whitespace at the end of the string */
  if (initiator_name != NULL)
    initiator_name = g_strchomp (initiator_name);

  return initiator_name;
}

static gboolean
handle_get_initiator_name (UDisksManagerISCSIInitiator *object,
                           GDBusMethodInvocation       *invocation)
{
  UDisksLinuxManagerISCSIInitiator *manager = UDISKS_LINUX_MANAGER_ISCSI_INITIATOR (object);
  gchar *initiator_name = NULL;
  GError *error = NULL;

  /* Enter a critical section. */
  g_mutex_lock (&manager->initiator_config_mutex);

  initiator_name = _get_initiator_name(&error);
  if (!initiator_name)
    {
      g_dbus_method_invocation_take_error (invocation, error);
      goto out;
    }

  /* Return the initiator name */
  udisks_manager_iscsi_initiator_complete_get_initiator_name (object,
                                                              invocation,
                                                              initiator_name);

out:
  /* Leave the critical section. */
  g_mutex_unlock (&manager->initiator_config_mutex);

  /* Release the resources */
  g_free (initiator_name);

  /* Indicate that we handled the method invocation */
  return TRUE;
}

static gboolean
handle_get_initiator_name_raw (UDisksManagerISCSIInitiator *object,
                               GDBusMethodInvocation       *invocation)
{
  UDisksLinuxManagerISCSIInitiator *manager = UDISKS_LINUX_MANAGER_ISCSI_INITIATOR (object);
  gchar *initiator_name = NULL;
  GError *error = NULL;

  /* Enter a critical section. */
  g_mutex_lock (&manager->initiator_config_mutex);

  initiator_name = _get_initiator_name(&error);
  if (!initiator_name)
    {
      g_dbus_method_invocation_take_error (invocation, error);
      goto out;
    }

  /* Return the initiator name */
  udisks_manager_iscsi_initiator_complete_get_initiator_name_raw (object,
                                                                  invocation,
                                                                  initiator_name);

out:
  /* Leave the critical section. */
  g_mutex_unlock (&manager->initiator_config_mutex);

  /* Release the resources */
  g_free (initiator_name);

  /* Indicate that we handled the method invocation */
  return TRUE;
}

static gboolean
handle_set_initiator_name (UDisksManagerISCSIInitiator *object,
                           GDBusMethodInvocation       *invocation,
                           const gchar                 *arg_name,
                           GVariant                    *arg_options)
{
  UDisksLinuxManagerISCSIInitiator *manager = UDISKS_LINUX_MANAGER_ISCSI_INITIATOR (object);
  UDisksDaemon *daemon;
  gchar *contents = NULL;
  gchar *contents_group;
  gchar *initiator_name;
  GKeyFile *key_file;
  GError *error = NULL;

  daemon = udisks_module_get_daemon (UDISKS_MODULE (manager->module));

  /* Policy check. */
  if (! udisks_daemon_util_check_authorization_sync (daemon,
                                                     NULL,
                                                     ISCSI_MODULE_POLICY_ACTION_ID,
                                                     arg_options,
                                                     N_("Authentication is required change iSCSI initiator name"),
                                                     invocation))
    return TRUE;

  if (!arg_name || strlen (arg_name) == 0)
    {
      g_dbus_method_invocation_return_error_literal (invocation,
                                                     UDISKS_ERROR,
                                                     UDISKS_ERROR_FAILED,
                                                     N_("Empty initiator name"));
      return TRUE;
    }

  /* enter critical section */
  g_mutex_lock (&manager->initiator_config_mutex);

  /* first try to read existing file */
  g_file_get_contents (INITIATOR_FILENAME, &contents, NULL, NULL /* ignore errors */);

  /* key-value pairs have to belong to some group, let's create a fake one just to satisfy the parser needs */
  contents_group = g_strconcat ("[" FAKE_GROUP_NAME "]\n", contents, NULL);
  g_free (contents);

  key_file = g_key_file_new ();
  if (! g_key_file_load_from_data (key_file, contents_group, -1,
                                   G_KEY_FILE_KEEP_COMMENTS | G_KEY_FILE_KEEP_TRANSLATIONS,
                                   NULL /* ignore errors */))
    {
      /* ignoring errors, leaving the keyfile instance empty (e.g. empty or non-existing config file) */
    }
  g_free (contents_group);

  /* ensure trailing space in the initiator name */
  if (arg_name[strlen (arg_name) - 1] != ' ')
    initiator_name = g_strconcat (arg_name, " ", NULL);
  else
    initiator_name = g_strdup (arg_name);
  g_key_file_set_string (key_file, FAKE_GROUP_NAME, INITIATOR_NAME_KEY, initiator_name);
  g_free (initiator_name);

  contents_group = g_key_file_to_data (key_file, NULL, NULL);
  /* strip the fake group name */
  contents = contents_group ? g_strrstr (contents_group, "[" FAKE_GROUP_NAME "]") : NULL;
  if (contents != NULL)
    contents += strlen ("[" FAKE_GROUP_NAME "]\n");

  if (contents == NULL)
    {
      g_dbus_method_invocation_return_error_literal (invocation,
                                                     UDISKS_ERROR,
                                                     UDISKS_ERROR_FAILED,
                                                     N_("Error parsing the iSCSI initiator name"));
    }
  else if (! g_file_set_contents (INITIATOR_FILENAME, contents, -1, &error))
    {
      g_prefix_error (&error, N_("Error writing to %s while setting iSCSI initiator name: "), INITIATOR_FILENAME);
      g_dbus_method_invocation_take_error (invocation, error);
    }
  else
    {
      /* finish with no error */
      udisks_manager_iscsi_initiator_complete_set_initiator_name (object, invocation);
    }
  g_free (contents_group);
  g_key_file_free (key_file);

  /* leave critical section */
  g_mutex_unlock (&manager->initiator_config_mutex);

  /* indicate that we handled the method invocation */
  return TRUE;
}

/**
 * discover_firmware:
 * @object: A #UDisksManagerISCSIInitiator
 * @nodes: A #GVariant containing an array with discovery results
 * @nodes_cnt: The count of discovered nodes
 * @errorstr: An error string pointer; may be NULL. Free with g_free().
 *
 * Performs firmware discovery (ppc or ibft).
 */
static gint
discover_firmware (UDisksManagerISCSIInitiator  *object,
                   GVariant                    **nodes,
                   gint                         *nodes_cnt,
                   gchar                       **errorstr)
{
  UDisksLinuxManagerISCSIInitiator *manager = UDISKS_LINUX_MANAGER_ISCSI_INITIATOR (object);
  struct libiscsi_context *ctx;
  struct libiscsi_node *found_nodes;
  gint rval;

  /* Enter a critical section. */
  udisks_linux_module_iscsi_lock_libiscsi_context (manager->module);

  /* Discovery */
  ctx = udisks_linux_module_iscsi_get_libiscsi_context (manager->module);
  rval = libiscsi_discover_firmware (ctx, nodes_cnt, &found_nodes);

  if (rval == 0)
    *nodes = iscsi_libiscsi_nodes_to_gvariant (found_nodes, *nodes_cnt);
  else if (errorstr)
    *errorstr = g_strdup (libiscsi_get_error_string (ctx));

  /* Leave the critical section. */
  udisks_linux_module_iscsi_unlock_libiscsi_context (manager->module);

  /* Release the resources */
  iscsi_libiscsi_nodes_free (found_nodes);

  return rval;
}

static gboolean
handle_discover_send_targets (UDisksManagerISCSIInitiator *object,
                              GDBusMethodInvocation       *invocation,
                              const gchar                 *arg_address,
                              const guint16                arg_port,
                              GVariant                    *arg_options)
{
  UDisksLinuxManagerISCSIInitiator *manager = UDISKS_LINUX_MANAGER_ISCSI_INITIATOR (object);
  UDisksDaemon *daemon;
  GVariant *nodes = NULL;
  gchar *errorstr = NULL;
  gint err = 0;
  gint nodes_cnt = 0;

  daemon = udisks_module_get_daemon (UDISKS_MODULE (manager->module));

  /* Policy check. */
  UDISKS_DAEMON_CHECK_AUTHORIZATION (daemon,
                                     NULL,
                                     ISCSI_MODULE_POLICY_ACTION_ID,
                                     arg_options,
                                     N_("Authentication is required to discover targets"),
                                     invocation);

  /* Enter a critical section. */
  udisks_linux_module_iscsi_lock_libiscsi_context (manager->module);

  /* Perform the discovery. */
  err = iscsi_discover_send_targets (manager->module, arg_address, arg_port, arg_options, &nodes, &nodes_cnt, &errorstr);

  /* Leave the critical section. */
  udisks_linux_module_iscsi_unlock_libiscsi_context (manager->module);

  if (err != 0)
    {
      /* Discovery failed. */
      g_dbus_method_invocation_return_error (invocation,
                                             UDISKS_ERROR,
                                             iscsi_error_to_udisks_error (err),
                                             N_("Discovery failed: %s"),
                                             errorstr);
      goto out;
    }

  /* Return discovered portals. */
  udisks_manager_iscsi_initiator_complete_discover_send_targets (object, invocation, nodes, nodes_cnt);

out:
  g_free (errorstr);

  /* Indicate that we handled the method invocation. */
  return TRUE;
}

static gboolean
handle_discover_firmware (UDisksManagerISCSIInitiator *object,
                          GDBusMethodInvocation       *invocation,
                          GVariant                    *arg_options)
{
  UDisksLinuxManagerISCSIInitiator *manager = UDISKS_LINUX_MANAGER_ISCSI_INITIATOR (object);
  UDisksDaemon *daemon;
  GVariant *nodes = NULL;
  gint err = 0;
  gint nodes_cnt = 0;
  gchar *errorstr = NULL;

  daemon = udisks_module_get_daemon (UDISKS_MODULE (manager->module));

  /* Policy check. */
  UDISKS_DAEMON_CHECK_AUTHORIZATION (daemon,
                                     NULL,
                                     ISCSI_MODULE_POLICY_ACTION_ID,
                                     arg_options,
                                     N_("Authentication is required to discover firmware targets"),
                                     invocation);

  /* Perform the discovery. */
  err = discover_firmware (object, &nodes, &nodes_cnt, &errorstr);

  if (err != 0)
    {
      /* Discovery failed. */
      g_dbus_method_invocation_return_error (invocation,
                                             UDISKS_ERROR,
                                             iscsi_error_to_udisks_error (err),
                                             N_("Discovery failed: %s"),
                                             errorstr);
      g_free (errorstr);
      goto out;
    }

  /* Return discovered portals. */
  udisks_manager_iscsi_initiator_complete_discover_firmware (object, invocation, nodes, nodes_cnt);

out:
  /* Indicate that we handled the method invocation. */
  return TRUE;
}

static gboolean
handle_login (UDisksManagerISCSIInitiator *object,
              GDBusMethodInvocation       *invocation,
              const gchar                 *arg_name,
              gint                         arg_tpgt,
              const gchar                 *arg_address,
              gint                         arg_port,
              const gchar                 *arg_iface,
              GVariant                    *arg_options)
{
  UDisksLinuxManagerISCSIInitiator *manager = UDISKS_LINUX_MANAGER_ISCSI_INITIATOR (object);
  UDisksDaemon *daemon;
  gint err = 0;
  gchar *errorstr = NULL;
  GError *error = NULL;
  UDisksObject *iscsi_object = NULL;
  UDisksObject *iscsi_session_object = NULL;

  daemon = udisks_module_get_daemon (UDISKS_MODULE (manager->module));

  /* Policy check. */
  UDISKS_DAEMON_CHECK_AUTHORIZATION (daemon,
                                     NULL,
                                     ISCSI_MODULE_POLICY_ACTION_ID,
                                     arg_options,
                                     N_("Authentication is required to perform iSCSI login"),
                                     invocation);

  /* Enter a critical section. */
  udisks_linux_module_iscsi_lock_libiscsi_context (manager->module);

  /* Login */
  err = iscsi_login (manager->module, arg_name, arg_tpgt, arg_address, arg_port, arg_iface, arg_options, &errorstr);

  /* Leave the critical section. */
  udisks_linux_module_iscsi_unlock_libiscsi_context (manager->module);

  if (err != 0)
    {
      /* Login failed. */
      g_dbus_method_invocation_return_error (invocation,
                                             UDISKS_ERROR,
                                             iscsi_error_to_udisks_error (err),
                                             N_("Login failed: %s"),
                                             errorstr);
      goto out;
    }

  /* sit and wait until the device appears on dbus */
  iscsi_object = udisks_daemon_wait_for_object_sync (daemon,
                                                     wait_for_iscsi_object,
                                                     g_strdup (arg_name),
                                                     g_free,
                                                     UDISKS_DEFAULT_WAIT_TIMEOUT,
                                                     &error);
   if (iscsi_object == NULL)
    {
      g_prefix_error (&error, "Error waiting for iSCSI device to appear: ");
      g_dbus_method_invocation_take_error (invocation, error);
      goto out;
    }

  if (udisks_manager_iscsi_initiator_get_sessions_supported (UDISKS_MANAGER_ISCSI_INITIATOR (manager)))
    {
      iscsi_session_object = udisks_daemon_wait_for_object_sync (daemon,
                                                                 wait_for_iscsi_session_object,
                                                                 g_strdup (arg_name),
                                                                 g_free,
                                                                 UDISKS_DEFAULT_WAIT_TIMEOUT,
                                                                 &error);
      if (iscsi_session_object == NULL)
        {
          g_prefix_error (&error, "Error waiting for iSCSI session object to appear: ");
          g_dbus_method_invocation_take_error (invocation, error);
          goto out;
        }
    }

  /* Complete DBus call. */
  udisks_manager_iscsi_initiator_complete_login (object, invocation);

out:
  g_clear_object (&iscsi_object);
  g_clear_object (&iscsi_session_object);
  g_free (errorstr);

  /* Indicate that we handled the method invocation. */
  return TRUE;
}

static gboolean
handle_logout(UDisksManagerISCSIInitiator *object,
              GDBusMethodInvocation       *invocation,
              const gchar                 *arg_name,
              gint                         arg_tpgt,
              const gchar                 *arg_address,
              gint                         arg_port,
              const gchar                 *arg_iface,
              GVariant                    *arg_options)
{
  UDisksLinuxManagerISCSIInitiator *manager = UDISKS_LINUX_MANAGER_ISCSI_INITIATOR (object);
  UDisksDaemon *daemon;
  gint err = 0;
  gchar *errorstr = NULL;
  GError *error = NULL;

  daemon = udisks_module_get_daemon (UDISKS_MODULE (manager->module));

  /* Policy check. */
  UDISKS_DAEMON_CHECK_AUTHORIZATION (daemon,
                                     NULL,
                                     ISCSI_MODULE_POLICY_ACTION_ID,
                                     arg_options,
                                     N_("Authentication is required to perform iSCSI logout"),
                                     invocation);

  /* Enter a critical section. */
  udisks_linux_module_iscsi_lock_libiscsi_context (manager->module);

  /* Logout */
  err = iscsi_logout (manager->module, arg_name, arg_tpgt, arg_address, arg_port, arg_iface, arg_options, &errorstr);

  /* Leave the critical section. */
  udisks_linux_module_iscsi_unlock_libiscsi_context (manager->module);

  if (err != 0)
    {
      /* Logout failed. */
      g_dbus_method_invocation_return_error (invocation,
                                             UDISKS_ERROR,
                                             iscsi_error_to_udisks_error (err),
                                             N_("Logout failed: %s"),
                                             errorstr);
      goto out;
    }

  /* now sit and wait until the device and session disappear on dbus */
  if (!udisks_daemon_wait_for_object_to_disappear_sync (daemon,
                                                        wait_for_iscsi_object,
                                                        g_strdup (arg_name),
                                                        g_free,
                                                        UDISKS_DEFAULT_WAIT_TIMEOUT,
                                                        &error))
    {
      g_prefix_error (&error, "Error waiting for iSCSI device to disappear: ");
      g_dbus_method_invocation_take_error (invocation, error);
      goto out;
    }

  if (udisks_manager_iscsi_initiator_get_sessions_supported (UDISKS_MANAGER_ISCSI_INITIATOR (manager)))
    {
      if (!udisks_daemon_wait_for_object_to_disappear_sync (daemon,
                                                            wait_for_iscsi_session_object,
                                                            g_strdup (arg_name),
                                                            g_free,
                                                            UDISKS_DEFAULT_WAIT_TIMEOUT,
                                                            &error))
        {
          g_prefix_error (&error, "Error waiting for iSCSI session object to disappear: ");
          g_dbus_method_invocation_take_error (invocation, error);
          goto out;
        }
    }

  /* Complete DBus call. */
  udisks_manager_iscsi_initiator_complete_logout (object, invocation);

out:
  g_free (errorstr);

  /* Indicate that we handled the method invocation. */
  return TRUE;
}

/* ---------------------------------------------------------------------------------------------------- */

static void
udisks_linux_manager_iscsi_initiator_iface_init (UDisksManagerISCSIInitiatorIface *iface)
{
  iface->handle_get_firmware_initiator_name = handle_get_firmware_initiator_name;
  iface->handle_get_initiator_name = handle_get_initiator_name;
  iface->handle_get_initiator_name_raw = handle_get_initiator_name_raw;
  iface->handle_set_initiator_name = handle_set_initiator_name;
  iface->handle_discover_send_targets = handle_discover_send_targets;
  iface->handle_discover_firmware = handle_discover_firmware;
  iface->handle_login = handle_login;
  iface->handle_logout = handle_logout;
}
