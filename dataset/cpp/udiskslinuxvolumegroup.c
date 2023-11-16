/* -*- mode: C; c-file-style: "gnu"; indent-tabs-mode: nil; -*-
 *
 * Copyright (C) 2007-2010 David Zeuthen <zeuthen@gmail.com>
 * Copyright (C) 2013 Marius Vollmer <marius.vollmer@gmail.com>
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
 */

#include "config.h"
#include <glib/gi18n-lib.h>

#include <sys/types.h>
#include <pwd.h>
#include <grp.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <mntent.h>

#include <glib/gstdio.h>

#include <blockdev/fs.h>
#include <blockdev/lvm.h>

#include <src/udiskslogging.h>
#include <src/udiskslinuxprovider.h>
#include <src/udisksdaemon.h>
#include <src/udisksstate.h>
#include <src/udisksdaemonutil.h>
#include <src/udiskslinuxdevice.h>
#include <src/udiskslinuxblock.h>
#include <src/udiskslinuxblockobject.h>

#include "udiskslinuxvolumegroup.h"
#include "udiskslinuxvolumegroupobject.h"
#include "udiskslinuxlogicalvolume.h"
#include "udiskslinuxlogicalvolumeobject.h"

#include "udiskslvm2daemonutil.h"
#include "jobhelpers.h"

/**
 * SECTION:udiskslinuxvolume_group
 * @title: UDisksLinuxVolumeGroup
 * @short_description: Linux implementation of #UDisksVolumeGroup
 *
 * This type provides an implementation of the #UDisksVolumeGroup interface
 * on Linux.
 */

typedef struct _UDisksLinuxVolumeGroupClass   UDisksLinuxVolumeGroupClass;

/**
 * UDisksLinuxVolumeGroup:
 *
 * The #UDisksLinuxVolumeGroup structure contains only private data and should
 * only be accessed using the provided API.
 */
struct _UDisksLinuxVolumeGroup
{
  UDisksVolumeGroupSkeleton parent_instance;
};

struct _UDisksLinuxVolumeGroupClass
{
  UDisksVolumeGroupSkeletonClass parent_class;
};

static void volume_group_iface_init (UDisksVolumeGroupIface *iface);

G_DEFINE_TYPE_WITH_CODE (UDisksLinuxVolumeGroup, udisks_linux_volume_group, UDISKS_TYPE_VOLUME_GROUP_SKELETON,
                         G_IMPLEMENT_INTERFACE (UDISKS_TYPE_VOLUME_GROUP, volume_group_iface_init));

/* ---------------------------------------------------------------------------------------------------- */

static void
udisks_linux_volume_group_finalize (GObject *object)
{
  if (G_OBJECT_CLASS (udisks_linux_volume_group_parent_class)->finalize != NULL)
    G_OBJECT_CLASS (udisks_linux_volume_group_parent_class)->finalize (object);
}

static void
udisks_linux_volume_group_init (UDisksLinuxVolumeGroup *volume_group)
{
  g_dbus_interface_skeleton_set_flags (G_DBUS_INTERFACE_SKELETON (volume_group),
                                       G_DBUS_INTERFACE_SKELETON_FLAGS_HANDLE_METHOD_INVOCATIONS_IN_THREAD);
}

static void
udisks_linux_volume_group_class_init (UDisksLinuxVolumeGroupClass *klass)
{
  GObjectClass *gobject_class;

  gobject_class = G_OBJECT_CLASS (klass);
  gobject_class->finalize = udisks_linux_volume_group_finalize;
}

/**
 * udisks_linux_volume_group_new:
 *
 * Creates a new #UDisksLinuxVolumeGroup instance.
 *
 * Returns: A new #UDisksLinuxVolumeGroup. Free with g_object_unref().
 */
UDisksVolumeGroup *
udisks_linux_volume_group_new (void)
{
  return UDISKS_VOLUME_GROUP (g_object_new (UDISKS_TYPE_LINUX_VOLUME_GROUP,
                                            NULL));
}

/**
 * udisks_linux_volume_group_update:
 * @volume_group: A #UDisksLinuxVolumeGroup.
 * @object: The enclosing #UDisksLinuxVolumeGroupObject instance.
 *
 * Updates the interface.
 */
void
udisks_linux_volume_group_update (UDisksLinuxVolumeGroup *volume_group,
                                  BDLVMVGdata            *vg_info,
                                  GSList                 *vg_pvs,
                                  gboolean               *needs_polling_ret)
{
  UDisksVolumeGroup *iface = UDISKS_VOLUME_GROUP (volume_group);

  udisks_volume_group_set_name (iface, vg_info->name);
  udisks_volume_group_set_uuid (iface, vg_info->uuid);
  udisks_volume_group_set_size (iface, vg_info->size);
  udisks_volume_group_set_free_size (iface, vg_info->free);
  udisks_volume_group_set_extent_size (iface, vg_info->extent_size);

  {
    g_autoptr(GStrvBuilder) missing_builder = g_strv_builder_new ();
    g_auto(GStrv) missing = NULL;

    for (GSList *vg_pvs_p=vg_pvs; vg_pvs_p; vg_pvs_p=vg_pvs_p->next)
      {
        BDLVMPVdata *pv_info = vg_pvs_p->data;
        if (pv_info->missing)
          g_strv_builder_add (missing_builder, pv_info->pv_uuid);
      }

    missing = g_strv_builder_end (missing_builder);
    udisks_volume_group_set_missing_physical_volumes (iface, (const gchar *const *)missing);
  }

  g_dbus_interface_skeleton_flush (G_DBUS_INTERFACE_SKELETON (iface));
}

/* ---------------------------------------------------------------------------------------------------- */

static gboolean
handle_poll (UDisksVolumeGroup *_group,
             GDBusMethodInvocation *invocation)
{
  UDisksLinuxVolumeGroup *group = UDISKS_LINUX_VOLUME_GROUP (_group);
  UDisksLinuxVolumeGroupObject *object = NULL;
  GError *error = NULL;

  object = udisks_daemon_util_dup_object (group, &error);
  if (object == NULL)
    {
      g_dbus_method_invocation_take_error (invocation, error);
      goto out;
    }

  udisks_linux_volume_group_object_poll (object);

  udisks_volume_group_complete_poll (_group, invocation);

 out:
  g_clear_object (&object);
  return TRUE;
}

/* ---------------------------------------------------------------------------------------------------- */

/*  transfer-full  */
GList *
udisks_linux_volume_group_get_logical_volumes (UDisksVolumeGroup *group,
                                               UDisksDaemon      *daemon)
{
  GList *ret = NULL;
  GList *l, *objects = NULL;
  GDBusObject *object;
  UDisksLogicalVolume *volume;

  object = g_dbus_interface_get_object (G_DBUS_INTERFACE (group));
  if (object == NULL)
    goto out;

  objects = udisks_daemon_get_objects (daemon);
  for (l = objects; l != NULL; l = l->next)
    {
      volume = udisks_object_peek_logical_volume (UDISKS_OBJECT (l->data));
      if (volume &&
          g_strcmp0 (udisks_logical_volume_get_volume_group (volume),
                     g_dbus_object_get_object_path (object)) == 0)
        ret = g_list_append (ret, g_object_ref (volume));
    }

 out:
  g_list_free_full (objects, g_object_unref);
  return ret;
}

/* ---------------------------------------------------------------------------------------------------- */

static gboolean
teardown_volume_group (UDisksVolumeGroup     *group,
                       UDisksDaemon          *daemon,
                       GDBusMethodInvocation *invocation,
                       GVariant              *options,
                       GError               **error)
{
  GList *volumes;
  GList *l;
  UDisksLogicalVolume *volume;

  volumes = udisks_linux_volume_group_get_logical_volumes (group, daemon);
  for (l = volumes; l; l = l->next)
    {
      volume = UDISKS_LOGICAL_VOLUME (l->data);
      if (g_strcmp0 (udisks_logical_volume_get_type_ (volume), "pool") != 0)
        {
          if (!udisks_linux_logical_volume_teardown_block (volume,
                                                           daemon,
                                                           invocation,
                                                           options,
                                                           error))
            {
              g_list_free_full (volumes, g_object_unref);
              return FALSE;
            }
        }
    }
  g_list_free_full (volumes, g_object_unref);

  return TRUE;
}

static gboolean
handle_delete (UDisksVolumeGroup     *_group,
               GDBusMethodInvocation *invocation,
               gboolean               arg_wipe,
               GVariant              *arg_options)
{
  GError *error = NULL;
  UDisksLinuxVolumeGroup *group = UDISKS_LINUX_VOLUME_GROUP (_group);
  UDisksLinuxVolumeGroupObject *object = NULL;
  UDisksLinuxModuleLVM2 *module;
  UDisksDaemon *daemon;
  uid_t caller_uid;
  gboolean teardown_flag = FALSE;
  GList *objects_to_wipe = NULL;
  GList *l;
  VGJobData data;

  g_variant_lookup (arg_options, "tear-down", "b", &teardown_flag);

  object = udisks_daemon_util_dup_object (group, &error);
  if (object == NULL)
    {
      g_dbus_method_invocation_take_error (invocation, error);
      goto out;
    }

  module = udisks_linux_volume_group_object_get_module (object);
  daemon = udisks_module_get_daemon (UDISKS_MODULE (module));

  /* Find physical volumes to wipe. */
  if (arg_wipe)
    {
      GList *objects = udisks_daemon_get_objects (daemon);
      for (l = objects; l; l = l->next)
        {
          UDisksPhysicalVolume *physical_volume;
          physical_volume = udisks_object_peek_physical_volume (UDISKS_OBJECT (l->data));
          if (physical_volume
              && g_strcmp0 (udisks_physical_volume_get_volume_group (physical_volume),
                            g_dbus_object_get_object_path (G_DBUS_OBJECT (object))) == 0)
            objects_to_wipe = g_list_append (objects_to_wipe, g_object_ref (l->data));
        }
      g_list_free_full (objects, g_object_unref);
    }

  if (!udisks_daemon_util_get_caller_uid_sync (daemon,
                                               invocation,
                                               NULL /* GCancellable */,
                                               &caller_uid,
                                               &error))
    {
      g_dbus_method_invocation_return_gerror (invocation, error);
      g_clear_error (&error);
      goto out;
    }

  /* Policy check. */
  UDISKS_DAEMON_CHECK_AUTHORIZATION (daemon,
                                     UDISKS_OBJECT (object),
                                     LVM2_POLICY_ACTION_ID,
                                     arg_options,
                                     N_("Authentication is required to delete a volume group"),
                                     invocation);

  if (teardown_flag &&
      !teardown_volume_group (_group,
                              daemon,
                              invocation,
                              arg_options,
                              &error))
    {
      g_dbus_method_invocation_take_error (invocation, error);
      goto out;
    }

  data.vg_name = udisks_linux_volume_group_object_get_name (object);

  if (!udisks_daemon_launch_threaded_job_sync (daemon,
                                               UDISKS_OBJECT (object),
                                               "lvm-vg-delete",
                                               caller_uid,
                                               vgremove_job_func,
                                               &data,
                                               NULL, /* user_data_free_func */
                                               NULL, /* GCancellable */
                                               &error))
    {
      g_dbus_method_invocation_return_error (invocation,
                                             UDISKS_ERROR,
                                             UDISKS_ERROR_FAILED,
                                             "Error deleting volume group: %s",
                                             error->message);
      g_clear_error (&error);
      goto out;
    }

  for (l = objects_to_wipe; l; l = l->next)
    {
      UDisksBlock *block = udisks_object_peek_block (l->data);
      if (block)
        udisks_daemon_util_lvm2_wipe_block (daemon, block, NULL);
    }

  udisks_volume_group_complete_delete (_group, invocation);

 out:
  g_list_free_full (objects_to_wipe, g_object_unref);
  g_clear_object (&object);
  return TRUE;
}

/* ---------------------------------------------------------------------------------------------------- */

typedef struct
{
  UDisksLinuxModuleLVM2 *module;
  const gchar *name;
} WaitForVolumeGroupObjectData;

static UDisksObject *
wait_for_volume_group_object (UDisksDaemon *daemon,
                              gpointer      user_data)
{
  WaitForVolumeGroupObjectData *data = user_data;
  UDisksLinuxVolumeGroupObject *object;

  object = udisks_linux_module_lvm2_find_volume_group_object (data->module, data->name);

  if (object == NULL)
    return NULL;

  return g_object_ref (UDISKS_OBJECT (object));
}

static gboolean
handle_rename (UDisksVolumeGroup     *_group,
               GDBusMethodInvocation *invocation,
               const gchar           *new_name,
               GVariant              *options)
{
  GError *error = NULL;
  UDisksLinuxVolumeGroup *group = UDISKS_LINUX_VOLUME_GROUP (_group);
  UDisksLinuxVolumeGroupObject *object = NULL;
  UDisksLinuxModuleLVM2 *module;
  UDisksDaemon *daemon;
  uid_t caller_uid;
  UDisksObject *group_object = NULL;
  VGJobData data;
  WaitForVolumeGroupObjectData wait_data;

  object = udisks_daemon_util_dup_object (group, &error);
  if (object == NULL)
    {
      g_dbus_method_invocation_take_error (invocation, error);
      goto out;
    }

  module = udisks_linux_volume_group_object_get_module (object);
  daemon = udisks_module_get_daemon (UDISKS_MODULE (module));

  if (!udisks_daemon_util_get_caller_uid_sync (daemon,
                                               invocation,
                                               NULL /* GCancellable */,
                                               &caller_uid,
                                               &error))
    {
      g_dbus_method_invocation_return_gerror (invocation, error);
      g_clear_error (&error);
      goto out;
    }

  /* Policy check. */
  UDISKS_DAEMON_CHECK_AUTHORIZATION (daemon,
                                     UDISKS_OBJECT (object),
                                     LVM2_POLICY_ACTION_ID,
                                     options,
                                     N_("Authentication is required to rename a volume group"),
                                     invocation);

  data.vg_name = udisks_linux_volume_group_object_get_name (object);
  data.new_vg_name = new_name;

  if (!udisks_daemon_launch_threaded_job_sync (daemon,
                                               UDISKS_OBJECT (object),
                                               "lvm-vg-rename",
                                               caller_uid,
                                               vgrename_job_func,
                                               &data,
                                               NULL, /* user_data_free_func */
                                               NULL, /* GCancellable */
                                               &error))
    {
      g_dbus_method_invocation_return_error (invocation,
                                             UDISKS_ERROR,
                                             UDISKS_ERROR_FAILED,
                                             "Error renaming volume group: %s",
                                             error->message);
      g_clear_error (&error);
      goto out;
    }

  wait_data.module = module;
  wait_data.name = new_name;
  group_object = udisks_daemon_wait_for_object_sync (daemon,
                                                     wait_for_volume_group_object,
                                                     &wait_data,
                                                     NULL,
                                                     UDISKS_DEFAULT_WAIT_TIMEOUT,
                                                     &error);
  if (group_object == NULL)
    {
      g_prefix_error (&error,
                      "Error waiting for volume group object for '%s': ",
                      new_name);
      g_dbus_method_invocation_take_error (invocation, error);
      goto out;
    }

  udisks_volume_group_complete_rename (_group,
                                       invocation,
                                       g_dbus_object_get_object_path (G_DBUS_OBJECT (group_object)));
  g_clear_object (&group_object);

 out:
  g_clear_object (&object);
  return TRUE;
}

/* ---------------------------------------------------------------------------------------------------- */

static gboolean
handle_add_device (UDisksVolumeGroup     *_group,
                   GDBusMethodInvocation *invocation,
                   const gchar           *new_member_device_objpath,
                   GVariant              *options)
{
  UDisksLinuxVolumeGroup *group = UDISKS_LINUX_VOLUME_GROUP (_group);
  UDisksLinuxModuleLVM2 *module;
  UDisksDaemon *daemon;
  UDisksLinuxVolumeGroupObject *object;
  uid_t caller_uid;
  GError *error = NULL;
  UDisksObject *new_member_device_object = NULL;
  UDisksBlock *new_member_device = NULL;
  UDisksPhysicalVolume *physical_volume = NULL;
  VGJobData data;

  object = udisks_daemon_util_dup_object (group, &error);
  if (object == NULL)
    {
      g_dbus_method_invocation_take_error (invocation, error);
      goto out;
    }

  module = udisks_linux_volume_group_object_get_module (object);
  daemon = udisks_module_get_daemon (UDISKS_MODULE (module));

  error = NULL;
  if (!udisks_daemon_util_get_caller_uid_sync (daemon,
                                               invocation,
                                               NULL /* GCancellable */,
                                               &caller_uid,
                                               &error))
    {
      g_dbus_method_invocation_return_gerror (invocation, error);
      g_clear_error (&error);
      goto out;
    }

  new_member_device_object = udisks_daemon_find_object (daemon, new_member_device_objpath);
  if (new_member_device_object == NULL)
    {
      g_dbus_method_invocation_return_error (invocation, UDISKS_ERROR, UDISKS_ERROR_FAILED,
                                             "No device for given object path");
      goto out;
    }

  new_member_device = udisks_object_get_block (new_member_device_object);
  if (new_member_device == NULL)
    {
      g_dbus_method_invocation_return_error (invocation, UDISKS_ERROR, UDISKS_ERROR_FAILED,
                                             "No block interface on given object");
      goto out;
    }

  /* Policy check. */
  UDISKS_DAEMON_CHECK_AUTHORIZATION (daemon,
                                     UDISKS_OBJECT (object),
                                     LVM2_POLICY_ACTION_ID,
                                     options,
                                     N_("Authentication is required to add a device to a volume group"),
                                     invocation);

  if (!udisks_daemon_util_lvm2_block_is_unused (new_member_device, &error))
    {
      g_dbus_method_invocation_take_error (invocation, error);
      goto out;
    }

  if (!udisks_daemon_util_lvm2_wipe_block (daemon, new_member_device, &error))
    {
      g_dbus_method_invocation_take_error (invocation, error);
      goto out;
    }

  physical_volume = udisks_object_peek_physical_volume (new_member_device_object);
  if (!physical_volume)
    {
      PVJobData pv_data;
      pv_data.path = udisks_block_get_device (new_member_device);
      if (!udisks_daemon_launch_threaded_job_sync (daemon,
                                                   UDISKS_OBJECT (object),
                                                   "lvm-pv-create",
                                                   caller_uid,
                                                   pvcreate_job_func,
                                                   &pv_data,
                                                   NULL, /* user_data_free_func */
                                                   NULL, /* GCancellable */
                                                   &error))
        {
          g_dbus_method_invocation_return_error (invocation,
                                                 UDISKS_ERROR,
                                                 UDISKS_ERROR_FAILED,
                                                 "Error creating LVM metadata on %s: %s",
                                                 pv_data.path,
                                                 error->message);
          g_clear_error (&error);
          goto out;
        }
    }


  data.vg_name = udisks_linux_volume_group_object_get_name (object);
  data.pv_path = udisks_block_get_device (new_member_device);

  if (!udisks_daemon_launch_threaded_job_sync (daemon,
                                               UDISKS_OBJECT (object),
                                               "lvm-vg-add-device",
                                               caller_uid,
                                               vgextend_job_func,
                                               &data,
                                               NULL, /* user_data_free_func */
                                               NULL, /* GCancellable */
                                               &error))
    {
      g_dbus_method_invocation_return_error (invocation,
                                             UDISKS_ERROR,
                                             UDISKS_ERROR_FAILED,
                                             "Error adding %s to volume group: %s",
                                             data.pv_path,
                                             error->message);
      g_clear_error (&error);
      goto out;
    }

  udisks_volume_group_complete_add_device (_group, invocation);

 out:
  g_clear_object (&new_member_device_object);
  g_clear_object (&new_member_device);
  g_clear_object (&object);
  return TRUE; /* returning TRUE means that we handled the method invocation */
}

/* ---------------------------------------------------------------------------------------------------- */
static gboolean
handle_remove_common (UDisksVolumeGroup     *_group,
                      GDBusMethodInvocation *invocation,
                      const gchar           *member_device_objpath,
                      GVariant              *options,
                      gboolean               is_remove,
                      gboolean               arg_wipe)
{
  UDisksLinuxVolumeGroup *group = UDISKS_LINUX_VOLUME_GROUP (_group);
  UDisksLinuxModuleLVM2 *module;
  UDisksDaemon *daemon;
  UDisksLinuxVolumeGroupObject *object;
  uid_t caller_uid;
  GError *error = NULL;
  UDisksObject *member_device_object = NULL;
  UDisksBlock *member_device = NULL;
  VGJobData data;
  const gchar *authentication_error_msg = NULL;
  const gchar *job_operation = NULL;
  UDisksThreadedJobFunc job_func = NULL;
  gboolean do_wipe = FALSE;

  if (is_remove)
    {
      authentication_error_msg = N_("Authentication is required to remove a device from a volume group");
      job_operation = "lvm-vg-rem-device";
      job_func = vgreduce_job_func;

      if (arg_wipe)
        do_wipe = TRUE;
    }
  else
    {
      authentication_error_msg = N_("Authentication is required to empty a device in a volume group");
      job_operation = "lvm-vg-empty-device";
      job_func = pvmove_job_func;
    }

  object = udisks_daemon_util_dup_object (group, &error);
  if (object == NULL)
    {
      g_dbus_method_invocation_take_error (invocation, error);
      goto out;
    }

  module = udisks_linux_volume_group_object_get_module (object);
  daemon = udisks_module_get_daemon (UDISKS_MODULE (module));

  error = NULL;
  if (!udisks_daemon_util_get_caller_uid_sync (daemon,
                                               invocation,
                                               NULL /* GCancellable */,
                                               &caller_uid,
                                               &error))
    {
      g_dbus_method_invocation_return_gerror (invocation, error);
      g_clear_error (&error);
      goto out;
    }

  member_device_object = udisks_daemon_find_object (daemon, member_device_objpath);
  if (member_device_object == NULL)
    {
      g_dbus_method_invocation_return_error (invocation, UDISKS_ERROR, UDISKS_ERROR_FAILED,
                                             "No device for given object path");
      goto out;
    }

  member_device = udisks_object_get_block (member_device_object);
  if (member_device == NULL)
    {
      g_dbus_method_invocation_return_error (invocation, UDISKS_ERROR, UDISKS_ERROR_FAILED,
                                             "No block interface on given object");
      goto out;
    }

  /* Policy check. */
  UDISKS_DAEMON_CHECK_AUTHORIZATION (daemon,
                                     UDISKS_OBJECT (object),
                                     LVM2_POLICY_ACTION_ID,
                                     options,
                                     authentication_error_msg,
                                     invocation);

  if (is_remove)
    data.vg_name = udisks_linux_volume_group_object_get_name (object);

  data.pv_path = udisks_block_get_device (member_device);

  if (!udisks_daemon_launch_threaded_job_sync (daemon,
                                               UDISKS_OBJECT (object),
                                               job_operation,
                                               caller_uid,
                                               job_func,
                                               &data,
                                               NULL, /* user_data_free_func */
                                               NULL, /* GCancellable */
                                               &error))
    {
      g_dbus_method_invocation_return_error (invocation,
                                             UDISKS_ERROR,
                                             UDISKS_ERROR_FAILED,
                                             (is_remove) ? "Error remove %s from volume group: %s" : "Error emptying %s: %s",
                                             data.pv_path,
                                             error->message);
      g_clear_error (&error);
      goto out;
    }

  if (do_wipe)
    {
      if (!udisks_daemon_launch_threaded_job_sync (daemon,
                                                   UDISKS_OBJECT (object),
                                                   "pv-format-erase",
                                                   caller_uid,
                                                   pvremove_job_func,
                                                   &data,
                                                   NULL, /* user_data_free_func */
                                                   NULL, /* GCancellable */
                                                   &error))
        {
          g_dbus_method_invocation_return_error (invocation,
                                                 UDISKS_ERROR,
                                                 UDISKS_ERROR_FAILED,
                                                 "Error wiping %s after removal from volume group %s: %s",
                                                 data.pv_path,
                                                 udisks_linux_volume_group_object_get_name (object),
                                                 error->message);
          g_clear_error (&error);
          goto out;
        }
    }

  udisks_volume_group_complete_remove_device (_group, invocation);

 out:
  g_clear_object (&member_device_object);
  g_clear_object (&member_device);
  g_clear_object (&object);
  return TRUE; /* returning TRUE means that we handled the method invocation */
}


static gboolean
handle_remove_device (UDisksVolumeGroup     *_group,
                      GDBusMethodInvocation *invocation,
                      const gchar           *member_device_objpath,
                      gboolean               arg_wipe,
                      GVariant              *options)
{
  return handle_remove_common (_group, invocation, member_device_objpath,
                               options, TRUE, arg_wipe);
}

/* ---------------------------------------------------------------------------------------------------- */

static gboolean
handle_empty_device (UDisksVolumeGroup     *_group,
                     GDBusMethodInvocation *invocation,
                     const gchar           *member_device_objpath,
                     GVariant              *options)
{
 return handle_remove_common (_group, invocation, member_device_objpath,
                              options, FALSE, FALSE);
}

/* ---------------------------------------------------------------------------------------------------- */

static gboolean
handle_remove_missing_physical_volumes (UDisksVolumeGroup     *_group,
                                        GDBusMethodInvocation *invocation,
                                        GVariant              *options)
{
  UDisksLinuxVolumeGroup *group = UDISKS_LINUX_VOLUME_GROUP (_group);
  UDisksLinuxModuleLVM2 *module;
  UDisksDaemon *daemon;
  UDisksLinuxVolumeGroupObject *object;
  uid_t caller_uid;
  GError *error = NULL;
  VGJobData data;

  object = udisks_daemon_util_dup_object (group, &error);
  if (object == NULL)
    {
      g_dbus_method_invocation_take_error (invocation, error);
      goto out;
    }

  module = udisks_linux_volume_group_object_get_module (object);
  daemon = udisks_module_get_daemon (UDISKS_MODULE (module));

  error = NULL;
  if (!udisks_daemon_util_get_caller_uid_sync (daemon,
                                               invocation,
                                               NULL /* GCancellable */,
                                               &caller_uid,
                                               &error))
    {
      g_dbus_method_invocation_return_gerror (invocation, error);
      g_clear_error (&error);
      goto out;
    }

  /* Policy check. */
  UDISKS_DAEMON_CHECK_AUTHORIZATION (daemon,
                                     UDISKS_OBJECT (object),
                                     LVM2_POLICY_ACTION_ID,
                                     options,
                                     N_("Authentication is required to remove missing physical volumes from a volume group"),
                                     invocation);

  data.vg_name = udisks_linux_volume_group_object_get_name (object);
  data.pv_path = NULL;

  if (!udisks_daemon_launch_threaded_job_sync (daemon,
                                               UDISKS_OBJECT (object),
                                               "lvm-vg-rem-device",
                                               caller_uid,
                                               vgreduce_job_func,
                                               &data,
                                               NULL, /* user_data_free_func */
                                               NULL, /* GCancellable */
                                               &error))
    {
      g_dbus_method_invocation_return_error (invocation,
                                             UDISKS_ERROR,
                                             UDISKS_ERROR_FAILED,
                                             "Error removing missing physical volumes: %s",
                                             error->message);
      g_clear_error (&error);
      goto out;
    }

  udisks_volume_group_complete_remove_missing_physical_volumes (_group, invocation);

 out:
  g_clear_object (&object);
  return TRUE; /* returning TRUE means that we handled the method invocation */
}

/* ---------------------------------------------------------------------------------------------------- */

struct WaitData {
  UDisksLinuxVolumeGroupObject *group_object;
  const gchar *name;
};

static UDisksObject *
wait_for_logical_volume_object (UDisksDaemon *daemon,
                                gpointer      user_data)
{
  struct WaitData *data = user_data;
  UDisksLinuxLogicalVolumeObject *object;

  object = udisks_linux_volume_group_object_find_logical_volume_object (data->group_object, data->name);
  if (object == NULL)
    return NULL;

  return g_object_ref (UDISKS_OBJECT (object));
}

static const gchar *
wait_for_logical_volume_path (UDisksLinuxVolumeGroupObject  *group_object,
                              const gchar                   *name,
                              GError                       **error)
{
  struct WaitData data;
  UDisksLinuxModuleLVM2 *module;
  UDisksDaemon *daemon;
  UDisksObject *volume_object;
  const gchar *object_path;

  data.group_object = group_object;
  data.name = name;
  module = udisks_linux_volume_group_object_get_module (group_object);
  daemon = udisks_module_get_daemon (UDISKS_MODULE (module));
  volume_object = udisks_daemon_wait_for_object_sync (daemon,
                                                      wait_for_logical_volume_object,
                                                      &data,
                                                      NULL,
                                                      UDISKS_DEFAULT_WAIT_TIMEOUT,
                                                      error);
  if (volume_object == NULL)
    return NULL;

  object_path = g_dbus_object_get_object_path (G_DBUS_OBJECT (volume_object));
  g_object_unref (volume_object);

  return object_path;
}

/* ---------------------------------------------------------------------------------------------------- */

enum VolumeType { VOL_PLAIN, VOL_THIN_POOL, VOL_THIN_VOLUME, VOL_VDO_VOLUME };
typedef void (*VolumeCompletionFunc) (UDisksVolumeGroup     *object,
                                      GDBusMethodInvocation *invocation,
                                      const gchar           *result);

static gboolean
handle_create_volume (UDisksVolumeGroup              *_group,
                      GDBusMethodInvocation          *invocation,
                      const gchar                    *arg_name,
                      guint64                         arg_size,
                      GVariant                       *options,
                      enum VolumeType                 vol_creation_type,
                      const gchar                    *arg_pool,
                      guint64                         arg_virtual_size,
                      guint64                         arg_index_memory,
                      gboolean                        arg_compression,
                      gboolean                        arg_deduplication,
                      const gchar                    *arg_write_policy,
                      const gchar                    *arg_layout,
                      const gchar             *const *arg_pvs)
{
  GError *error = NULL;
  UDisksLinuxVolumeGroup *group = UDISKS_LINUX_VOLUME_GROUP (_group);
  UDisksLinuxVolumeGroupObject *object = NULL;
  UDisksLinuxModuleLVM2 *module;
  UDisksDaemon *daemon;
  uid_t caller_uid;
  const gchar *lv_objpath;
  LVJobData data = {0};
  g_auto(GStrv) pvs = NULL;
  UDisksLinuxLogicalVolumeObject *pool_object = NULL;
  const gchar *auth_error_msg = NULL;
  UDisksThreadedJobFunc create_function = NULL;
  VolumeCompletionFunc completion_function = NULL;

  if (VOL_PLAIN == vol_creation_type)
    {
      auth_error_msg = N_("Authentication is required to create a logical volume");
      create_function = lvcreate_job_func;
      completion_function = udisks_volume_group_complete_create_plain_volume;
    }
  else if (VOL_THIN_VOLUME == vol_creation_type)
    {
      auth_error_msg = N_("Authentication is required to create a thin volume");
      create_function = lvcreate_thin_job_func;
      completion_function = udisks_volume_group_complete_create_thin_volume;
    }
  else if (VOL_THIN_POOL == vol_creation_type)
    {
      auth_error_msg = N_("Authentication is required to create a thin pool volume");
      create_function = lvcreate_thin_pool_job_func;
      completion_function = udisks_volume_group_complete_create_thin_pool_volume;
    }
  else if (VOL_VDO_VOLUME == vol_creation_type)
    {
      auth_error_msg = N_("Authentication is required to create a VDO volume");
      create_function = lvcreate_vdo_job_func;
      completion_function = udisks_volume_group_complete_create_vdo_volume;
    }

  object = udisks_daemon_util_dup_object (group, &error);
  if (object == NULL)
    {
      g_dbus_method_invocation_take_error (invocation, error);
      goto out;
    }

  module = udisks_linux_volume_group_object_get_module (object);
  daemon = udisks_module_get_daemon (UDISKS_MODULE (module));
  if (!udisks_daemon_util_get_caller_uid_sync (daemon,
                                               invocation,
                                               NULL /* GCancellable */,
                                               &caller_uid,
                                               &error))
    {
      g_dbus_method_invocation_return_gerror (invocation, error);
      g_clear_error (&error);
      goto out;
    }

  /* Policy check. */
  UDISKS_DAEMON_CHECK_AUTHORIZATION (daemon,
                                     UDISKS_OBJECT (object),
                                     LVM2_POLICY_ACTION_ID,
                                     options,
                                     auth_error_msg,
                                     invocation);

  data.vg_name = udisks_linux_volume_group_object_get_name (object);
  data.new_lv_name = arg_name;
  data.new_lv_size = arg_size;

  if (VOL_PLAIN == vol_creation_type)
    {
      if (arg_layout != NULL
          && g_strcmp0 (arg_layout, "linear") != 0
          && !g_str_has_prefix (arg_layout, "raid")
          && g_strcmp0 (arg_layout, "mirror") != 0)
        {
          g_dbus_method_invocation_return_error (invocation, UDISKS_ERROR, UDISKS_ERROR_FAILED,
                                                 "Not a valid layout for a plain volume: %s", arg_layout);
          goto out;
        }

      data.new_lv_layout = arg_layout;
      data.new_lv_stripes = 0;
      data.new_lv_mirrors = 0;

      if (arg_pvs)
        {
          gint32 n_subvolumes = g_strv_length ((gchar **)arg_pvs);
          g_variant_lookup (options, "subvolumes", "i", &n_subvolumes);

          if (g_strcmp0 (arg_layout, "raid0") == 0)
            data.new_lv_stripes = n_subvolumes;
          else if (g_strcmp0 (arg_layout, "raid1") == 0)
            data.new_lv_mirrors = n_subvolumes - 1;
          else if (g_strcmp0 (arg_layout, "raid10") == 0) {
            if (n_subvolumes % 2 != 0)
              {
                g_dbus_method_invocation_return_error (invocation, UDISKS_ERROR, UDISKS_ERROR_FAILED,
                                                       "Number of subvolumes for raid10 must be even");
                goto out;
              }
            data.new_lv_stripes = n_subvolumes / 2;
          } else if (g_strcmp0 (arg_layout, "raid4") == 0 ||g_strcmp0 (arg_layout, "raid5") == 0)
            data.new_lv_stripes = n_subvolumes - 1;
          else if (g_strcmp0 (arg_layout, "raid6") == 0)
            data.new_lv_stripes = n_subvolumes - 2;

          pvs = udisks_daemon_util_lvm2_gather_pvs (daemon, object, arg_pvs, &error);
          if (pvs == NULL)
            {
              g_dbus_method_invocation_take_error (invocation, error);
              goto out;
            }
          data.new_lv_pvs = (const gchar **)pvs;
        }
    }

  if (VOL_THIN_POOL == vol_creation_type)
    data.extent_size = udisks_volume_group_get_extent_size (UDISKS_VOLUME_GROUP (group));

  if (VOL_THIN_VOLUME == vol_creation_type)
    {
      pool_object = UDISKS_LINUX_LOGICAL_VOLUME_OBJECT (udisks_daemon_find_object (daemon, arg_pool));
      if (pool_object == NULL || !UDISKS_IS_LINUX_LOGICAL_VOLUME_OBJECT (pool_object))
        {
          g_dbus_method_invocation_return_error (invocation, UDISKS_ERROR, UDISKS_ERROR_FAILED,
                                                 "Not a logical volume");
          goto out;
        }
      data.pool_name = udisks_linux_logical_volume_object_get_name (pool_object);
    }

  if (VOL_VDO_VOLUME == vol_creation_type)
    {
      data.pool_name = arg_pool;
      data.virtual_size = arg_virtual_size;
      data.index_memory = arg_index_memory;
      data.compression = arg_compression;
      data.deduplication = arg_deduplication;
      data.write_policy = arg_write_policy;
    }

  if (!udisks_daemon_launch_threaded_job_sync (daemon,
                                               UDISKS_OBJECT (object),
                                               "lvm-vg-create-volume",
                                               caller_uid,
                                               create_function,
                                               &data,
                                               NULL, /* user_data_free_func */
                                               NULL, /* GCancellable */
                                               &error))
    {
      g_dbus_method_invocation_return_error (invocation,
                                             UDISKS_ERROR,
                                             UDISKS_ERROR_FAILED,
                                             "Error creating volume: %s",
                                             error->message);
      g_clear_error (&error);
      goto out;
    }

  lv_objpath = wait_for_logical_volume_path (object, arg_name, &error);
  if (lv_objpath == NULL)
    {
      g_prefix_error (&error,
                      "Error waiting for logical volume object for '%s': ",
                      arg_name);
      g_dbus_method_invocation_take_error (invocation, error);
      goto out;
    }

  completion_function (_group, invocation, lv_objpath);

 out:
  g_clear_object (&pool_object);
  g_clear_object (&object);
  return TRUE;
}

static gboolean
handle_create_plain_volume (UDisksVolumeGroup     *_group,
                            GDBusMethodInvocation *invocation,
                            const gchar           *arg_name,
                            guint64                arg_size,
                            GVariant              *options)
{
  return handle_create_volume(_group, invocation, arg_name, arg_size, options,
                              VOL_PLAIN, NULL, 0, 0, FALSE, FALSE, NULL,
                              NULL, NULL);
}

static gboolean
handle_create_plain_volume_with_layout (UDisksVolumeGroup     *_group,
                                        GDBusMethodInvocation *invocation,
                                        const gchar           *arg_name,
                                        guint64                arg_size,
                                        const gchar           *arg_layout,
                                        const gchar *const    *arg_pvs,
                                        GVariant              *options)
{
  return handle_create_volume(_group, invocation, arg_name, arg_size, options,
                              VOL_PLAIN, NULL, 0, 0, FALSE, FALSE, NULL,
                              arg_layout, arg_pvs);
}

/* ---------------------------------------------------------------------------------------------------- */

static gboolean
handle_create_thin_pool_volume (UDisksVolumeGroup     *_group,
                                GDBusMethodInvocation *invocation,
                                const gchar           *arg_name,
                                guint64                arg_size,
                                GVariant              *options)
{
  return handle_create_volume(_group, invocation, arg_name, arg_size, options,
                              VOL_THIN_POOL, NULL, 0, 0, FALSE, FALSE, NULL,
                              NULL, NULL);
}

/* ---------------------------------------------------------------------------------------------------- */

static gboolean
handle_create_thin_volume (UDisksVolumeGroup     *_group,
                           GDBusMethodInvocation *invocation,
                           const gchar           *arg_name,
                           guint64                arg_size,
                           const gchar           *arg_pool,
                           GVariant              *options)
{
  return handle_create_volume(_group, invocation, arg_name, arg_size, options,
                              VOL_THIN_VOLUME, arg_pool, 0, 0, FALSE, FALSE, NULL,
                              NULL, NULL);
}

static gboolean
handle_create_vdo_volume (UDisksVolumeGroup     *_group,
                          GDBusMethodInvocation *invocation,
                          const gchar           *arg_name,
                          const gchar           *arg_pool,
                          guint64                arg_size,
                          guint64                arg_virtual_size,
                          guint64                arg_index_memory,
                          gboolean               arg_compression,
                          gboolean               arg_deduplication,
                          const gchar           *arg_write_policy,
                          GVariant              *options)
{
  return handle_create_volume (_group, invocation, arg_name, arg_size, options,
                               VOL_VDO_VOLUME, arg_pool, arg_virtual_size, arg_index_memory,
                               arg_compression, arg_deduplication, arg_write_policy,
                               NULL, NULL);
}

/* ---------------------------------------------------------------------------------------------------- */

static void
volume_group_iface_init (UDisksVolumeGroupIface *iface)
{
  iface->handle_poll = handle_poll;

  iface->handle_delete = handle_delete;
  iface->handle_rename = handle_rename;

  iface->handle_add_device = handle_add_device;
  iface->handle_remove_device = handle_remove_device;
  iface->handle_empty_device = handle_empty_device;
  iface->handle_remove_missing_physical_volumes = handle_remove_missing_physical_volumes;

  iface->handle_create_plain_volume = handle_create_plain_volume;
  iface->handle_create_plain_volume_with_layout = handle_create_plain_volume_with_layout;
  iface->handle_create_thin_pool_volume = handle_create_thin_pool_volume;
  iface->handle_create_thin_volume = handle_create_thin_volume;

  iface->handle_create_vdo_volume = handle_create_vdo_volume;
}
