/*
 * Copyright (c) 2017-2021 gnome-mpv
 *
 * This file is part of Celluloid.
 *
 * Celluloid is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * Celluloid is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with Celluloid.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <glib/gi18n.h>
#include <string.h>

#include "celluloid-file-chooser.h"
#include "celluloid-def.h"

static void
load_last_folder(GtkFileChooser *chooser);

static void
save_last_folder(GtkFileChooser *chooser);

static void
response_handler(GtkDialog *dialog, gint response_id, gpointer data);

static void
load_last_folder(GtkFileChooser *chooser)
{
	GSettings *win_config = g_settings_new(CONFIG_WIN_STATE);
	gchar *uri = g_settings_get_string(win_config, "last-folder-uri");

	if(uri && *uri)
	{
		GFile *folder = g_file_new_for_uri(uri);

		gtk_file_chooser_set_current_folder(chooser, folder, NULL);
		g_object_unref(folder);
	}

	g_free(uri);
	g_object_unref(win_config);
}

static void
save_last_folder(GtkFileChooser *chooser)
{
	GFile *folder = gtk_file_chooser_get_current_folder(chooser);
	gchar *uri = g_file_get_uri(folder);
	GSettings *win_config = g_settings_new(CONFIG_WIN_STATE);

	g_settings_set_string(win_config, "last-folder-uri", uri?:"");

	g_object_unref(win_config);
	g_free(uri);
	g_object_unref(folder);
}

static void
response_handler(GtkDialog *dialog, gint response_id, gpointer data)
{
	if(response_id == GTK_RESPONSE_ACCEPT)
	{
		GSettings *main_config;
		gboolean last_folder_enable;

		main_config = g_settings_new(CONFIG_ROOT);
		last_folder_enable =	g_settings_get_boolean
					(main_config, "last-folder-enable");

		if(last_folder_enable)
		{
			save_last_folder(GTK_FILE_CHOOSER(dialog));
		}

		g_object_unref(main_config);
	}
}

CelluloidFileChooser *
celluloid_file_chooser_new(	const gchar *title,
				GtkWindow *parent,
				GtkFileChooserAction action,
				gboolean restore_state )
{
	CelluloidFileChooser *chooser;
	GSettings *main_config;
	gboolean last_folder_enable;

	chooser = gtk_file_chooser_native_new(title, parent, action, NULL, NULL);
	main_config = g_settings_new(CONFIG_ROOT);
	last_folder_enable =	g_settings_get_boolean
				(main_config, "last-folder-enable");

	if(restore_state && last_folder_enable)
	{
		load_last_folder(GTK_FILE_CHOOSER(chooser));
	}

	celluloid_file_chooser_set_modal(chooser, TRUE);

	g_signal_connect(chooser, "response", G_CALLBACK(response_handler), NULL);

	g_object_unref(main_config);

	return chooser;
}

void
celluloid_file_chooser_destroy(CelluloidFileChooser *chooser)
{
	gtk_native_dialog_destroy(GTK_NATIVE_DIALOG(chooser));
	g_object_unref(chooser);
}

void
celluloid_file_chooser_set_default_filters(	CelluloidFileChooser *chooser,
						gboolean audio,
						gboolean video,
						gboolean image,
						gboolean subtitle )
{
	GtkFileChooser *gtk_chooser = GTK_FILE_CHOOSER(chooser);
	GListModel *filters = gtk_file_chooser_get_filters(gtk_chooser);
	const guint filters_count = g_list_model_get_n_items(filters);

	for(guint i = 0; i < filters_count; i++)
	{
		GtkFileFilter *filter = g_list_model_get_item(filters, i);

		gtk_file_chooser_remove_filter(gtk_chooser, filter);
	}

	if(audio || video || image || subtitle)
	{
		GtkFileFilter *filter = gtk_file_filter_new();
		gtk_file_filter_set_name(filter, _("All Files"));
		gtk_file_filter_add_pattern(filter, "*");
		gtk_file_chooser_add_filter(gtk_chooser, filter);
	}

	if(audio && video && image)
	{
		GtkFileFilter *filter = gtk_file_filter_new();
		gtk_file_filter_set_name(filter, _("Media Files"));
		gtk_file_filter_add_mime_type(filter, "audio/*");
		gtk_file_filter_add_mime_type(filter, "video/*");
		gtk_file_filter_add_mime_type(filter, "image/*");
		gtk_file_chooser_add_filter(gtk_chooser, filter);
		gtk_file_chooser_set_filter(gtk_chooser, filter);
	}

	if(audio)
	{
		GtkFileFilter *filter = gtk_file_filter_new();
		gtk_file_filter_set_name(filter, _("Audio Files"));
		gtk_file_filter_add_mime_type(filter, "audio/*");
		gtk_file_chooser_add_filter(gtk_chooser, filter);
	}

	if(video)
	{
		GtkFileFilter *filter = gtk_file_filter_new();
		gtk_file_filter_set_name(filter, _("Video Files"));
		gtk_file_filter_add_mime_type(filter, "video/*");
		gtk_file_chooser_add_filter(gtk_chooser, filter);
	}

	if(image)
	{
		GtkFileFilter *filter = gtk_file_filter_new();
		gtk_file_filter_set_name(filter, _("Image Files"));
		gtk_file_filter_add_mime_type(filter, "image/*");
		gtk_file_chooser_add_filter(gtk_chooser, filter);
	}

	if(subtitle)
	{
		GtkFileFilter *filter = gtk_file_filter_new();
		const gchar *exts[] = SUBTITLE_EXTS;

		gtk_file_filter_set_name(filter, _("Subtitle Files"));

		for(gint i = 0; exts[i]; i++)
		{
			gchar *pattern = g_strdup_printf("*.%s", exts[i]);

			gtk_file_filter_add_pattern(filter, pattern);
			g_free(pattern);
		}

		gtk_file_chooser_add_filter(gtk_chooser, filter);

		if(!(audio || video || image))
		{
			gtk_file_chooser_set_filter(gtk_chooser, filter);
		}
	}

	g_object_unref(filters);
}
