#include "exm-search-row.h"

#include "exm-install-button.h"

#include "exm-types.h"
#include "exm-enums.h"

#include <glib/gi18n.h>

struct _ExmSearchRow
{
    GtkListBoxRow parent_instance;

    ExmSearchResult *search_result;
    gboolean is_installed;
    gboolean is_supported;
    gchar *uuid;

    GtkLabel *description_label;
    ExmInstallButton *install_btn;
    GtkLabel *title;
    GtkLabel *subtitle;
};

G_DEFINE_FINAL_TYPE (ExmSearchRow, exm_search_row, GTK_TYPE_LIST_BOX_ROW)

enum {
    PROP_0,
    PROP_SEARCH_RESULT,
    PROP_IS_INSTALLED,
    PROP_IS_SUPPORTED,
    N_PROPS
};

static GParamSpec *properties [N_PROPS];

ExmSearchRow *
exm_search_row_new (ExmSearchResult *search_result,
                    gboolean         is_installed,
                    gboolean         is_supported)
{
    return g_object_new (EXM_TYPE_SEARCH_ROW,
                         "search-result", search_result,
                         "is-installed", is_installed,
                         "is-supported", is_supported,
                         NULL);
}

static void
exm_search_row_finalize (GObject *object)
{
    ExmSearchRow *self = (ExmSearchRow *)object;

    G_OBJECT_CLASS (exm_search_row_parent_class)->finalize (object);
}

static void
exm_search_row_get_property (GObject    *object,
                             guint       prop_id,
                             GValue     *value,
                             GParamSpec *pspec)
{
    ExmSearchRow *self = EXM_SEARCH_ROW (object);

    switch (prop_id)
    {
    case PROP_SEARCH_RESULT:
        g_value_set_object (value, self->search_result);
        break;
    case PROP_IS_INSTALLED:
        g_value_set_boolean (value, self->is_installed);
        break;
    case PROP_IS_SUPPORTED:
        g_value_set_boolean (value, self->is_supported);
        break;
    default:
        G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
exm_search_row_set_property (GObject      *object,
                             guint         prop_id,
                             const GValue *value,
                             GParamSpec   *pspec)
{
    ExmSearchRow *self = EXM_SEARCH_ROW (object);

    switch (prop_id)
    {
    case PROP_SEARCH_RESULT:
        self->search_result = g_value_get_object (value);
        if (self->search_result)
        {
            // TODO: Bind here, rather than in constructed()
            g_object_get (self->search_result,
                          "uuid", &self->uuid,
                          NULL);
        }
        break;
    case PROP_IS_INSTALLED:
        self->is_installed = g_value_get_boolean (value);
        break;
    case PROP_IS_SUPPORTED:
        self->is_supported = g_value_get_boolean (value);
        break;
    default:
        G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
install_remote (GtkButton    *button,
                ExmSearchRow *self)
{
    gboolean warn;
    ExmInstallButtonState state;

    g_object_get (self->install_btn, "state", &state, NULL);

    warn = (state == EXM_INSTALL_BUTTON_STATE_UNSUPPORTED);
    gtk_widget_activate_action (GTK_WIDGET (button),
                                "ext.install",
                                "(sb)", self->uuid, warn);
}

static void
exm_search_row_constructed (GObject *object)
{
    // TODO: This big block of property assignments is currently copy/pasted
    // from ExmExtension. We can replace this with GtkExpression lookups
    // once blueprint-compiler supports expressions.
    // (See https://gitlab.gnome.org/jwestman/blueprint-compiler/-/issues/5)

    ExmSearchRow *self = EXM_SEARCH_ROW (object);

    ExmInstallButtonState install_state;

    gchar *uri;
    int pk;

    gchar *uuid, *name, *creator, *icon_uri, *screenshot_uri, *link, *description;
    g_object_get (self->search_result,
                  "uuid", &uuid,
                  "name", &name,
                  "creator", &creator,
                  "icon", &icon_uri,
                  "screenshot", &screenshot_uri,
                  "link", &link,
                  "description", &description,
                  "pk", &pk,
                  NULL);

    uri = g_uri_resolve_relative ("https://extensions.gnome.org/",
                                  link,
                                  G_URI_FLAGS_NONE,
                                  NULL);

    gtk_actionable_set_action_name (GTK_ACTIONABLE (self), "win.show-detail");
    gtk_actionable_set_action_target (GTK_ACTIONABLE (self), "s", uuid);

    gtk_label_set_label (self->title, name);
    gtk_label_set_label (self->subtitle, creator);
    gtk_label_set_label (self->description_label, description);

    install_state = self->is_installed
        ? EXM_INSTALL_BUTTON_STATE_INSTALLED
        : (self->is_supported
           ? EXM_INSTALL_BUTTON_STATE_DEFAULT
           : EXM_INSTALL_BUTTON_STATE_UNSUPPORTED);

    g_signal_connect (self->install_btn, "clicked", G_CALLBACK (install_remote), self);
    g_object_set (self->install_btn, "state", install_state, NULL);

    G_OBJECT_CLASS (exm_search_row_parent_class)->constructed (object);
}

static void
exm_search_row_class_init (ExmSearchRowClass *klass)
{
    GObjectClass *object_class = G_OBJECT_CLASS (klass);

    object_class->finalize = exm_search_row_finalize;
    object_class->get_property = exm_search_row_get_property;
    object_class->set_property = exm_search_row_set_property;
    object_class->constructed = exm_search_row_constructed;

    properties [PROP_SEARCH_RESULT] =
        g_param_spec_object ("search-result",
                             "Search Result",
                             "Search Result",
                             EXM_TYPE_SEARCH_RESULT,
                             G_PARAM_READWRITE|G_PARAM_CONSTRUCT_ONLY);

    properties [PROP_IS_INSTALLED] =
        g_param_spec_boolean ("is-installed",
                              "Is Installed",
                              "Is Installed",
                              FALSE,
                              G_PARAM_READWRITE|G_PARAM_CONSTRUCT_ONLY);

    properties [PROP_IS_SUPPORTED] =
        g_param_spec_boolean ("is-supported",
                              "Is Supported",
                              "Is Supported",
                              FALSE,
                              G_PARAM_READWRITE|G_PARAM_CONSTRUCT_ONLY);

    g_object_class_install_properties (object_class, N_PROPS, properties);

    GtkWidgetClass *widget_class = GTK_WIDGET_CLASS (klass);

    gtk_widget_class_set_template_from_resource (widget_class, "/com/mattjakeman/ExtensionManager/exm-search-row.ui");

    gtk_widget_class_bind_template_child (widget_class, ExmSearchRow, description_label);
    gtk_widget_class_bind_template_child (widget_class, ExmSearchRow, install_btn);
    gtk_widget_class_bind_template_child (widget_class, ExmSearchRow, title);
    gtk_widget_class_bind_template_child (widget_class, ExmSearchRow, subtitle);
}

static void
exm_search_row_init (ExmSearchRow *self)
{
    gtk_widget_init_template (GTK_WIDGET (self));
}
