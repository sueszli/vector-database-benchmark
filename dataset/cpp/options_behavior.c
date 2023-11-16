#include "private.h"


#include <math.h>
#include <Elementary.h>
#include <assert.h>
#include "termpty.h"
#include "backlog.h"
#include "config.h"
#include "termio.h"
#include "options.h"
#include "options_behavior.h"
#include "main.h"
#include "theme.h"

typedef struct tag_Behavior_Ctx {
     Config *config;
     Evas_Object *term;
     Evas_Object *op_w;
     Evas_Object *op_h;
     Evas_Object *op_wh_current;
     Evas_Object *backlock_label;
     char *backlog_msg;
} Behavior_Ctx;

extern Eina_Bool multisense_available;

OPTIONS_CB(Behavior_Ctx, jump_on_change, 0);
OPTIONS_CB(Behavior_Ctx, jump_on_keypress, 0);
OPTIONS_CB(Behavior_Ctx, disable_visual_bell, 1);
OPTIONS_CB(Behavior_Ctx, bell_rings, 0);
OPTIONS_CB(Behavior_Ctx, flicker_on_key, 0);
OPTIONS_CB(Behavior_Ctx, urg_bell, 0);
OPTIONS_CB(Behavior_Ctx, multi_instance, 0);
OPTIONS_CB(Behavior_Ctx, xterm_256color, 0);
OPTIONS_CB(Behavior_Ctx, erase_is_del, 0);
OPTIONS_CB(Behavior_Ctx, login_shell, 0);
OPTIONS_CB(Behavior_Ctx, show_tabs,  0);
OPTIONS_CB(Behavior_Ctx, mv_always_show, 0);
OPTIONS_CB(Behavior_Ctx, ty_escapes, 0);
OPTIONS_CB(Behavior_Ctx, selection_escapes, 0);
OPTIONS_CB(Behavior_Ctx, changedir_to_current, 0);
OPTIONS_CB(Behavior_Ctx, emoji_dbl_width, 0);
OPTIONS_CB(Behavior_Ctx, group_all, 0);

static unsigned int
sback_double_to_expo_int(double d)
{
    if (d < 1.0)
        return 0;
    if (d >= 17.0)
        d = 17.0;
    return 1 << (unsigned char) d;
}

static char *
sback_indicator_units_format(double d)
{
    return (char*)eina_stringshare_printf("%'d", sback_double_to_expo_int(d));
}
static char *
sback_units_format(double d)
{
    return (char*)eina_stringshare_printf(_("%'d lines"), sback_double_to_expo_int(d));
}

static void
_update_backlog_title(Behavior_Ctx *ctx)
{
   char *factor = " KMG";
   double amount = termpty_backlog_memory_get();

   while (amount > 1024.0 && factor[1] != '\0')
     {
        amount /= 1024;
        factor++;
     }
   eina_stringshare_del(ctx->backlog_msg);
   ctx->backlog_msg = (char*) eina_stringshare_printf(
      _("Scrollback (current memory usage: %'.2f%cB):"),
      amount, factor[0]);
   elm_object_text_set(ctx->backlock_label, ctx->backlog_msg);
}

static void
_cb_op_behavior_sback_chg(void *data,
                          Evas_Object *obj,
                          void *_event EINA_UNUSED)
{
   Behavior_Ctx *ctx = data;
   Config *config = ctx->config;

   config->scrollback = (double) sback_double_to_expo_int(elm_slider_value_get(obj));
   termio_config_update(ctx->term);
   _update_backlog_title(ctx);
   config_save(config);
}

static void
_cb_op_behavior_tab_zoom_slider_chg(void *data,
                                    Evas_Object *obj,
                                    void *_event EINA_UNUSED)
{
   Behavior_Ctx *ctx = data;
   Config *config = ctx->config;

   config->tab_zoom = (double)(int)round(elm_slider_value_get(obj) * 10.0) / 10.0;
   termio_config_update(ctx->term);
   config_save(config);
}

static void
_cb_op_behavior_custom_geometry_current_set(void *data,
                                Evas_Object *obj EINA_UNUSED,
                                void *_event EINA_UNUSED)
{
   Behavior_Ctx *ctx = data;
   Config *config = ctx->config;

   if (config->custom_geometry)
     {
        termio_size_get(ctx->term, &config->cg_width, &config->cg_height);
        elm_spinner_value_set(ctx->op_w, config->cg_width);
        elm_spinner_value_set(ctx->op_h, config->cg_height);
     }
   config_save(config);
}

static void
_cb_op_behavior_custom_geometry(void *data,
                                Evas_Object *obj,
                                void *_event EINA_UNUSED)
{
   Behavior_Ctx *ctx = data;
   Config *config = ctx->config;

   config->custom_geometry = elm_check_state_get(obj);
   if (config->custom_geometry)
     {
        config->cg_width = (int) elm_spinner_value_get(ctx->op_w);
        config->cg_height = (int) elm_spinner_value_get(ctx->op_h);
     }
   config_save(config);

   elm_object_disabled_set(ctx->op_w, !config->custom_geometry);
   elm_object_disabled_set(ctx->op_h, !config->custom_geometry);
   elm_object_disabled_set(ctx->op_wh_current, !config->custom_geometry);
}

static void
_cb_op_behavior_cg_width(void *data,
                         Evas_Object *obj,
                         void *_event EINA_UNUSED)
{
   Behavior_Ctx *ctx = data;
   Config *config = ctx->config;

   if (config->custom_geometry)
     {
        config->cg_width = (int) elm_spinner_value_get(obj);
        config_save(config);
     }
}

static void
_cb_op_behavior_cg_height(void *data,
                          Evas_Object *obj,
                          void *_event EINA_UNUSED)
{
   Behavior_Ctx *ctx = data;
   Config *config = ctx->config;

   if (config->custom_geometry)
     {
        config->cg_height = (int) elm_spinner_value_get(obj);
        config_save(config);
     }
}

static void
_parent_del_cb(void *data,
               Evas *_e EINA_UNUSED,
               Evas_Object *_obj EINA_UNUSED,
               void *_event_info EINA_UNUSED)
{
   Behavior_Ctx *ctx = data;

   eina_stringshare_del(ctx->backlog_msg);
   free(ctx);
}

static void
_cursors_changed_cb(void *data, Evas_Object *obj,
                    void *event_info EINA_UNUSED)
{
   Behavior_Ctx *ctx = data;
   Config *config = ctx->config;
   int value = elm_radio_value_get(obj) - 1;

   config->disable_cursor_blink = value % 2;
   config->cursor_shape = value / 2;

   termio_config_update(ctx->term);
   windows_update();
   config_save(config);
}


static void
_add_cursors_option(Evas_Object *bx,
                    Behavior_Ctx *ctx)
{
   Evas_Object *lbl, *rd, *rdg, *layout;

   lbl = elm_label_add(bx);
   evas_object_size_hint_weight_set(lbl, EVAS_HINT_EXPAND, 0.0);
   evas_object_size_hint_align_set(lbl, 0.0, 0.0);
   elm_layout_text_set(lbl, NULL, _("Default cursor:"));
   elm_box_pack_end(bx, lbl);
   evas_object_show(lbl);

   /* Blinking Block */
   rd = elm_radio_add(bx);
   evas_object_size_hint_weight_set(rd, EVAS_HINT_EXPAND, 0.0);
   evas_object_size_hint_align_set(rd, EVAS_HINT_FILL, 0.5);
   rdg = rd;
   elm_object_text_set(rd, _("Blinking Block"));
   elm_radio_state_value_set(rd, 1);
   layout = elm_layout_add(rd);
   theme_apply(layout, ctx->config, "terminology/cursor",
               NULL, NULL, EINA_TRUE);
   evas_object_size_hint_weight_set(layout, EVAS_HINT_EXPAND, EVAS_HINT_EXPAND);
   evas_object_size_hint_align_set(layout, EVAS_HINT_FILL, EVAS_HINT_FILL);
   elm_image_resizable_set(layout, EINA_FALSE, EINA_FALSE);
   elm_object_part_content_set(rd, "icon", layout);
   elm_box_pack_end(bx, rd);
   evas_object_show(rd);
   elm_layout_signal_emit(layout, "focus,out", "terminology");
   elm_layout_signal_emit(layout, "focus,in", "terminology");
   evas_object_smart_callback_add(rd, "changed", _cursors_changed_cb, ctx);

   /* Steady Block */
   rd = elm_radio_add(bx);
   evas_object_size_hint_weight_set(rd, EVAS_HINT_EXPAND, 0.0);
   evas_object_size_hint_align_set(rd, EVAS_HINT_FILL, 0.5);
   elm_object_text_set(rd, _("Steady Block"));
   elm_radio_state_value_set(rd, 2);
   elm_radio_group_add(rd, rdg);
   layout = elm_layout_add(rd);
   theme_apply(layout, ctx->config, "terminology/cursor",
               NULL, NULL, EINA_TRUE);
   evas_object_size_hint_weight_set(layout, EVAS_HINT_EXPAND, EVAS_HINT_EXPAND);
   evas_object_size_hint_align_set(layout, EVAS_HINT_FILL, EVAS_HINT_FILL);
   elm_image_resizable_set(layout, EINA_FALSE, EINA_FALSE);
   elm_object_part_content_set(rd, "icon", layout);
   elm_box_pack_end(bx, rd);
   evas_object_show(rd);
   elm_layout_signal_emit(layout, "focus,out", "terminology");
   elm_layout_signal_emit(layout, "focus,in,noblink", "terminology");
   evas_object_smart_callback_add(rd, "changed", _cursors_changed_cb, ctx);

   /* Blinking Underline */
   rd = elm_radio_add(bx);
   evas_object_size_hint_weight_set(rd, EVAS_HINT_EXPAND, 0.0);
   evas_object_size_hint_align_set(rd, EVAS_HINT_FILL, 0.5);
   elm_object_text_set(rd, _("Blinking Underline"));
   elm_radio_state_value_set(rd, 3);
   elm_radio_group_add(rd, rdg);
   layout = elm_layout_add(rd);
   theme_apply(layout, ctx->config, "terminology/cursor_underline",
               NULL, NULL, EINA_TRUE);
   evas_object_size_hint_weight_set(layout, EVAS_HINT_EXPAND, EVAS_HINT_EXPAND);
   evas_object_size_hint_align_set(layout, EVAS_HINT_FILL, EVAS_HINT_FILL);
   elm_image_resizable_set(layout, EINA_FALSE, EINA_FALSE);
   elm_object_part_content_set(rd, "icon", layout);
   elm_box_pack_end(bx, rd);
   evas_object_show(rd);
   elm_layout_signal_emit(layout, "focus,out", "terminology");
   elm_layout_signal_emit(layout, "focus,in", "terminology");
   evas_object_smart_callback_add(rd, "changed", _cursors_changed_cb, ctx);

   /* Steady Underline */
   rd = elm_radio_add(bx);
   evas_object_size_hint_weight_set(rd, EVAS_HINT_EXPAND, 0.0);
   evas_object_size_hint_align_set(rd, EVAS_HINT_FILL, 0.5);
   elm_object_text_set(rd, _("Steady Underline"));
   elm_radio_state_value_set(rd, 4);
   elm_radio_group_add(rd, rdg);
   layout = elm_layout_add(rd);
   theme_apply(layout, ctx->config, "terminology/cursor_underline",
               NULL, NULL, EINA_TRUE);
   evas_object_size_hint_weight_set(layout, EVAS_HINT_EXPAND, EVAS_HINT_EXPAND);
   evas_object_size_hint_align_set(layout, EVAS_HINT_FILL, EVAS_HINT_FILL);
   elm_image_resizable_set(layout, EINA_FALSE, EINA_FALSE);
   elm_object_part_content_set(rd, "icon", layout);
   elm_box_pack_end(bx, rd);
   evas_object_show(rd);
   elm_layout_signal_emit(layout, "focus,out", "terminology");
   elm_layout_signal_emit(layout, "focus,in,noblink", "terminology");
   evas_object_smart_callback_add(rd, "changed", _cursors_changed_cb, ctx);

   /* Blinking Bar */
   rd = elm_radio_add(bx);
   evas_object_size_hint_weight_set(rd, EVAS_HINT_EXPAND, 0.0);
   evas_object_size_hint_align_set(rd, EVAS_HINT_FILL, 0.5);
   elm_object_text_set(rd, _("Blinking Bar"));
   elm_radio_state_value_set(rd, 5);
   elm_radio_group_add(rd, rdg);
   layout = elm_layout_add(rd);
   theme_apply(layout, ctx->config, "terminology/cursor_bar",
               NULL, NULL, EINA_TRUE);
   evas_object_size_hint_weight_set(layout, EVAS_HINT_EXPAND, EVAS_HINT_EXPAND);
   evas_object_size_hint_align_set(layout, EVAS_HINT_FILL, EVAS_HINT_FILL);
   elm_image_resizable_set(layout, EINA_FALSE, EINA_FALSE);
   elm_object_part_content_set(rd, "icon", layout);
   elm_box_pack_end(bx, rd);
   evas_object_show(rd);
   elm_layout_signal_emit(layout, "focus,out", "terminology");
   elm_layout_signal_emit(layout, "focus,in", "terminology");
   evas_object_smart_callback_add(rd, "changed", _cursors_changed_cb, ctx);

   /* Steady Bar */
   rd = elm_radio_add(bx);
   evas_object_size_hint_weight_set(rd, EVAS_HINT_EXPAND, 0.0);
   evas_object_size_hint_align_set(rd, EVAS_HINT_FILL, 0.5);
   elm_object_text_set(rd, _("Steady Bar"));
   elm_radio_state_value_set(rd, 6);
   elm_radio_group_add(rd, rdg);
   layout = elm_layout_add(rd);
   theme_apply(layout, ctx->config, "terminology/cursor_bar",
               NULL, NULL, EINA_TRUE);
   evas_object_size_hint_weight_set(layout, EVAS_HINT_EXPAND, EVAS_HINT_EXPAND);
   evas_object_size_hint_align_set(layout, EVAS_HINT_FILL, EVAS_HINT_FILL);
   elm_image_resizable_set(layout, EINA_FALSE, EINA_FALSE);
   elm_object_part_content_set(rd, "icon", layout);
   elm_box_pack_end(bx, rd);
   evas_object_show(rd);
   elm_layout_signal_emit(layout, "focus,out", "terminology");
   elm_layout_signal_emit(layout, "focus,in,noblink", "terminology");
   evas_object_smart_callback_add(rd, "changed", _cursors_changed_cb, ctx);

   elm_radio_value_set(rdg,
     1 +  2 * ctx->config->cursor_shape + (ctx->config->disable_cursor_blink ? 1 : 0));
}

void
options_behavior(Evas_Object *opbox, Evas_Object *term)
{
   Config *config = termio_config_get(term);
   Evas_Object *o, *bx, *sc, *frame, *lbl;
   int w, h;
   const char *tooltip;
   Behavior_Ctx *ctx;

   termio_size_get(term, &w, &h);

   ctx = calloc(1, sizeof(*ctx));
   assert(ctx);

   ctx->config = config;
   ctx->term = term;

   frame = o = elm_frame_add(opbox);
   evas_object_size_hint_weight_set(o, EVAS_HINT_EXPAND, EVAS_HINT_EXPAND);
   evas_object_size_hint_align_set(o, EVAS_HINT_FILL, EVAS_HINT_FILL);
   elm_object_text_set(o, _("Behavior"));
   elm_box_pack_end(opbox, o);
   evas_object_show(o);

   evas_object_event_callback_add(o, EVAS_CALLBACK_DEL,
                                  _parent_del_cb, ctx);

   sc = o = elm_scroller_add(opbox);
   elm_scroller_content_min_limit(sc, EINA_TRUE, EINA_FALSE);
   evas_object_size_hint_weight_set(o, EVAS_HINT_EXPAND, EVAS_HINT_EXPAND);
   evas_object_size_hint_align_set(o, EVAS_HINT_FILL, EVAS_HINT_FILL);
   elm_object_content_set(frame, o);
   evas_object_show(o);

   bx = o = elm_box_add(opbox);
   evas_object_size_hint_weight_set(o, EVAS_HINT_EXPAND, 0.0);
   evas_object_size_hint_align_set(o, EVAS_HINT_FILL, 0.0);
   elm_object_content_set(sc, o);
   evas_object_show(o);

   OPTIONS_CX(_("Show tabs"), show_tabs, 0);
   OPTIONS_SEPARATOR;

   OPTIONS_CX(_("Scroll to bottom on new content"), jump_on_change, 0);
   OPTIONS_CX(_("Scroll to bottom when a key is pressed"), jump_on_keypress, 0);

   OPTIONS_SEPARATOR;

   _add_cursors_option(bx, ctx);

   OPTIONS_CX(_("React to key presses (typing sounds and animations)"), flicker_on_key, 0);
   if (!multisense_available)
     {
        lbl = elm_label_add(bx);
        evas_object_size_hint_weight_set(lbl, EVAS_HINT_EXPAND, 0.0);
        evas_object_size_hint_align_set(lbl, 0.0, 0.5);
        elm_box_pack_before(bx, lbl, o);
        elm_object_text_set(lbl, _("Audio Support for key presses <failure>DISABLED</failure>!"));
        evas_object_show(lbl);
     }
   OPTIONS_SEPARATOR;
   OPTIONS_CX(_("Visual Bell"), disable_visual_bell, 1);
   OPTIONS_CX(_("Bell rings"), bell_rings, 0);
   OPTIONS_CX(_("Urgent Bell"), urg_bell, 0);
   OPTIONS_SEPARATOR;
   OPTIONS_CX(_("Multiple instances, one process"), multi_instance, 0);
   OPTIONS_CX(_("Set TERM to xterm-256color"), xterm_256color, 0);
   OPTIONS_CX(_("BackArrow sends Del (instead of BackSpace)"), erase_is_del, 0);
   OPTIONS_CX(_("Start as login shell"), login_shell, 0);
   OPTIONS_CX(_("Open new terminals in current working directory"), changedir_to_current, 0);
   OPTIONS_CX(_("Always show miniview"), mv_always_show, 0);
   OPTIONS_CX(_("Enable special Terminology escape codes"), ty_escapes, 0);
   OPTIONS_CX(_("Enable escape codes manipulating selections"), selection_escapes, 0);
   OPTIONS_CX(_("Always treat Emojis as double-width characters"), emoji_dbl_width, 0);
   OPTIONS_CX(_("When grouping input, do it on all terminals and not just the visible ones"), group_all, 0);

   OPTIONS_SEPARATOR;

   o = elm_check_add(bx);
   evas_object_size_hint_weight_set(o, EVAS_HINT_EXPAND, 0.0);
   evas_object_size_hint_align_set(o, EVAS_HINT_FILL, 0.5);
   elm_object_text_set(o, _("Always open at size:"));
   elm_check_state_set(o, config->custom_geometry);
   elm_box_pack_end(bx, o);
   evas_object_show(o);
   evas_object_smart_callback_add(o, "changed",
                                  _cb_op_behavior_custom_geometry, ctx);

   ctx->op_wh_current = o = elm_button_add(bx);
   evas_object_size_hint_weight_set(o, 0.0, 0.0);
   evas_object_size_hint_align_set(o, 0.0, 0.5);
   elm_object_text_set(o, _("Set Current:"));
   elm_box_pack_end(bx, o);
   evas_object_show(o);
   elm_object_disabled_set(o, !config->custom_geometry);
   evas_object_smart_callback_add(o, "clicked",
                                  _cb_op_behavior_custom_geometry_current_set,
                                  ctx);

   o = elm_label_add(bx);
   evas_object_size_hint_weight_set(o, 0.0, 0.0);
   evas_object_size_hint_align_set(o, 0.0, 0.5);
   elm_object_text_set(o, _("Width:"));
   elm_box_pack_end(bx, o);
   evas_object_show(o);

   ctx->op_w = o = elm_spinner_add(bx);
   elm_spinner_editable_set(o, EINA_TRUE);
   elm_spinner_min_max_set(o, 2.0, 350.0);
   evas_object_size_hint_weight_set(o, EVAS_HINT_EXPAND, 0.0);
   evas_object_size_hint_align_set(o, EVAS_HINT_FILL, 0.5);
   if (config->custom_geometry)
     elm_spinner_value_set(o, config->cg_width);
   else
     elm_spinner_value_set(o, w);
   elm_object_disabled_set(o, !config->custom_geometry);
   elm_box_pack_end(bx, o);
   evas_object_show(o);
   evas_object_smart_callback_add(o, "changed",
                                  _cb_op_behavior_cg_width, ctx);

   o = elm_label_add(bx);
   evas_object_size_hint_weight_set(o, 0.0, 0.0);
   evas_object_size_hint_align_set(o, 0.0, 0.5);
   elm_object_text_set(o, _("Height:"));
   elm_box_pack_end(bx, o);
   evas_object_show(o);

   ctx->op_h = o = elm_spinner_add(bx);
   elm_spinner_editable_set(o, EINA_TRUE);
   elm_spinner_min_max_set(o, 1.0, 150.0);
   evas_object_size_hint_weight_set(o, EVAS_HINT_EXPAND, 0.0);
   evas_object_size_hint_align_set(o, EVAS_HINT_FILL, 0.5);
   if (config->custom_geometry)
     elm_spinner_value_set(o, config->cg_height);
   else
     elm_spinner_value_set(o, h);
   elm_object_disabled_set(o, !config->custom_geometry);
   elm_box_pack_end(bx, o);
   evas_object_show(o);
   evas_object_smart_callback_add(o, "changed",
                                  _cb_op_behavior_cg_height, ctx);

   OPTIONS_SEPARATOR;

   o = elm_label_add(bx);
   evas_object_size_hint_weight_set(o, 0.0, 0.0);
   evas_object_size_hint_align_set(o, 0.0, 0.5);
   ctx->backlock_label = o;
   _update_backlog_title(ctx);
   elm_box_pack_end(bx, o);
   evas_object_show(o);

   o = elm_slider_add(bx);
   elm_slider_indicator_format_function_set(o,
                                            sback_indicator_units_format,
                                            (void(*)(char*))eina_stringshare_del);
   elm_slider_units_format_function_set(o,
                                        sback_units_format,
                                        (void(*)(char*))eina_stringshare_del);
   elm_slider_span_size_set(o, 40);
   elm_slider_step_set(o, 1);
   /* http://graphics.stanford.edu/~seander/bithacks.html#IntegerLogFloat */
   union {
       float v;
       int c;
   } u;
   u.v = config->scrollback;
   u.c = (u.c >> 23) - 127;
   elm_slider_min_max_set(o, 0.0, 17.0);
   elm_slider_value_set(o, u.c);
   evas_object_size_hint_weight_set(o, EVAS_HINT_EXPAND, 0.0);
   evas_object_size_hint_align_set(o, EVAS_HINT_FILL, 0.0);
   elm_box_pack_end(bx, o);
   evas_object_show(o);
   evas_object_smart_callback_add(o, "delay,changed",
                                  _cb_op_behavior_sback_chg, ctx);

   OPTIONS_SEPARATOR;

   o = elm_label_add(bx);
   evas_object_size_hint_weight_set(o, 0.0, 0.0);
   evas_object_size_hint_align_set(o, 0.0, 0.5);
   tooltip = _("Set the time of the animation that<br>"
       "takes places during tab switches,<br>"
       "be those done due to key bindings or <br>"
       "mouse wheel over the tabs panel");
   elm_object_tooltip_text_set(o, tooltip);
   elm_object_text_set(o, _("Tab zoom/switch animation time:"));
   elm_box_pack_end(bx, o);
   evas_object_show(o);

   o = elm_slider_add(bx);
   evas_object_size_hint_weight_set(o, EVAS_HINT_EXPAND, 0.0);
   evas_object_size_hint_align_set(o, EVAS_HINT_FILL, 0.0);
   elm_slider_span_size_set(o, 40);
   elm_slider_unit_format_set(o, _("%1.1f s"));
   elm_slider_indicator_format_set(o, _("%1.1f s"));
   elm_slider_min_max_set(o, 0.0, 1.0);
   elm_slider_value_set(o, config->tab_zoom);
   elm_box_pack_end(bx, o);
   evas_object_show(o);
   evas_object_smart_callback_add(o, "delay,changed",
                                  _cb_op_behavior_tab_zoom_slider_chg, ctx);

   evas_object_size_hint_weight_set(opbox, EVAS_HINT_EXPAND, EVAS_HINT_EXPAND);
   evas_object_size_hint_align_set(opbox, EVAS_HINT_FILL, EVAS_HINT_FILL);
   evas_object_show(o);

}
