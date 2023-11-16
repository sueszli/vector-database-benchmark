#include "private.h"

#include <Elementary.h>
#include "config.h"
#include "termio.h"
#include "termpty.h"
#include "options.h"
#include "options_themepv.h"
#include "theme.h"
#include "main.h"
#include "colors.h"

static void
_row_set(Evas_Object *o, int y, const char *txt)
{
   Evas_Textgrid_Cell *tc;
   int x, tw, th;
   const char *s;
   int fg, bg;

   evas_object_textgrid_size_get(o, &tw, &th);
   if (y >= th) return;
   tc = evas_object_textgrid_cellrow_get(o, y);
   if (!tc) return;
   fg = COL_DEF;
   bg = COL_INVIS;
   for (s = txt, x = 0; x < tw; x++)
     {
        unsigned int codepoint = ' ';

        if ((s) && (*s == 0))
          {
             s = NULL;
             codepoint = ' ';
          }
        else if (s)
          {
             while ((*s <= 0x02) && (*s != 0))
               {
                  if (*s == 0x01)
                    {
                       s++;
                       if (((*s) >= 0x01) && (*s <= 0x0a))      fg = (*s - 0x01);
                       else if (((*s) >= 0x11) && (*s <= 0x1a)) fg = 12 + (*s - 0x11);
                    }
                  else if (*s == 0x02)
                    {
                       s++;
                       if (((*s) >= 0x01) && (*s <= 0x0a))      bg = (*s - 0x01);
                       else if (((*s) >= 0x11) && (*s <= 0x1a)) bg = 12 + (*s - 0x11);
                    }
                  s++;
               }
             if (*s != 0) codepoint = *s;
             else s = NULL;
          }

        tc[x].fg_extended = 0;
        tc[x].bg_extended = 0;
        tc[x].underline = 0;
        tc[x].strikethrough = 0;
        tc[x].double_width = 0;
        tc[x].fg_extended = 0;
        tc[x].bg_extended = 0;

        tc[x].fg = fg;
        tc[x].bg = bg;
        tc[x].codepoint = codepoint;
        if (s) s++;
     }
   evas_object_textgrid_cellrow_set(o, y, tc);
   evas_object_textgrid_update_add(o, 0, y, tw, 1);
}

static void
_fill_gyw_textgrid(Evas_Object *o)
{
   int x = 0, y = 0;
   int tw, th;
   Evas_Textgrid_Cell *tc;
   const char *FGs[] = {
        "  ",
        "  ",
        " 0",
        "b0",
        " 1",
        "b1",
        " 2",
        "b2",
        " 3",
        "b3",
        " 4",
        "b4",
        " 5",
        "b5",
        " 6",
        "b6",
        " 7",
        "b7",
        NULL
   };
   const char *fg_string;
   const char *title = "        0   1   2   3   4   5   6   7";
   const char *s;

   evas_object_textgrid_size_get(o, &tw, &th);

   /* write title line */
   s = title;
   tc = evas_object_textgrid_cellrow_get(o, 0);
   if (!tc)
     return;
   for (x = 0; x < tw && *s; x++, s++)
     {
        tc[x] = (Evas_Textgrid_Cell) {
             .fg = COL_DEF,
             .bg = COL_INVIS,
             .codepoint = s[0],
        };
     }
   for (; x < tw; x++)
     {
        tc[x] = (Evas_Textgrid_Cell) {
             .fg = COL_DEF,
             .bg = COL_INVIS,
             .codepoint = ' ',
        };
     }

   for (fg_string = FGs[0], y = 1;
        fg_string && y < th;
        fg_string = FGs[y], y++)
     {
        int fg, bg;
        Eina_Bool bold = (y % 2) == 0;

        fg = (y - 1) / 2;
        if (bold)
          fg += 12;
        bg = COL_INVIS;
        x = 0;

        tc = evas_object_textgrid_cellrow_get(o, y);
        if (!tc) return;

        /* write color fg */
        for (s = fg_string; x < tw && *s; x++, s++)
          {
             tc[x] = (Evas_Textgrid_Cell) {
                  .fg = COL_DEF,
                  .bg = COL_INVIS,
                  .codepoint = s[0],
             };
          }

        /* write colors */
        for (bg = 0; bg <= 8; bg++)
          {
             if (x < tw)
               {
                  tc[x] = (Evas_Textgrid_Cell) {
                       .fg = COL_DEF,
                          .bg = COL_INVIS,
                          .codepoint = ' ',
                  };
                  x++;
               }
             for (s = "gYw"; x < tw && *s; x++, s++)
               {
                  tc[x] = (Evas_Textgrid_Cell) {
                       .fg = fg,
                       .bg = (bg == 0) ? COL_INVIS : bg,
                       .bold = bold,
                       .codepoint = s[0],
                  };
               }
          }
        for (; x < tw; x++)
          {
             tc[x] = (Evas_Textgrid_Cell) {
                  .fg = COL_DEF,
                  .bg = COL_INVIS,
                  .codepoint = ' ',
             };
          }
     }
   for (; y < th; ++y)
     {
        tc = evas_object_textgrid_cellrow_get(o, y);
        if (!tc) return;

        for (x = 0; x < tw; x++)
          {
             tc[x] = (Evas_Textgrid_Cell) {
                  .fg = COL_DEF,
                  .bg = COL_INVIS,
                  .codepoint = ' ',
             };
          }
     }
   evas_object_textgrid_update_add(o, 0, 0, tw, y);
}

static void
_cb_resize(void *_data EINA_UNUSED,
           Evas *_e EINA_UNUSED,
           Evas_Object *obj,
           void *_info EINA_UNUSED)
{
   Evas_Object *grid = obj;
   Evas_Object *textgrid = evas_object_data_get(grid, "textgrid");
   Evas_Object *cursor = evas_object_data_get(grid, "cursor");
   Evas_Object *selection = evas_object_data_get(grid, "selection");
   Evas_Coord w, h, ww, hh;

   evas_object_geometry_get(grid, NULL, NULL, &w, &h);
   evas_object_textgrid_cell_size_get(textgrid, &ww, &hh);
   elm_grid_size_set(grid, w, h);
   elm_grid_pack(grid, textgrid, 0, 0, w, h);
   elm_grid_pack(grid, cursor, 0, 0, ww, hh);
   elm_grid_pack(grid, selection, ww * 5, hh * 1, ww * 8, hh * 1);
}


Evas_Object *
options_theme_preview_add(Evas_Object *parent,
                          const Config *config,
                          const char *file,
                          const Color_Scheme *cs,
                          Evas_Coord w, Evas_Coord h,
                          Eina_Bool colors_mode)
{
   Evas_Object *o, *oo, *obase, *grid;
   Evas *evas = evas_object_evas_get(parent);
   Evas_Coord ww, hh, y;

   if (colors_mode)
     {
        o = elm_layout_add(parent);
        theme_apply(o, config, "terminology/colorscheme_preview",
                    file, cs, EINA_TRUE);
        evas_object_show(o);
        obase = oo = o;
     }
   else
     {
        obase = elm_grid_add(parent);
        elm_grid_size_set(obase, w, h);

        // make core frame
        o = elm_layout_add(parent);
        theme_apply(o, config, "terminology/background",
                    file, cs, EINA_TRUE);
        if (config->translucent)
          elm_layout_signal_emit(o, "translucent,on", "terminology");
        else
          elm_layout_signal_emit(o, "translucent,off", "terminology");

        elm_layout_signal_emit(o, "focus,in", "terminology");
        elm_layout_signal_emit(o, "tabcount,on", "terminology");
        elm_layout_signal_emit(o, "tabmissed,on", "terminology");
        elm_layout_text_set(o, "terminology.tabcount.label", "5/8");
        elm_layout_text_set(o, "terminology.tabmissed.label", "2");
        elm_grid_pack(obase, o, 0, 0, w, h);
        evas_object_show(o);

        oo = o;

        // create a bg and swallow into core frame
        o = elm_layout_add(parent);
        theme_apply(o, config, "terminology/core",
                    file, cs, EINA_TRUE);
        if (config->translucent)
          elm_layout_signal_emit(o, "translucent,on", "terminology");
        else
          elm_layout_signal_emit(o, "translucent,off", "terminology");
        elm_layout_signal_emit(o, "focus,in", "terminology");
        evas_object_show(o);
        elm_layout_content_set(oo, "terminology.content", o);

        oo = o;
     }

   // create a grid proportional layout to hold selection, textgrid and cursor
   grid = o = elm_grid_add(parent);
   evas_object_event_callback_add(o, EVAS_CALLBACK_RESIZE, _cb_resize, NULL);
   elm_grid_size_set(o, 100, 100);
   evas_object_show(o);
   elm_layout_content_set(oo, "terminology.content", o);

   // create a texgrid and swallow pack into grid
   o = evas_object_textgrid_add(evas);
   colors_term_init(o, cs ? cs: config->color_scheme);
   if (config->font.bitmap)
     {
        char buf[PATH_MAX];

        snprintf(buf, sizeof(buf), "%s/fonts/%s",
                 elm_app_data_dir_get(), config->font.name);
        evas_object_textgrid_font_set(o, buf, config->font.size);
     }
   else
     evas_object_textgrid_font_set(o, config->font.name, config->font.size);
   evas_object_scale_set(o, elm_config_scale_get());
   evas_object_textgrid_size_set(o, COLOR_MODE_PREVIEW_WIDTH,
                                 COLOR_MODE_PREVIEW_HEIGHT);

   evas_object_textgrid_cell_size_get(o, &ww, &hh);
   if (ww < 1) ww = 1;
   if (hh < 1) hh = 1;

   if (colors_mode)
     _fill_gyw_textgrid(o);
   else
     {
   // cmds:
   // \x01 = set fg
   // \x02 = set bg
   //
   // cols:
   //
   // \x01 = def
   // \x02 = black
   // \x03 = red
   // ...
   // \x09 = white
   //
   // \x11 = def BRIGHT
   // \x12 = black BRIGHT
   // \x13 = red BRIGHT
   // ...
   // \x19 = white BRIGHT

#define F(_x) "\x01"_x
#define X "\x01\x01\x02\x10"
        _row_set(o, 0, " "F("\x04")"$"X" "F("\x19")">"X" test");
        _row_set(o, 1, F("\x02")"black"X" "F("\x03")"red"X" "F("\x04")"green"X" "F("\x05")"yellow");
        _row_set(o, 2, F("\x06")"blue"X" "F("\x07")"mag"X" "F("\x08")"cyan"X" "F("\x09")"white");
        _row_set(o, 3, F("\x12")"black"X" "F("\x13")"red"X" "F("\x14")"green"X" "F("\x15")"yellow");
        _row_set(o, 4, F("\x16")"blue"X" "F("\x17")"mag"X" "F("\x18")"cyan"X" "F("\x19")"white");
        for (y = 5; y < 10; y++) _row_set(o, y, "");
     }

   evas_object_show(o);
   evas_object_data_set(grid, "textgrid", o);
   elm_grid_pack(grid, o, 0, 0, 100, 100);

   // create a cursor and put it in the grid
   o = elm_layout_add(parent);
   theme_apply(o, config, "terminology/cursor",
               file, cs, EINA_TRUE);
   if (colors_mode)
     elm_layout_signal_emit(o, "focus,in,noblink", "terminology");
   else
     elm_layout_signal_emit(o, "focus,in", "terminology");
   evas_object_show(o);
   evas_object_data_set(grid, "cursor", o);
   elm_grid_pack(grid, o, 0, 0, 10, 10);

   // create a selection and put it in the grid
   o = edje_object_add(evas);
   theme_apply(o, config, "terminology/selection",
               file, cs, EINA_FALSE);
   edje_object_signal_emit(o, "focus,in", "terminology");
   edje_object_signal_emit(o, "mode,oneline", "terminology");
   evas_object_show(o);
   evas_object_data_set(grid, "selection", o);
   elm_grid_pack(grid, o, 0, 0, 10, 10);

   evas_object_size_hint_min_set(obase, w, h);
   return obase;
}
