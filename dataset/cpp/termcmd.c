#include "private.h"

#include <Elementary.h>
#include "main.h"
#include "win.h"
#include "termio.h"
#include "config.h"
#include "controls.h"
#include "media.h"
#include "theme.h"
#include "termcmd.h"

static Eina_Bool
_termcmd_search(Evas_Object *_obj EINA_UNUSED,
                Evas_Object *_win EINA_UNUSED,
                Evas_Object *_bg EINA_UNUSED,
                const char *cmd)
{
   if (cmd[0] == 0) // clear search
     {
        printf("search clear\n");
        return EINA_TRUE;
     }
   printf("search '%s'\n", cmd);
   return EINA_TRUE;
}

static Eina_Bool
_termcmd_font_size(Evas_Object *obj,
                   Evas_Object *_win EINA_UNUSED,
                   Evas_Object *_bg EINA_UNUSED,
                   const char *cmd)
{
   Config *config = termio_config_get(obj);

   if (config)
     {
        Term *term = termio_term_get(obj);
        Win *wn = term_win_get(term);
        int new_size;

        if (cmd[0] == 0) // back to default
          {
             config->font.bitmap = config->font.orig_bitmap;
             if (config->font.orig_name)
               {
                  eina_stringshare_del(config->font.name);
                  config->font.name = eina_stringshare_add(config->font.orig_name);
               }
             new_size = config->font.orig_size;
          }
        else if (cmd[0] == 'b') // big font size
          {
             if (config->font.orig_bitmap)
               {
                  config->font.bitmap = 1;
                  eina_stringshare_del(config->font.name);
                  config->font.name = eina_stringshare_add("10x20.pcf");
               }
             new_size = config->font.size * 2;
          }
        else if (cmd[0] == '+') // size up
          {
             int i;

             new_size = config->font.size;
             for (i = 0; cmd[i] == '+'; i++)
               {
                  new_size = ((double)new_size * 1.4) + 1;
               }
          }
        else if (cmd[0] == '-') // size down
          {
             int i;

             new_size = config->font.size;
             for (i = 0; cmd[i] == '-'; i++)
               {
                  new_size = (double)(new_size - 1) / 1.4;
               }
          }
        else
          {
             ERR(_("Unknown font command: %s"), cmd);
             return EINA_TRUE;
          }

        win_font_size_set(wn, new_size);
     }
   return EINA_TRUE;
}

static Eina_Bool
_termcmd_grid_size(Evas_Object *obj,
                   Evas_Object *_win EINA_UNUSED,
                   Evas_Object *_bg EINA_UNUSED,
                   const char *cmd)
{
   int w = -1, h = -1;
   int r = sscanf(cmd, "%ix%i", &w, &h);

   if (r == 1)
     {
        static const int size_table[][2] = {
           { 80, 24 }, { 80, 40 }, { 80, 60 }, { 80, 80 }, { 120, 24 },
           { 120, 40 }, { 120, 60 }, { 120, 80 }, { 120, 120 }
        };

        if (w >= 0 && w <= 8)
          {
             h = size_table[w][1];
             w = size_table[w][0];
          }
     }
   if ((w > 0) && (h > 0))
     termio_grid_size_set(obj, w, h);
   else
     ERR(_("Unknown grid size command: %s"), cmd);

   return EINA_TRUE;
}

static Eina_Bool
_termcmd_background(Evas_Object *obj,
                    Evas_Object *_win EINA_UNUSED,
                    Evas_Object *_bg EINA_UNUSED,
                    const char *cmd)
{
   Config *config = termio_config_get(obj);

   if (!config) return EINA_TRUE;

   if (cmd[0] == 0)
     {
        config->temporary = EINA_TRUE;
        eina_stringshare_replace(&(config->background), NULL);
        main_media_update(config);
     }
   else if (ecore_file_can_read(cmd))
     {
        config->temporary = EINA_TRUE;
        eina_stringshare_replace(&(config->background), cmd);
        main_media_update(config);
     }
   else
     ERR(_("Background file could not be read: %s"), cmd);

   return EINA_TRUE;
}

// called as u type
Eina_Bool
termcmd_watch(Evas_Object *obj, Evas_Object *win, Evas_Object *bg, const char *cmd)
{
   if (!cmd) return EINA_FALSE;
   if ((cmd[0] == '/') || (cmd[0] == 's'))
     return _termcmd_search(obj, win, bg, cmd + 1);
   return EINA_FALSE;
}

// called when you hit enter
Eina_Bool
termcmd_do(Evas_Object *obj, Evas_Object *win, Evas_Object *bg, const char *cmd)
{
   if (!cmd || !cmd[0]) return EINA_FALSE;
   if ((cmd[0] == '/') || (cmd[0] == 's'))
     return _termcmd_search(obj, win, bg, cmd + 1);
   if ((cmd[0] == 'f') || (cmd[0] == 'F'))
     return _termcmd_font_size(obj, win, bg, cmd + 1);
   if ((cmd[0] == 'g') || (cmd[0] == 'G'))
     return _termcmd_grid_size(obj, win, bg, cmd + 1);
   if ((cmd[0] == 'b') || (cmd[0] == 'B'))
     return _termcmd_background(obj, win, bg, cmd + 1);

   ERR(_("Unknown command: %s"), cmd);
   return EINA_FALSE;
}
