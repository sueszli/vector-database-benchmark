/*
 * Copyright (c) 2002-2017 Balabit
 * Copyright (c) 1998-2017 Balázs Scheidler
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

#include "cfg-block-generator.h"
#include "cfg-lexer.h"

const gchar *
cfg_block_generator_format_name_method(CfgBlockGenerator *self, gchar *buf, gsize buf_len)
{
  g_snprintf(buf, buf_len, "%s generator %s",
             cfg_lexer_lookup_context_name_by_type(self->context),
             self->name);
  return buf;
}

gboolean
cfg_block_generator_generate(CfgBlockGenerator *self, GlobalConfig *cfg, gpointer args, GString *result,
                             const gchar *reference)
{
  gchar block_name[1024];
  cfg_block_generator_format_name(self, block_name, sizeof(block_name)/sizeof(block_name[0]));

  g_string_append_printf(result, "\n#Start Block %s\n", block_name);
  const gboolean res = self->generate(self, cfg, args, result, reference);
  g_string_append_printf(result, "\n#End Block %s\n", block_name);

  return res;
}

void
cfg_block_generator_init_instance(CfgBlockGenerator *self, gint context, const gchar *name)
{
  self->ref_cnt = 1;
  self->context = context;
  self->name = g_strdup(name);
  self->format_name = cfg_block_generator_format_name_method;
  self->free_fn = cfg_block_generator_free_instance;
}

void
cfg_block_generator_free_instance(CfgBlockGenerator *self)
{
  g_free(self->name);
}

CfgBlockGenerator *
cfg_block_generator_ref(CfgBlockGenerator *self)
{
  self->ref_cnt++;
  return self;
}

void
cfg_block_generator_unref(CfgBlockGenerator *self)
{
  if (--self->ref_cnt == 0)
    {
      if (self->free_fn)
        self->free_fn(self);
      g_free(self);
    }
}
