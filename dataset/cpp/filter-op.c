/*
 * Copyright (c) 2002-2013 Balabit
 * Copyright (c) 1998-2013 Balázs Scheidler
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
#include "filter-op.h"

typedef struct _FilterOp
{
  FilterExprNode super;
  FilterExprNode *left, *right;
} FilterOp;

static gboolean
fop_init(FilterExprNode *s, GlobalConfig *cfg)
{
  FilterOp *self = (FilterOp *) s;

  g_assert(self->left);
  g_assert(self->right);

  if (!filter_expr_init(self->left, cfg))
    return FALSE;

  if (!filter_expr_init(self->right, cfg))
    return FALSE;

  self->super.modify = self->left->modify || self->right->modify;

  return TRUE;
}

static void
fop_free(FilterExprNode *s)
{
  FilterOp *self = (FilterOp *) s;

  filter_expr_unref(self->left);
  filter_expr_unref(self->right);
  g_free((gchar *) self->super.type);
}

FilterExprNode *
fop_clone(FilterExprNode *s)
{
  FilterOp *self = (FilterOp *) s;
  FilterOp *cloned_self = g_new0(FilterOp, 1);
  filter_expr_node_init_instance(&cloned_self->super);

  cloned_self->super.init = fop_init;
  cloned_self->super.free_fn = fop_free;
  cloned_self->super.clone = fop_clone;
  cloned_self->super.eval = self->super.eval;
  cloned_self->left = filter_expr_clone(self->left);
  cloned_self->right = filter_expr_clone(self->right);
  cloned_self->super.type = g_strdup(self->super.type);
  return &cloned_self->super;
}

static void
fop_init_instance(FilterOp *self)
{
  filter_expr_node_init_instance(&self->super);
  self->super.init = fop_init;
  self->super.free_fn = fop_free;
  self->super.clone = fop_clone;
}

static gboolean
fop_or_eval(FilterExprNode *s, LogMessage **msgs, gint num_msg, LogTemplateEvalOptions *options)
{
  FilterOp *self = (FilterOp *) s;

  return (filter_expr_eval_with_context(self->left, msgs, num_msg, options)
          || filter_expr_eval_with_context(self->right, msgs, num_msg, options)) ^ s->comp;
}

FilterExprNode *
fop_or_new(FilterExprNode *e1, FilterExprNode *e2)
{
  FilterOp *self = g_new0(FilterOp, 1);

  fop_init_instance(self);
  self->super.eval = fop_or_eval;
  self->left = e1;
  self->right = e2;
  self->super.type = g_strdup("OR");
  return &self->super;
}

static gboolean
fop_and_eval(FilterExprNode *s, LogMessage **msgs, gint num_msg, LogTemplateEvalOptions *options)
{
  FilterOp *self = (FilterOp *) s;

  return (filter_expr_eval_with_context(self->left, msgs, num_msg, options)
          && filter_expr_eval_with_context(self->right, msgs, num_msg, options)) ^ s->comp;
}

FilterExprNode *
fop_and_new(FilterExprNode *e1, FilterExprNode *e2)
{
  FilterOp *self = g_new0(FilterOp, 1);

  fop_init_instance(self);
  self->super.eval = fop_and_eval;
  self->left = e1;
  self->right = e2;
  self->super.type = g_strdup("AND");
  return &self->super;
}
