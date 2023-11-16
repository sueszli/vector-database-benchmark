// SPDX-FileCopyrightText: © 2019-2021 Alexandros Theodotou <alex@zrythm.org>
// SPDX-License-Identifier: LicenseRef-ZrythmLicense
/*
 * This file incorporates work covered by the following copyright and
 * permission notice:
 *
 * ---
 *
 * Copyright (C) 2010-2011 Dmitry Vyukov
 * Copyright (C) 2017, 2019 Robin Gareus <robin@gareus.org>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 *
 * ---
 */

#include <stdint.h>
#include <stdlib.h>

#include "utils/mpmc_queue.h"
#include "utils/objects.h"

CONST
static size_t
power_of_two_size (size_t sz)
{
  int32_t power_of_two;
  for (power_of_two = 1; 1U << power_of_two < sz; ++power_of_two)
    ;
  return 1U << power_of_two;
}

void
mpmc_queue_reserve (MPMCQueue * self, size_t buffer_size)
{
  buffer_size = power_of_two_size (buffer_size);
  g_return_if_fail (
    (buffer_size >= 2) && ((buffer_size & (buffer_size - 1)) == 0));

  if (self->buffer_mask >= buffer_size - 1)
    return;

  if (self->buffer)
    free (self->buffer);

  self->buffer = object_new_n (buffer_size, cell_t);
  self->buffer_mask = buffer_size - 1;

  mpmc_queue_clear (self);
}

MPMCQueue *
mpmc_queue_new (void)
{
  MPMCQueue * self = object_new (MPMCQueue);

  mpmc_queue_reserve (self, 8);

  return self;
}

void
mpmc_queue_free (MPMCQueue * self)
{
  free (self->buffer);

  free (self);
}

void
mpmc_queue_clear (MPMCQueue * self)
{
  for (size_t i = 0; i <= self->buffer_mask; ++i)
    {
#if MPMC_USE_STD_ATOMIC
      atomic_store_explicit (&self->buffer[i].sequence, i, memory_order_relaxed);
#else
      g_atomic_int_set (&self->buffer[i].sequence, (guint) i);
#endif
    }
#if MPMC_USE_STD_ATOMIC
  atomic_store_explicit (&self->enqueue_pos, 0, memory_order_relaxed);
  atomic_store_explicit (&self->dequeue_pos, 0, memory_order_relaxed);
#else
  g_atomic_int_set (&self->enqueue_pos, 0);
  g_atomic_int_set (&self->dequeue_pos, 0);
#endif
}

int
mpmc_queue_push_back (MPMCQueue * self, void * const data)
{
  cell_t * cell;
#if MPMC_USE_STD_ATOMIC
  unsigned int pos =
    atomic_load_explicit (&self->enqueue_pos, memory_order_relaxed);
#else
  gint pos = g_atomic_int_get (&self->enqueue_pos);
#endif
  for (;;)
    {
      cell = &self->buffer[(size_t) pos & self->buffer_mask];
#if MPMC_USE_STD_ATOMIC
      unsigned int seq =
        (guint) atomic_load_explicit (&cell->sequence, memory_order_acquire);
#else
      guint seq = (guint) g_atomic_int_get (&cell->sequence);
#endif
      intptr_t dif = (intptr_t) seq - (intptr_t) pos;
      if (dif == 0)
        {
#if MPMC_USE_STD_ATOMIC
          if (
            atomic_compare_exchange_weak_explicit (
              &self->enqueue_pos, &pos, pos + 1, memory_order_acquire,
              memory_order_acquire))
#else
          if (g_atomic_int_compare_and_exchange (
                &self->enqueue_pos, pos, (pos + 1)))
#endif
            {
              break;
            }
        }
      else if (G_UNLIKELY (dif < 0))
        {
          g_return_val_if_reached (0);
        }
      else
        {
#if MPMC_USE_STD_ATOMIC
          pos = atomic_load_explicit (&self->enqueue_pos, memory_order_relaxed);
#else
          pos = g_atomic_int_get (&self->enqueue_pos);
#endif
        }
    }
  cell->data = data;
#if MPMC_USE_STD_ATOMIC
  atomic_store_explicit (&cell->sequence, pos + 1, memory_order_release);
#else
  g_atomic_int_set (&cell->sequence, pos + 1);
#endif

  return 1;
}

int
mpmc_queue_dequeue (MPMCQueue * self, void ** data)
{
  cell_t * cell;
#if MPMC_USE_STD_ATOMIC
  unsigned int pos =
    atomic_load_explicit (&self->dequeue_pos, memory_order_relaxed);
#else
  gint pos = g_atomic_int_get (&self->dequeue_pos);
#endif
  for (;;)
    {
      cell = &self->buffer[(size_t) pos & self->buffer_mask];
#if MPMC_USE_STD_ATOMIC
      unsigned int seq =
        (guint) atomic_load_explicit (&cell->sequence, memory_order_acquire);
#else
      guint seq = (guint) g_atomic_int_get (&cell->sequence);
#endif
      intptr_t dif = (intptr_t) seq - (intptr_t) (pos + 1);
      if (dif == 0)
        {
#if MPMC_USE_STD_ATOMIC
          if (
            atomic_compare_exchange_weak_explicit (
              &self->dequeue_pos, &pos, (pos + 1), memory_order_relaxed,
              memory_order_relaxed))
#else
          if (g_atomic_int_compare_and_exchange (
                &self->dequeue_pos, pos, (pos + 1)))
#endif
            break;
        }
      else if (dif < 0)
        {
          return 0;
        }
      else
        {
#if MPMC_USE_STD_ATOMIC
          pos = atomic_load_explicit (&self->dequeue_pos, memory_order_relaxed);
#else
          pos = g_atomic_int_get (&self->dequeue_pos);
#endif
        }
    }
  *data = cell->data;
#if MPMC_USE_STD_ATOMIC
  atomic_store_explicit (
    &cell->sequence, pos + self->buffer_mask + 1, memory_order_release);
#else
  g_atomic_int_set (&cell->sequence, pos + (gint) self->buffer_mask + 1);
#endif

  return 1;
}
