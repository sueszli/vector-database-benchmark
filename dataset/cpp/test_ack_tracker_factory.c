/*
 * Copyright (c) 2020 One Identity
 * Copyright (c) 2020 Laszlo Budai <laszlo.budai@outlook.com>
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

#include <criterion/criterion.h>

#include "ack-tracker/ack_tracker_factory.h"

static void
_dummy_on_batch_acked(GList *ack_records, gpointer user_data)
{
}

Test(AckTrackerFactory, position_tracking)
{
  AckTrackerFactory *instant = instant_ack_tracker_factory_new();
  AckTrackerFactory *bookmarkless = instant_ack_tracker_bookmarkless_factory_new();
  AckTrackerFactory *consecutive = consecutive_ack_tracker_factory_new();
  AckTrackerFactory *batched = batched_ack_tracker_factory_new(0, 1, _dummy_on_batch_acked, NULL);

  cr_expect(ack_tracker_type_is_position_tracked(instant->type));
  cr_expect_not(ack_tracker_type_is_position_tracked(bookmarkless->type));
  cr_expect(ack_tracker_type_is_position_tracked(consecutive->type));
  cr_expect(ack_tracker_type_is_position_tracked(batched->type));

  ack_tracker_factory_unref(instant);
  ack_tracker_factory_unref(bookmarkless);
  ack_tracker_factory_unref(consecutive);
  ack_tracker_factory_unref(batched);
}
