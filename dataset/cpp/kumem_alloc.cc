/*
 * (c) 2010 Adam Lackorzynski <adam@os.inf.tu-dresden.de>,
 *          Alexander Warg <warg@os.inf.tu-dresden.de>
 *     economic rights: Technische Universität Dresden (Germany)
 *
 * This file is part of TUD:OS and distributed under the terms of the
 * GNU General Public License 2.
 * Please see the COPYING-GPL-2 file for details.
 *
 * As a special exception, you may use this file as part of a free software
 * library without restriction.  Specifically, if other files instantiate
 * templates or use macros or inline functions from this file, or you compile
 * this file and link it with other files to produce an executable, this
 * file does not by itself cause the resulting executable to be covered by
 * the GNU General Public License.  This exception does not however
 * invalidate any other reasons why the executable file might be covered by
 * the GNU General Public License.
 */
#include <l4/re/c/util/kumem_alloc.h>
#include <l4/re/util/kumem_alloc>

L4_CV int
l4re_util_kumem_alloc(l4_addr_t *mem, unsigned pages_order,
                      l4_cap_idx_t task, l4_cap_idx_t regmgr) L4_NOTHROW
{
  L4::Cap<L4::Task> t(task);
  L4::Cap<L4Re::Rm> r(regmgr);

  return L4Re::Util::kumem_alloc(mem, pages_order, t, r);
}
