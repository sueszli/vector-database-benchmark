/*
 * SObjectizer-5
 */

/*!
 * \file
 * \brief Implementation details of event_queue_hook.
 *
 * \since
 * v.5.5.24
 */

#include <so_5/event_queue_hook.hpp>

namespace so_5
{

void
event_queue_hook_t::default_deleter( event_queue_hook_t * what ) noexcept
{
	delete what;
}

void
event_queue_hook_t::noop_deleter( event_queue_hook_t * ) noexcept
{}

} /* namespace so_5 */

