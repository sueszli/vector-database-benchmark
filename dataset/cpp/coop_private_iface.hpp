/*
	SObjectizer 5.
*/

/*!
 * \file
 * \brief Private interface for a coop.
 *
 * \since
 * v.5.6.0
 */

#pragma once

#include <so_5/coop.hpp>

namespace so_5
{

namespace impl
{

//
// coop_private_iface_t
//
/*!
 * \since
 * v.5.2.3
 *
 * \brief A special class for accessing private members of agent_coop.
 */
class coop_private_iface_t
{
	public :
		static coop_unique_holder_t
		make_coop(
			coop_id_t id,
			coop_handle_t parent,
			disp_binder_shptr_t default_binder,
			outliving_reference_t< environment_t > env )
			{
				return {
						coop_shptr_t {
								new coop_t{
									id,
									std::move(parent),
									std::move(default_binder),
									env
								}
						}
				};
			}

		static coop_shptr_t
		make_from( coop_unique_holder_t holder ) noexcept
			{
				return holder.release();
			}

		static void
		increment_usage_count( coop_t & coop ) noexcept
			{
				coop.increment_usage_count();
			}

		static void
		decrement_usage_count( coop_t & coop )
			{
				coop.decrement_usage_count();
			}

		static void
		destroy_content( coop_t & coop )
			{
				coop_impl_t::destroy_content( coop );
			}

		static void
		do_registration_specific_actions( coop_t & coop )
			{
				coop_impl_t::do_registration_specific_actions( coop );
			}

		static void
		do_final_deregistration_actions( coop_t & coop )
			{
				coop_impl_t::do_final_deregistration_actions( coop );
			}

		static coop_reg_notificators_container_ref_t
		giveout_reg_notificators( coop_t & coop ) noexcept
			{
				return std::move(coop.m_reg_notificators);
			}

		static coop_dereg_notificators_container_ref_t
		giveout_dereg_notificators( coop_t & coop ) noexcept
			{
				return std::move(coop.m_dereg_notificators);
			}

		static coop_dereg_reason_t
		dereg_reason( const coop_t & coop ) noexcept
			{
				return coop.m_dereg_reason;
			}

		/*!
		 * \brief Set the next coop in the chain of coops for the final
		 * deregistration.
		 *
		 * It's expected that \a coop is the last item in this chain at
		 * the moment. After completion the \a next_in_chain will be the
		 * last item in the chain.
		 *
		 * \since v.5.8.0
		 */
		static void
		set_next_in_final_dereg_chain(
			coop_t & coop,
			coop_shptr_t next_in_chain ) noexcept
		{
			coop.m_next_in_final_dereg_chain = std::move(next_in_chain);
		}

		/*!
		 * \brief Extract a smart pointer to the next coop in the chain of
		 * coops for the final deregistration.
		 *
		 * It may return empty coop_shptr_t if \a coop is the last item in
		 * this chain.
		 *
		 * \since v.5.8.0
		 */
		[[nodiscard]]
		static coop_shptr_t
		giveout_next_in_final_dereg_chain(
			coop_t & coop ) noexcept
		{
			return std::exchange( coop.m_next_in_final_dereg_chain, coop_shptr_t{} );
		}
};

} /* namespace impl */

} /* namespace so_5 */

