/*
	SObjectizer 5.
*/

#include <so_5/coop.hpp>

#include <so_5/impl/internal_env_iface.hpp>
#include <so_5/impl/internal_agent_iface.hpp>
#include <so_5/impl/agent_ptr_compare.hpp>

#include <so_5/details/rollback_on_exception.hpp>

#include <so_5/exception.hpp>
#include <so_5/environment.hpp>

#include <exception>
#include <algorithm>

namespace so_5
{

//
// coop_reg_notificators_container_t
//
void
coop_reg_notificators_container_t::call_all(
	environment_t & env,
	const coop_handle_t & coop ) const noexcept
	{
		for( auto & n : m_notificators )
			n( env, coop );
	}

//
// coop_dereg_notificators_container_t
//
void
coop_dereg_notificators_container_t::call_all(
	environment_t & env,
	const coop_handle_t & coop,
	const coop_dereg_reason_t & reason ) const noexcept
	{
		for( auto & n : m_notificators )
			n( env, coop, reason );
	}

namespace impl
{

//
// coop_impl_t
//
void
coop_impl_t::destroy_content(
	coop_t & coop ) noexcept
	{
		// Initiate deleting of agents by hand to guarantee that
		// agents will be destroyed before return from coop_t
		// destructor.
		//
		// NOTE: because agents are stored here by smart references
		// for some agents this operation will lead only to reference
		// counter descrement. Not to deletion of agent.
		coop.m_agent_array.clear();

		// Don't expect exceptions here because all resource deleters have
		// to be noexcept.
		for( auto & d : coop.m_resource_deleters )
		{
			static_assert( noexcept( d() ),
					"resource deleter is expected to be noexcept" );

			d();
		}
		coop.m_resource_deleters.clear();
	}

void
coop_impl_t::do_add_agent(
	coop_t & coop,
	agent_ref_t agent_ref )
	{
		internal_agent_iface_t agent_iface{ *agent_ref };
		so_5::details::do_with_rollback_on_exception(
				[&]() {
					agent_iface.set_disp_binder( coop.m_coop_disp_binder );
					coop.m_agent_array.emplace_back( std::move(agent_ref) );
				},
				[&agent_iface]() noexcept {
					agent_iface.drop_disp_binder();
				} );
	}

void
coop_impl_t::do_add_agent(
	coop_t & coop,
	agent_ref_t agent_ref,
	disp_binder_shptr_t disp_binder )
	{
		internal_agent_iface_t agent_iface{ *agent_ref };
		so_5::details::do_with_rollback_on_exception(
				[&]() {
					agent_iface.set_disp_binder( std::move(disp_binder) );
					coop.m_agent_array.emplace_back( std::move(agent_ref) );
				},
				[&agent_iface]() noexcept {
					agent_iface.drop_disp_binder();
				} );
	}

namespace
{
	/*!
	 * \since
	 * v.5.2.3
	 *
	 * \brief Helper function for notificator addition.
	 */
	template< class C, class N >
	inline void
	do_add_notificator_to(
		intrusive_ptr_t< C > & to,
		N notificator )
	{
		if( !to )
		{
			to = intrusive_ptr_t< C >( new C() );
		}

		to->add( std::move(notificator) );
	}

} /* namespace anonymous */

void
coop_impl_t::add_reg_notificator(
	coop_t & coop,
	coop_reg_notificator_t notificator )
	{
		do_add_notificator_to(
				coop.m_reg_notificators,
				std::move(notificator) );
	}

void
coop_impl_t::add_dereg_notificator(
	coop_t & coop,
	coop_dereg_notificator_t notificator )
	{
		do_add_notificator_to(
				coop.m_dereg_notificators,
				std::move(notificator) );
	}

[[nodiscard]]
exception_reaction_t
coop_impl_t::exception_reaction(
	const coop_t & coop ) noexcept
	{
		if( inherit_exception_reaction == coop.m_exception_reaction )
			{
				const auto parent = so_5::low_level_api::to_shptr_noexcept(
						coop.m_parent );
				if( parent )
					return parent->exception_reaction();
				else
					return coop.environment().exception_reaction();
			}

		return coop.m_exception_reaction;
	}

void
coop_impl_t::do_decrement_reference_count(
	coop_t & coop ) noexcept
	{
		// If it is the last working agent then Environment should be
		// informed that the cooperation is ready to be deregistered.
		if( 0 == --coop.m_reference_count )
			{
				// NOTE: usage counter incremented and decremented during
				// registration process even if registration of cooperation failed.
				// So decrement_usage_count() could be called when cooperation
				// has coop_not_registered status.
				//
				// It is possible that reference counter become 0 several times.
				// For example when a child coop is being registered while
				// the parent coop is in deregistration process.
				// Because of that it is necessary to check the current status
				// of the coop.
				// 
				// If the coop should be deregistered finally its status should
				// be changed to deregistration_in_final_stage.
				const auto should_finalize = [&] {
					std::lock_guard< std::mutex > lock{ coop.m_lock };

					using status_t = coop_t::registration_status_t;
					if( status_t::coop_registered == coop.m_registration_status ||
						status_t::coop_deregistering == coop.m_registration_status )
						{
							coop.m_registration_status =
									status_t::deregistration_in_final_stage;
							return true;
						}
					else
						return false;
				};

				if( should_finalize() )
					{
						impl::internal_env_iface_t{ coop.m_env.get() }
								.ready_to_deregister_notify( coop.shared_from_this() );
					}
			}
	}

class coop_impl_t::registration_performer_t
	{
		coop_t & m_coop;

		void
		perform_actions_without_rollback_on_exception()
			{
				reorder_agents_with_respect_to_priorities();
				bind_agents_to_coop();
				preallocate_disp_resources();
			}

		void
		perform_actions_with_rollback_on_exception()
			{
				so_5::details::do_with_rollback_on_exception( [this] {
						define_all_agents();

						// Coop's lock should be acquired before notification
						// of the parent coop.
						std::lock_guard< std::mutex > lock{ m_coop.m_lock };
						make_relation_with_parent_coop();

						// These actions shouldn't throw.
						details::invoke_noexcept_code( [&] {
							// This operation shouldn't throw because dispatchers
							// allocated resources for agents.
							//
							// But it is possible that an exception will be throw
							// during an attempt to send evt_start message to agents.
							// In that case it is simpler to call std::terminate().
							bind_agents_to_disp();

							// Cooperation should assume that it is registered now.
							m_coop.m_registration_status =
									coop_t::registration_status_t::coop_registered;

							// Increment reference count to reflect that cooperation
							// is registered. This is necessary in v.5.5.12 to prevent
							// automatic deregistration of the cooperation right after
							// finish of registration process for empty cooperation.
							m_coop.increment_usage_count();
						} );
					},
					[this] {
						// NOTE: we use the fact that actual binding of agents to
						// dispatchers can't throw. It means that exception was thrown
						// at earlier stages (in define_all_agents() or
						// make_relation_with_parent_coop()).
						deallocate_disp_resources();
					} );
			}

		void
		reorder_agents_with_respect_to_priorities() noexcept
			{
				std::sort(
						std::begin(m_coop.m_agent_array),
						std::end(m_coop.m_agent_array),
						[]( const auto & a, const auto & b ) noexcept {
							return special_agent_ptr_compare( *a, *b );
						} );
			}

		void
		bind_agents_to_coop()
			{
				for( const auto & agent_ref : m_coop.m_agent_array )
				{
					internal_agent_iface_t{ *agent_ref }.bind_to_coop( m_coop );
				}
			}

		void
		preallocate_disp_resources()
			{
				// In case of an exception we should undo preallocation only for
				// those agents for which preallocation was successful.
				coop_t::agent_array_t::iterator it;
				try
					{
						for( it = m_coop.m_agent_array.begin();
								it != m_coop.m_agent_array.end();
								++it )
							{
								agent_t & agent = **it;
								internal_agent_iface_t agent_iface{ agent };
								agent_iface.query_disp_binder()
										.preallocate_resources( agent );
							}
					}
				catch( const std::exception & x )
					{
						// All preallocated resources should be returned back.
						for( auto it2 = m_coop.m_agent_array.begin();
								it2 != it;
								++it2 )
							{
								agent_t & agent = **it2;
								internal_agent_iface_t agent_iface{ agent };
								agent_iface.query_disp_binder()
										.undo_preallocation( agent );
							}

						SO_5_THROW_EXCEPTION(
								rc_agent_to_disp_binding_failed,
								std::string{
										"an exception during the first stage of "
										"binding agent to the dispatcher, exception: " }
								+ x.what() );
					}
			}

		void
		define_all_agents()
			{
				try
					{
						for( const auto & agent_ref : m_coop.m_agent_array )
							internal_agent_iface_t{ *agent_ref }
									.initiate_agent_definition();
					}
				catch( const exception_t & )
					{
						throw;
					}
				catch( const std::exception & ex )
					{
						SO_5_THROW_EXCEPTION(
							rc_coop_define_agent_failed,
							ex.what() );
					}
				catch( ... )
					{
						SO_5_THROW_EXCEPTION(
							rc_coop_define_agent_failed,
							"exception of unknown type has been thrown in "
							"so_define_agent()" );
					}
			}

		void
		make_relation_with_parent_coop()
			{
				so_5::low_level_api::to_shptr(m_coop.m_parent)->add_child(
						m_coop.shared_from_this() );
			}

		void
		bind_agents_to_disp() noexcept
			{
				for( const auto & agent_ref : m_coop.m_agent_array )
					{
						internal_agent_iface_t agent_iface{ *agent_ref };
						agent_iface.query_disp_binder().bind( *agent_ref );
					}
			}

		void
		deallocate_disp_resources() noexcept
			{
				for( const auto & agent_ref : m_coop.m_agent_array )
					{
						internal_agent_iface_t agent_iface{ *agent_ref };
						agent_iface.query_disp_binder()
								.undo_preallocation( *agent_ref );
					}
			}

	public :
		explicit registration_performer_t( coop_t & coop ) noexcept
			:	m_coop{ coop }
			{}

		void
		perform()
			{
				// On first phase we perform actions that don't require
				// any rollback on exception.
				perform_actions_without_rollback_on_exception();

				// Then we should perform some actions that require some
				// rollback in the case of an exception.
				perform_actions_with_rollback_on_exception();
			}
	};

void
coop_impl_t::do_registration_specific_actions( coop_t & coop )
	{
		registration_performer_t{ coop }.perform();
	}

//
// deregistration_performer_t
//
//! A helper for coop's deregistration procedure.
class coop_impl_t::deregistration_performer_t
	{
		coop_t & m_coop;
		const coop_dereg_reason_t m_reason;

		enum class phase1_result_t
			{
				dereg_initiated,
				dereg_already_in_progress
			};

		phase1_result_t
		perform_phase1() noexcept
			{
				// The first phase should be performed on locked object.
				std::lock_guard< std::mutex > lock{ m_coop.m_lock };

				if( coop_t::registration_status_t::coop_registered !=
						m_coop.m_registration_status )
					// Deregistration is already in progress.
					// Nothing to do.
					return phase1_result_t::dereg_already_in_progress;

				// Deregistration process should be started.
				m_coop.m_registration_status =
						coop_t::registration_status_t::coop_deregistering;
				m_coop.m_dereg_reason = m_reason;

				initiate_deregistration_for_children();

				return phase1_result_t::dereg_initiated;
			}

		void
		shutdown_all_agents() noexcept
			{
				for( const auto & agent_ref : m_coop.m_agent_array )
					{
						internal_agent_iface_t{ *agent_ref }.shutdown_agent();
					}
			}

		void
		initiate_deregistration_for_children() noexcept
			{
				m_coop.for_each_child( []( coop_t & coop ) {
						coop.deregister( dereg_reason::parent_deregistration );
					} );
			}

	public :
		deregistration_performer_t(
			coop_t & coop,
			coop_dereg_reason_t reason ) noexcept
			:	m_coop{ coop }
			,	m_reason{ reason }
			{}

		void
		perform() noexcept
			{
				auto result = perform_phase1();

				if( phase1_result_t::dereg_initiated == result )
					{
						// Deregistration is initiated the first time.

						// All agents should be shut down.
						shutdown_all_agents();

						// Reference count to this coop can be decremented.
						// If there is no more uses of that coop then the coop
						// will be deregistered completely.
						m_coop.decrement_usage_count();
					}
			}
	};

void
coop_impl_t::do_deregistration_specific_actions(
	coop_t & coop,
	coop_dereg_reason_t reason ) noexcept
	{
		deregistration_performer_t{ coop, reason }.perform();
	}

void
coop_impl_t::do_final_deregistration_actions(
	coop_t & coop )
	{
		// Agents should be unbound from their dispatchers.
		for( const auto & agent_ref : coop.m_agent_array )
			{
				agent_t & agent = *agent_ref;
				internal_agent_iface_t agent_iface{ agent };
				agent_iface.query_disp_binder().unbind( agent );
			}

		// Now the coop can be removed from it's parent.
		// We don't except an exception here because m_parent should
		// contain an actual value.
		// But if not then we have a serious problem and it is better
		// to terminate the application.
		so_5::low_level_api::to_shptr(coop.m_parent)->remove_child( coop );
	}

void
coop_impl_t::do_add_child(
	coop_t & parent,
	coop_shptr_t child )
	{
		// Count of users on this coop is incremented.
		parent.increment_usage_count();

		// If an exception is throw below then usage count for the parent
		// coop should be decremented.
		so_5::details::do_with_rollback_on_exception( [&] {
				// Modification of parent-child relationship must be performed
				// on locked object.
				std::lock_guard< std::mutex > lock{ parent.m_lock };

				// A new coop can't be added as a child if coop is being
				// deregistered.
				if( coop_t::registration_status_t::coop_registered !=
						parent.m_registration_status )
					SO_5_THROW_EXCEPTION(
							rc_coop_is_not_in_registered_state,
							"add_child() can be processed only when coop "
							"is registered" );

				// New child will be inserted to the head of children list.
				if( parent.m_first_child )
					parent.m_first_child->m_prev_sibling = child;

				child->m_next_sibling = std::move(parent.m_first_child);

				parent.m_first_child = std::move(child);
			},
			[&parent] {
				// Something went wrong. Count of references should be
				// returned back.
				parent.decrement_usage_count();
			} );
	}

void
coop_impl_t::do_remove_child(
	coop_t & parent,
	coop_t & child ) noexcept
	{
		{
			// Modification of parent-child relationship must be performed
			// on locked object.
			std::lock_guard< std::mutex > lock{ parent.m_lock };

			if( parent.m_first_child.get() == &child )
			{
				// Child was a head of children chain. There is no prev-sibling
				// for the child to be removed.
				parent.m_first_child = child.m_next_sibling;
				if( parent.m_first_child )
					parent.m_first_child->m_prev_sibling.reset();
			}
			else
			{
				child.m_prev_sibling->m_next_sibling = child.m_next_sibling;
				if( child.m_next_sibling )
					child.m_next_sibling->m_prev_sibling = child.m_prev_sibling;
			}
		}

		// Count of references to the parent coop can be decremented now.
		parent.decrement_usage_count();
	}

} /* namespace impl */

} /* namespace so_5 */

