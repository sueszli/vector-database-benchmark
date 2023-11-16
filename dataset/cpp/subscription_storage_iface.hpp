/*
 * SObjectizer-5
 */

/*!
 * \since
 * v.5.5.3
 *
 * \file
 * \brief An interface of subscription storage.
 */

#pragma once

#include <so_5/types.hpp>

#include <so_5/mbox.hpp>
#include <so_5/state.hpp>
#include <so_5/execution_demand.hpp>
#include <so_5/subscription_storage_fwd.hpp>

#include <functional>
#include <ostream>
#include <sstream>
#include <vector>

namespace so_5
{

namespace impl
{

//
// event_handler_data_t
//
/*!
 * \since
 * v.5.4.0
 *
 * \brief Information about event_handler and its properties.
 */
struct event_handler_data_t
	{
		//! Method for handling event.
		event_handler_method_t m_method;
		//! Is event handler thread safe or not.
		thread_safety_t m_thread_safety;
		//! Kind of this event handler.
		/*!
		 * \since
		 * v.5.7.0
		 */
		event_handler_kind_t m_kind;

		event_handler_data_t(
			event_handler_method_t method,
			thread_safety_t thread_safety,
			event_handler_kind_t kind )
			:	m_method( std::move( method ) )
			,	m_thread_safety( thread_safety )
			,	m_kind( kind )
			{}
	};

/*!
 * \since
 * v.5.5.3
 *
 * \brief Common stuff for various subscription storage implementations.
 */
namespace subscription_storage_common
{

/*!
 * \since
 * v.5.5.3
 *
 * \brief An information about one subscription.
 */
struct subscr_info_t
	{
		//! Reference to mbox.
		/*!
		 * Reference must be stored because we must have
		 * access to mbox during destroyment of all
		 * subscriptions in destructor.
		 */
		mbox_t m_mbox;
		std::type_index m_msg_type;
		//! Message sink used for subscription.
		std::reference_wrapper< abstract_message_sink_t > m_message_sink;
		const state_t * m_state;
		event_handler_data_t m_handler;

		subscr_info_t(
			mbox_t mbox,
			std::type_index msg_type,
			abstract_message_sink_t & message_sink,
			const state_t & state,
			const event_handler_method_t & method,
			thread_safety_t thread_safety,
			event_handler_kind_t handler_kind )
			:	m_mbox( std::move( mbox ) )
			,	m_msg_type( std::move( msg_type ) )
			,	m_message_sink( message_sink )
			,	m_state( &state )
			,	m_handler( method, thread_safety, handler_kind )
			{}
	};

/*!
 * \since
 * v.5.5.3
 *
 * \brief Type of vector with subscription information.
 */
using subscr_info_vector_t = std::vector< subscr_info_t >;

/*!
 * \since
 * v.5.5.3
 *
 * \brief A helper function for creating subscription description.
 */
inline std::string
make_subscription_description(
	const mbox_t & mbox_ref,
	std::type_index msg_type,
	const state_t & state )
	{
		std::ostringstream s;
		s << "(mbox:'" << mbox_ref->query_name()
			<< "', msg_type:'" << msg_type.name() << "', state:'"
			<< state.query_name() << "')";

		return s.str();
	}

} /* namespace subscription_storage_common */

/*!
 * \since
 * v.5.5.3
 *
 * \brief An interface of subscription storage
 *
 * Prior to v.5.5.3 there where only one subscription_storage implementation.
 * But this implementation was not efficient for all cases.
 *
 * Sometimes an agent has very few subscriptions and efficient 
 * implementation for that case can be based on std::vector.
 * Sometimes an agent has very many subscriptions and efficient
 * implementation can require std::map or std::unordered_map.
 *
 * The purpose of this interface is hiding details of concrete
 * subscription_storage implementation.
 */
class subscription_storage_t
	{
	public :
		subscription_storage_t();
		virtual ~subscription_storage_t() noexcept = default;

		virtual void
		create_event_subscription(
			const mbox_t & mbox,
			const std::type_index & msg_type,
			abstract_message_sink_t & message_sink,
			const state_t & target_state,
			const event_handler_method_t & method,
			thread_safety_t thread_safety,
			event_handler_kind_t handler_kind ) = 0;

		virtual void
		drop_subscription(
			const mbox_t & mbox,
			const std::type_index & msg_type,
			const state_t & target_state ) noexcept = 0;

		virtual void
		drop_subscription_for_all_states(
			const mbox_t & mbox,
			const std::type_index & msg_type ) noexcept = 0;

		/*!
		 * \brief Drop all subscriptions.
		 *
		 * All subscribed event-handlers have to be unsubscribed.
		 * The object should remain in the valid state and creation
		 * of new subscriptions have to be allowed.
		 *
		 * This method is intended to be used when an agents what
		 * to erase all subscriptions at once.
		 *
		 * \since v.5.7.3
		 */
		virtual void
		drop_all_subscriptions() noexcept = 0;

		virtual const event_handler_data_t *
		find_handler(
			mbox_id_t mbox_id,
			const std::type_index & msg_type,
			const state_t & current_state ) const noexcept = 0;

		virtual void
		debug_dump( std::ostream & to ) const = 0;

		//! Drop all content.
		/*!
		 * All information about subscription must be erased,
		 * but without real unsubscription.
		 *
		 * This method will be called after successful copy of
		 * subscription information to another storage.
		 */
		virtual void
		drop_content() noexcept = 0;

		//! Get content for copying subscription information
		//! to another storage object.
		virtual subscription_storage_common::subscr_info_vector_t
		query_content() const = 0;

		//! Setup content from information from another storage object.
		virtual void
		setup_content(
			subscription_storage_common::subscr_info_vector_t && info ) = 0;

		//! Count of subscriptions in the storage.
		virtual std::size_t
		query_subscriptions_count() const = 0;
	};

} /* namespace impl */

} /* namespace so_5 */

