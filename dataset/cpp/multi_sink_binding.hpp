/*
 * SObjectizer-5
 */

/*!
 * \file
 * \brief Stuff for multi_sink_binding implementation.
 *
 * \since v.5.8.0
 */

#pragma once

#include <so_5/single_sink_binding.hpp>

#include <so_5/exception.hpp>
#include <so_5/ret_code.hpp>

#include <so_5/details/sync_helpers.hpp>
#include <so_5/details/rollback_on_exception.hpp>

#include <map>
#include <tuple>

namespace so_5
{

namespace multi_sink_binding_impl
{

/*!
 * \brief Type of container for bindings for messages.
 *
 * There could be just one binding for one message type.
 *
 * \since v.5.8.0
 */
using one_sink_bindings_t = std::map< std::type_index, single_sink_binding_t >;

/*!
 * \brief Type of container for bindings for one msink.
 *
 * There could be bindings for several message types for the same msink.
 *
 * \since v.5.8.0
 */
using one_mbox_bindings_t = std::map<
		msink_t,
		one_sink_bindings_t,
		so_5::impl::msink_less_comparator_t >;

/*!
 * \brief Type of container for bindings for messages from mboxes.
 *
 * Several msinks can be bound to one mbox.
 *
 * \since v.5.8.0
 */
using bindings_map_t = std::map< mbox_id_t, one_mbox_bindings_t >;

/*!
 * \brief Helper type for insertion into a std::map with automatic rollback.
 *
 * It's expected that \a Container is a std::map. It's expected that
 * Container::mapped_type is DefaultConstructible.
 *
 * If commit() isn't called and a new item was inserted to the map
 * in the constructor then that item will be erased in the destructor.
 *
 * See actual_binding_handler_t::do_actual_bind() for usage examples.
 *
 * \since v.5.8.0
 */
template< class Container >
class insertion_it_with_auto_erase_if_not_committed_t final
	{
		using iterator_type = typename Container::iterator;

		Container & m_container;
		iterator_type m_it;
		bool m_modified;
		bool m_commited{ false };

	public:
		/*!
		 * \brief Initializing constructor.
		 *
		 * Tries to find an item with \a k in \a container.
		 * If item isn't found then a new item with \a k as key will be inserted
		 * into \a container.
		 */
		insertion_it_with_auto_erase_if_not_committed_t(
			Container & container,
			typename Container::key_type const & k )
			:	m_container{ container }
			,	m_it{ container.find( k ) }
			{
				if( container.end() == m_it )
					{
						m_it = m_container.emplace( k, typename Container::mapped_type{} ).first;
						m_modified = true;
					}
				else
					m_modified = false;
			}

		insertion_it_with_auto_erase_if_not_committed_t(
			const insertion_it_with_auto_erase_if_not_committed_t & ) = delete;
		insertion_it_with_auto_erase_if_not_committed_t(
			insertion_it_with_auto_erase_if_not_committed_t && ) = delete;

		~insertion_it_with_auto_erase_if_not_committed_t() noexcept
			{
				if( m_modified && !m_commited )
					m_container.erase( m_it );
			}

		void
		commit() noexcept
			{
				m_commited = true;
			}

		[[nodiscard]]
		iterator_type
		operator->() const
			{
				return m_it;
			}

		/*!
		 * \retval true if new item was inserted in the constructor.
		 * \retval false if item was found in the container.
		 */
		[[nodiscard]]
		bool
		modified() const noexcept
			{
				return m_modified;
			}
	};

// Deduction guide for insertion_it_with_auto_erase_if_not_committed_t.
template<typename C, typename K>
insertion_it_with_auto_erase_if_not_committed_t(C &, const K&) ->
		insertion_it_with_auto_erase_if_not_committed_t<C>;

/*!
 * \brief Class that actually holds multiple sinks bindings.
 *
 * \attention
 * It's not a thread-safe class. Thread-safety has to be provided by
 * an owner of actual_binding_handler_t instance.
 *
 * \since v.5.8.0
 */
class actual_binding_handler_t
	{
		//! All bindings.
		bindings_map_t m_bindings;

		/*!
		 * \brief Implementation of bind procedure.
		 *
		 * \tparam Single_Sink_Modificator Type of functor that will be called
		 * for modification of the corresponding single_sink_binding_t instance.
		 * This functor should have a prototype like:
		 * \code
		 * void(const std::type_index & msg_type, single_sink_binding_t & binding);
		 * \endcode
		 */
		template< typename Single_Sink_Modificator >
		void
		do_actual_bind(
			const std::type_index & msg_type,
			const mbox_t & from,
			const msink_t & dest,
			Single_Sink_Modificator && single_sink_modificator )
			{
				insertion_it_with_auto_erase_if_not_committed_t it_mbox{ m_bindings, from->id() };

				insertion_it_with_auto_erase_if_not_committed_t it_msink{ it_mbox->second, dest };

				insertion_it_with_auto_erase_if_not_committed_t it_msg{ it_msink->second, msg_type };
				// If new item wasn't inserted then it's an error.
				if( !it_msg.modified() )
					{
						SO_5_THROW_EXCEPTION(
								rc_evt_handler_already_provided,
								std::string{ "msink already subscribed to a message" } +
								"(mbox:'" + from->query_name() +
								"', msg_type:'" + msg_type.name() + "'" );
					}

				single_sink_modificator( msg_type, it_msg->second );

				it_msg.commit();
				it_msink.commit();
				it_mbox.commit();
			}

	public:
		/*!
		 * This method can be used for messages and signals.
		 */
		template< typename Msg >
		void
		do_bind(
			const mbox_t & from,
			const msink_t & dest )
			{
				do_actual_bind(
						message_payload_type< Msg >::subscription_type_index(),
						from,
						dest,
						[&](
							const std::type_index & msg_type,
							single_sink_binding_t & binding )
						{
							binding.bind_for_msg_type(
									msg_type,
									from,
									dest );
						} );
			}

		/*!
		 * This method can only be used if \a Msg isn't a signal.
		 *
		 * It's expected that \a delivery_filter isn't nullptr.
		 */
		template< typename Msg >
		void
		do_bind(
			const mbox_t & from,
			const msink_t & dest,
			delivery_filter_unique_ptr_t delivery_filter )
			{
				// Msg can't be a signal!
				ensure_not_signal< Msg >();

				do_actual_bind(
						message_payload_type< Msg >::subscription_type_index(),
						from,
						dest,
						[&](
							const std::type_index & msg_type,
							single_sink_binding_t & binding )
						{
							binding.bind_for_msg_type(
									msg_type,
									from,
									dest,
									std::move(delivery_filter) );
						} );
			}

		/*!
		 * This method can be used for messages and signals.
		 */
		template< typename Msg >
		void
		do_unbind(
			const mbox_t & from,
			const msink_t & dest ) noexcept
			{
				auto it_mbox = m_bindings.find( from->id() );
				if( it_mbox == m_bindings.end() )
					return;

				auto & msinks = it_mbox->second;
				auto it_msink = msinks.find( dest );
				if( it_msink == msinks.end() )
					return;

				const auto & msg_type =
						message_payload_type< Msg >::subscription_type_index();

				auto & msgs = it_msink->second;
				msgs.erase( msg_type );

				if( msgs.empty() )
					{
						msinks.erase( it_msink );
						if( msinks.empty() )
							{
								m_bindings.erase( it_mbox );
							}
					}
			}

		/*!
		 * Remove binding for all types of messages/signals that were
		 * made for \a dest from \a from.
		 */
		void
		do_unbind_all_for(
			const mbox_t & from,
			const msink_t & dest ) noexcept
			{
				auto it_mbox = m_bindings.find( from->id() );
				if( it_mbox == m_bindings.end() )
					return;

				auto & msinks = it_mbox->second;
				msinks.erase( dest );
				if( msinks.empty() )
					m_bindings.erase( it_mbox );
			}

		/*!
		 * Removes all bindings.
		 */
		void
		do_clear() noexcept
			{
				m_bindings.clear();
			}
	};

} /* namespace multi_sink_binding_impl */

/*!
 * \brief Helper class for managing multiple sink bindings.
 *
 * An instance of multi_sink_binding_t drops all binding in the destructor.
 * If it's necessary to drop all binding manually then clear() method
 * can be used.
 *
 * Usage examples:
 * \code
 * // Use as a part of an agent.
 * class coordinator final : public so_5::agent_t
 * {
 * 	const so_5::mbox_t broadcasting_mbox_;
 * 	so_5::multi_sink_binding_t<> bindings_;
 * ...
 * 	void on_some_event(mhood_t<msg_some_command> cmd) {
 * 		// Create a child coop and bind agents to broadcasting mbox.
 * 		so_5::introduce_child_coop(*this, [](so_5::coop_t & coop) {
 * 				auto * first = coop.make_agent<first_worker>(...);
 * 				auto first_worker_msink = so_5::wrap_to_msink(first->so_direct_mbox());
 *
 * 				auto * second = coop.make_agent<second_worker>(...);
 * 				auto second_worker_msink = so_5::wrap_to_msink(second->so_direct_mbox());
 *
 * 				bindings_.bind<msg_some_data>(broadcasting_mbox_, first_worker_msink);
 * 				bindings_.bind<msg_some_data>(broadcasting_mbox_, second_worker_msink);
 * 				bindings_.bind<msg_some_notify>(broadcasting_mbox_, first_worker_msink);
 * 				bindings_.bind<msg_another_notify>(broadcasting_mbox_, second_worker_msink);
 * 				...
 * 			});
 * 	}
 * };
 *
 * // Use as object controlled by a coop.
 * so_5::environment_t & env = ...;
 * env.introduce_coop([](so_5::coop_t & coop) {
 * 		const auto broadcasting_mbox = coop.environment().create_mbox();
 * 		auto * bindings = coop.take_under_control(
 * 			std::make_unique<so_5::multi_sink_binding_t<>>());
 *
 * 		auto * first = coop.make_agent<first_worker>(...);
 * 		auto first_worker_msink = so_5::wrap_to_msink(first->so_direct_mbox());
 *
 * 		auto * second = coop.make_agent<second_worker>(...);
 * 		auto second_worker_msink = so_5::wrap_to_msink(second->so_direct_mbox());
 *
 * 		bindings.bind<msg_some_data>(broadcasting_mbox, first_worker_msink);
 * 		bindings.bind<msg_some_data>(broadcasting_mbox, second_worker_msink);
 * 		bindings.bind<msg_some_notify>(broadcasting_mbox, first_worker_msink);
 * 		bindings.bind<msg_another_notify>(broadcasting_mbox, second_worker_msink);
 * 		...
 * 	});
 * \endcode
 *
 * The instance of multi_sink_binding_t is thread safe by the default (if
 * the default value is used for template parameter \a Lock_Type).
 * In single-threaded environments this can be unnecessary, so_5::null_mutex_t
 * can be used in those cases:
 * \code
 * // It's assumed that code works in single-threaded environment.
 * so_5::environment_t & env = ...;
 * env.introduce_coop([](so_5::coop_t & coop) {
 * 		const auto broadcasting_mbox = coop.environment().create_mbox();
 * 		auto * bindings = coop.take_under_control(
 * 			std::make_unique<so_5::multi_sink_binding_t<so_5::null_mutex_t>>());
 * \endcode
 *
 * \note
 * This class isn't Copyable, not Moveable. Once created an instance of
 * that type can't be copied or moved.
 *
 * \tparam Lock_Type Type to be used for thread-safety.
 * It should be a std::mutex-like class. If thread-safety isn't needed
 * then so_5::null_mutex_t can be used.
 *
 * \since v.5.8.0
 */
template< typename Lock_Type = std::mutex >
class multi_sink_binding_t
	:	protected so_5::details::lock_holder_detector< Lock_Type >::type
	,	protected multi_sink_binding_impl::actual_binding_handler_t
	{
	public:
		multi_sink_binding_t() = default;

		multi_sink_binding_t( const multi_sink_binding_t & ) = delete;
		multi_sink_binding_t &
		operator=( const multi_sink_binding_t & ) = delete;

		multi_sink_binding_t( multi_sink_binding_t && ) = delete;
		multi_sink_binding_t &
		operator=( multi_sink_binding_t && ) = delete;

		/*!
		 * Create a binding for message/signal of type \a Msg from mbox \a from
		 * to the destination \a dest.
		 *
		 * This binding won't use a delivery filter.
		 *
		 * An exception will be thrown if such a binding already exists.
		 *
		 * Usage example:
		 * \code
		 * const so_5::mbox_t & source = ...;
		 * const so_5::msink_t & dest = ...;
		 * auto binding = std::make_unique< so_5::multi_sink_binding_t<> >();
		 *
		 * binding->bind<my_message>(source, dest);
		 * binding->bind<my_signal>(source, dest);
		 * \endcode
		 *
		 * It it's required to make a binding for a mutable message then
		 * so_5::mutable_msg marker has to be used:
		 * \code
		 * const so_5::mbox_t & source = ...;
		 * const so_5::msink_t & dest = ...;
		 * auto binding = std::make_unique< so_5::multi_sink_binding_t<> >();
		 *
		 * binding->bind< so_5::mutable_msg<my_message> >(source, dest);
		 * \endcode
		 */
		template< typename Msg >
		void
		bind(
			//! The source mbox.
			const mbox_t & from,
			//! The destination for messages.
			const msink_t & dest )
			{
				this->lock_and_perform( [&]() {
						this->template do_bind< Msg >(
								from,
								dest );
					} );
			}

		/*!
		 * Create a binding for message of type \a Msg from mbox \a from
		 * to the destination \a dest.
		 *
		 * This binding should use delivery filter \a delivery_filter.
		 *
		 * An exception will be thrown if such a binding already exists.
		 *
		 * \note
		 * This method can't be used for binding signals.
		 */
		template< typename Msg >
		void
		bind(
			//! The source mbox.
			const mbox_t & from,
			//! The destination for messages.
			const msink_t & dest,
			//! Delivery filter to be used. It shouldn't be nullptr.
			delivery_filter_unique_ptr_t delivery_filter )
			{
				so_5::low_level_api::ensure_not_null( delivery_filter );

				this->lock_and_perform( [&]() {
						this->template do_bind< Msg >(
								from,
								dest,
								std::move(delivery_filter) );
					} );
			}

		/*!
		 * Create a binding for message of type \a Msg from mbox \a from
		 * to the destination \a dest.
		 *
		 * The lambda (or functor) \a filter will be used as delivery filter
		 * for messages.
		 *
		 * An exception will be thrown if such a binding already exists.
		 *
		 * \note
		 * This method can't be used for binding signals.
		 *
		 * Usage example:
		 * \code
		 * const so_5::mbox_t & source = ...;
		 * const so_5::msink_t & dest = ...;
		 * auto binding = std::make_unique< so_5::multi_sink_binding_t<> >();
		 *
		 * binding->bind<my_message>(source, dest,
		 * 	[](const my_message & msg) {
		 * 		... // should return `true` or `false`.
		 * 	});
		 * \endcode
		 *
		 * It it's required to make a binding for a mutable message then
		 * so_5::mutable_msg marker has to be used, but note the type of
		 * delivery filter argument:
		 * \code
		 * const so_5::mbox_t & source = ...;
		 * const so_5::msink_t & dest = ...;
		 * auto binding = std::make_unique< so_5::multi_sink_binding_t<> >();
		 *
		 * binding->bind< so_5::mutable_msg<my_message> >(source, dest,
		 * 	[](const my_message & msg) {
		 * 		... // should return `true` or `false`.
		 * 	});
		 * \endcode
		 */
		template< typename Msg, typename Lambda >
		void
		bind(
			//! The source mbox.
			const mbox_t & from,
			//! The destination for messages.
			const msink_t & dest,
			//! Delivery filter to be used.
			Lambda && filter )
			{
				using lambda_type = std::remove_reference_t< Lambda >;

				using detectable_arg_type =
						sink_bindings_details::lambda_with_detectable_arg_type_t< lambda_type >;

				delivery_filter_unique_ptr_t filter_holder;

				if constexpr( detectable_arg_type::value )
					{
						// Type of filter lambda can be checked by a static_assert.
						using argument_type = typename detectable_arg_type::argument_t;

						// Try to check delivery filter lambda argument type
						// at the compile time.
						sink_bindings_details::ensure_valid_argument_for_delivery_filter<
								typename so_5::message_payload_type<Msg>::payload_type,
								argument_type
							>();

						filter_holder.reset(
								new low_level_api::lambda_as_filter_t< lambda_type, argument_type >(
										std::move(filter) )
							);

					}
				else
					{
						// Assume that filter lambda is in form:
						//
						// [](const auto & msg) -> bool {...}
						//
						// so we don't know the type of the argument.

						using argument_type = typename message_payload_type< Msg >::payload_type;

						filter_holder.reset(
								new low_level_api::lambda_as_filter_t< lambda_type, argument_type >(
										std::move(filter) )
							);
					}

				this->bind< Msg >( from, dest, std::move(filter_holder) );
			}

		/*!
		 * Remove binding for message/signal of type \a Msg from mbox \a from
		 * to the destination \a dest.
		 *
		 * It is safe to call this method if such a binding doesn't exist.
		 *
		 * Usage example:
		 * \code
		 * const so_5::mbox_t & source = ...;
		 * const so_5::msink_t & dest = ...;
		 * so_5::multi_sink_binding_t & bindings = ...;
		 *
		 * binding->unbind<my_message>(source, dest);
		 * binding->unbind<my_signal>(source, dest);
		 * \endcode
		 */
		template< typename Msg >
		void
		unbind(
			//! The source mbox.
			const mbox_t & from,
			//! The destination for messages.
			const msink_t & dest ) noexcept
			{
				this->lock_and_perform( [&]() {
						this->template do_unbind< Msg >( from, dest );
					} );
			}

		/*!
		 * Remove binding for all message/signal types from mbox \a from
		 * to the destination \a dest.
		 *
		 * It is safe to call this method if there is no any binding for
		 * (\a form, \a dest) pair.
		 *
		 * Usage example:
		 * \code
		 * const so_5::mbox_t & source = ...;
		 * const so_5::msink_t & dest = ...;
		 * so_5::multi_sink_binding_t & bindings = ...;
		 *
		 * binding->unbind_all_for(source, dest);
		 * \endcode
		 */
		void
		unbind_all_for(
			//! The source mbox.
			const mbox_t & from,
			//! The destination for messages.
			const msink_t & dest ) noexcept
			{
				this->lock_and_perform( [&]() {
						this->do_unbind_all_for( from, dest );
					} );
			}

		/*!
		 * Remove all exising bindings.
		 */
		void
		clear() noexcept
			{
				this->lock_and_perform( [&]() {
						this->do_clear();
					} );
			}
	};

} /* namespace so_5 */

