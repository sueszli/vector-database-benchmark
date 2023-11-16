/*
	SObjectizer 5.
*/

/*!
	\file
	\brief Definition of the template class mhood.
*/

#pragma once

#include <so_5/message.hpp>
#include <so_5/message_holder.hpp>

#include <so_5/compiler_features.hpp>

#include <type_traits>

namespace so_5
{

template< typename M >
class mhood_t;

namespace details
{

/*!
 * \since
 * v.5.5.14
 *
 * \brief A special selector for %mhood_t implementations.
 */
enum class mhood_type_t
{
	//! Message type is a classical message derived from so_5::message_t.
	classical_message,
	//! Message type is a classical signal derived from so_5::signal_t.
	classical_signal,
	//! Message type is not related to so_5::message_t.
	user_type_message
};

/*!
 * \since
 * v.5.5.14
 *
 * \brief A type to be used as a base for %mhood_t implementation.
 */
template< typename M, mhood_type_t T, message_mutability_t Mut >
class mhood_base_t
{};

/*!
 * \since
 * v.5.5.14
 *
 * \brief A special base for the case when message type is derived
 * from %message_t.
 *
 * \note
 * This is specialization for immutable message.
 * 
 * \tparam M type of message.
 */
template< typename M >
class mhood_base_t<
		M,
		mhood_type_t::classical_message,
		message_mutability_t::immutable_message >
{
public :
	using payload_type = typename message_payload_type< M >::payload_type;
	using envelope_type = typename message_payload_type< M >::envelope_type;

	mhood_base_t( message_ref_t & mf ) noexcept
		: m_msg{ message_payload_type< M >::extract_payload_ptr( mf ) }
		{}

	//! Access to the message.
	const payload_type *
	get() const noexcept { return m_msg; }

	//! Create a smart pointer for the message envelope.
	[[nodiscard]]
	intrusive_ptr_t< envelope_type >
	make_reference() const noexcept
		{
			return intrusive_ptr_t< envelope_type >{m_msg};
		}

	//! Create a holder for this message.
	/*!
	 * \since
	 * v.5.6.0
	 */
	template<
		message_ownership_t Ownership = message_ownership_t::autodetected >
	[[nodiscard]]
	message_holder_t< M, Ownership >
	make_holder() const noexcept
		{
			return { make_reference() };
		}

	//! Access to the message.
	const payload_type &
	operator * () const noexcept { return *get(); }

	//! Access to the message via pointer.
	const payload_type *
	operator->() const noexcept { return get(); }

private :
	/*!
	 * \note
	 * It should be a non-const pointer because of make_reference() method.
	 */
	payload_type * m_msg;
};

/*!
 * \since
 * v.5.5.19
 *
 * \brief A special base for the case when message type is derived
 * from %message_t.
 *
 * \note
 * This is specialization for mutable message.
 * 
 * \tparam M type of message.
 */
template< typename M >
class mhood_base_t<
		M,
		mhood_type_t::classical_message,
		message_mutability_t::mutable_message >
{
public :
	using payload_type = typename message_payload_type< M >::payload_type;
	using envelope_type = typename message_payload_type< M >::envelope_type;

	mhood_base_t( message_ref_t & mf ) noexcept
		: m_msg{ message_payload_type< M >::extract_payload_ptr( mf ) }
		{}

	mhood_base_t( const mhood_base_t & ) = delete;

	mhood_base_t( mhood_base_t && other ) noexcept
		: m_msg{ other.m_msg }
		{
			other.m_msg = nullptr;
		}

	friend void
	swap( mhood_base_t & a, mhood_base_t & b ) noexcept
		{
			using namespace std;
			swap( a, b );
		}

	mhood_base_t &
	operator=( const mhood_base_t & ) = delete;

	mhood_base_t &
	operator=( mhood_base_t && other ) noexcept
		{
			mhood_base_t tmp{ std::move(other) };
			swap( *this, tmp );
			return *this;
		}

	//! Access to the message.
	payload_type *
	get() const noexcept { return m_msg; }

	//! Create a smart pointer for the message envelope.
	[[nodiscard]]
	intrusive_ptr_t< envelope_type >
	make_reference() noexcept
		{
			intrusive_ptr_t< envelope_type > result{m_msg};
			m_msg = nullptr;
			return result;
		}

	//! Create a holder for this message.
	/*!
	 * \since
	 * v.5.6.0
	 */
	template<
		message_ownership_t Ownership = message_ownership_t::autodetected >
	[[nodiscard]]
	message_holder_t< M, Ownership >
	make_holder() noexcept
		{
			return { make_reference() };
		}

	//! Access to the message.
	payload_type &
	operator*() noexcept { return *get(); }

	//! Access to the message via pointer.
	payload_type *
	operator->() noexcept { return get(); }

private :
	payload_type * m_msg;
};

/*!
 * \since
 * v.5.5.14
 *
 * \brief A special base for the case when message type is derived
 * from %message_t.
 * 
 * \tparam M type of signal.
 */
template< typename M >
class mhood_base_t<
		M,
		mhood_type_t::classical_signal,
		message_mutability_t::immutable_message >
{
public :
	mhood_base_t( message_ref_t & ) {}
};

/*!
 * \since
 * v.5.5.19
 *
 * \brief A special base for the case when message type is derived
 * from %message_t.
 * 
 * \tparam M type of signal.
 */
template< typename M >
class mhood_base_t<
		M,
		mhood_type_t::classical_signal,
		message_mutability_t::mutable_message >
{
public :
	mhood_base_t( message_ref_t & ) {}
};

/*!
 * \since
 * v.5.5.14
 *
 * \brief A special base for the case when message type is not
 * related to %message_t.
 *
 * \note
 * This specialization is for immutable message.
 * 
 * \tparam M type of message.
 */
template< typename M >
class mhood_base_t<
		M,
		mhood_type_t::user_type_message,
		message_mutability_t::immutable_message >
{
public :
	using payload_type = typename message_payload_type< M >::payload_type;
	using envelope_type = typename message_payload_type< M >::envelope_type;

	mhood_base_t( message_ref_t & mf ) noexcept
		: m_payload{ message_payload_type< M >::extract_payload_ptr( mf ) }
		, m_envelope{ message_payload_type< M >::extract_envelope_ptr( mf ) }
		{}

	//! Access to the message.
	const payload_type *
	get() const noexcept { return m_payload; }

	//! Create a smart pointer for the message envelope.
	intrusive_ptr_t< envelope_type >
	make_reference() const noexcept
		{
			return intrusive_ptr_t< envelope_type >{m_envelope};
		}

	//! Create a holder for this message.
	/*!
	 * \since
	 * v.5.6.0
	 */
	template<
		message_ownership_t Ownership = message_ownership_t::autodetected >
	[[nodiscard]]
	message_holder_t< M, Ownership >
	make_holder() const noexcept
		{
			return { make_reference() };
		}

	//! Access to the message.
	const payload_type &
	operator*() const noexcept { return *get(); }

	//! Access to the message via pointer.
	const payload_type *
	operator->() const noexcept { return get(); }

private :
	const payload_type * m_payload;
	envelope_type * m_envelope;
};

/*!
 * \since
 * v.5.5.19
 *
 * \brief A special base for the case when message type is not
 * related to %message_t.
 *
 * \note
 * This implementation is for mutable message.
 * 
 * \tparam M type of message.
 */
template< typename M >
class mhood_base_t<
		M,
		mhood_type_t::user_type_message,
		message_mutability_t::mutable_message >
{
public :
	using payload_type = typename message_payload_type< M >::payload_type;
	using envelope_type = typename message_payload_type< M >::envelope_type;

	mhood_base_t( message_ref_t & mf ) noexcept
		: m_payload{ message_payload_type< M >::extract_payload_ptr( mf ) }
		, m_envelope{ message_payload_type< M >::extract_envelope_ptr( mf ) }
		{}

	mhood_base_t( const mhood_base_t & ) = delete;

	mhood_base_t( mhood_base_t && other ) noexcept
		: m_payload( other.m_payload )
		, m_envelope( other.m_envelope )
		{
			other.m_payload = nullptr;
			other.m_envelope = nullptr;
		}

	friend void
	swap( mhood_base_t & a, mhood_base_t & b ) noexcept
		{
			using namespace std;
			swap( a.m_payload, b.m_payload );
			swap( a.m_envelope, b.m_envelope );
		}

	mhood_base_t &
	operator=( const mhood_base_t & ) = delete;

	mhood_base_t &
	operator=( mhood_base_t && other ) noexcept
		{
			mhood_base_t tmp{ std::move(other) };
			swap( *this, tmp );
			return *this;
		}

	//! Access to the message.
	payload_type *
	get() noexcept { return m_payload; }

	//! Create a smart pointer for the message envelope.
	[[nodiscard]]
	intrusive_ptr_t< envelope_type >
	make_reference() noexcept
		{
			intrusive_ptr_t< envelope_type > result{m_envelope};

			m_payload = nullptr;
			m_envelope = nullptr;

			return result;
		}

	//! Create a holder for this message.
	/*!
	 * \since
	 * v.5.6.0
	 */
	template<
		message_ownership_t Ownership = message_ownership_t::autodetected >
	[[nodiscard]]
	message_holder_t< M, Ownership >
	make_holder() noexcept
		{
			return { make_reference() };
		}

	//! Access to the message.
	payload_type &
	operator*() noexcept { return *get(); }

	//! Access to the message via pointer.
	payload_type *
	operator->() noexcept { return get(); }

private :
	payload_type * m_payload;
	envelope_type * m_envelope;
};

/*!
 * \since
 * v.5.5.14
 *
 * \brief A special selector of message hood type.
 */
template< typename T >
struct mhood_type_detector
{
	static constexpr const mhood_type_t mhood_type =
			is_signal< T >::value ?
					mhood_type_t::classical_signal :
					is_classical_message< T >::value ?
							mhood_type_t::classical_message :
							mhood_type_t::user_type_message;
};

/*!
 * \brief A special detector of message immutability/mutability.
 *
 * \since
 * v.5.5.19
 */
template< typename T >
struct mhood_mutability_detector
{
	static constexpr const message_mutability_t mutability =
		message_mutability_traits<T>::mutability;
};

/*!
 * \brief Just a useful typename for base type of mhood implementation.
 *
 * \since
 * v.5.5.19
 */
template< typename Msg >
using actual_mhood_base_type = mhood_base_t<
		Msg,
		mhood_type_detector<Msg>::mhood_type,
		mhood_mutability_detector<Msg>::mutability >;

} /* namespace details */

/*!
 * \since
 * v.5.5.14
 *
 * \brief A message wrapped to be used as type of argument for
 * event handlers.
 *
 * \tparam M type of message or signal.
 *
 * \note This class provides different sets of methods in dependency of
 * type \a M. If M is a type derived from so_5::message_t, then there
 * will be the following methods for immutable message M:
 * \code
	const M * get() const;
	const M * operator->() const;
	const M & operator*() const;
	so_5::intrusive_ptr_t< M > make_reference() const;
	so_5::message_holder_t< M, Ownership > make_holder<Ownership>() const;
 * \endcode
 * If M is a type derived from so_5::signal_t, then there will no be methods
 * at all. It is because there is no actual message object for a signal.
 * If M is user-type message then there will be the following methods
 * for immutable message M:
 * \code
	const M * get() const;
	const M * operator->() const;
	const M & operator*() const;
	so_5::intrusive_ptr_t< so_5::user_type_message_t<M> > make_reference() const;
	so_5::message_holder_t< M, Ownership > make_holder<Ownership>() const;
 * \endcode
 * For mutable message M there will be the following methods is M is
 * derived from so_5::message_t:
 * \code
	M * get();
	M * operator->();
	M & operator*();
	so_5::intrusive_ptr_t< M > make_reference();
	so_5::message_holder_t< M, Ownership > make_holder<Ownership>();
 * \endcode
 * If mutable message M is not derived from so_5::message_t then there
 * will be the following methods:
 * \code
	M * get();
	M * operator->();
	M & operator*();
	so_5::intrusive_ptr_t< so_5::user_type_message_t<M> > make_reference();
	so_5::message_holder_t< M, Ownership > make_holder<Ownership>();
 * \endcode
 *
 * \note
 * Method make_holder() uses so_5::message_ownership_t::autodetected by
 * default.
 *
 * \attention
 * Methods make_reference() for mhood_t<mutable_msg<M>> will leave mhood object
 * in nullptr state. It means that mhood must not be used after calling
 * make_reference().
 *
 * \attention
 * Methods make_holder() for mhood_t<mutable_msg<M>> will leave
 * the mhood object empty always. Regardless of Ownership parameter.
 * Methods make_holder() for mhood_t<M> and mhood_t<immutable_msg<M>> will
 * leave value for mhood object regardless of Ownership parameter.
 * For example:
 * \code
 * class demo : public agent_t {
 * 	void on_immutable_msg(mhood_t<some_msg> cmd) {
 * 		auto unique_holder = cmd.make_holder<so_5::message_ownership_t::unique>();
 * 		auto v = cmd->some_value(); // cmd still has its value.
 * 		...
 * 	}
 *		...
 *		void on_mutable_msg(mhood_t< mutable_msg<another_msg> > cmd) {
 * 		auto shared_holder = cmd.make_holder<so_5::message_ownership_t::shared>();
 * 		// Now cmd is empty!
 * 		auto v = cmd->some_value(); // THIS IS UB!	
 *		}
 * };
 * \endcode
 *
 * \note Class mhood_t can be used for redirection of messages of user types.
 * For example (since v.5.5.19):
	\code
	class redirector : public so_5::agent_t
	{
		...
		void some_event( mhood_t< std::string > evt ) {
			// Redirection of message to the different mbox.
			so_5::send(m_another_mbox, evt);
		}
	};
	\endcode
 * If mhood_t<mutable_msg<M>> is used then a redirection must be done this
 * way:
	\code
	class redirector : public so_5::agent_t
	{
		...
		void some_event( mhood_t< mutable_msg<std::string> > evt ) {
			// Redirection of message to the different mbox.
			so_5::send(m_another_mbox, std::move(evt));
			// NOTE: evt is nullptr at this point. It must not be used anymore.
		}
	};
	\endcode
 */
template< typename M >
class mhood_t
	: public details::actual_mhood_base_type<M>
{
	using base_type_t = details::actual_mhood_base_type<M>;

public :
	mhood_t( message_ref_t & mf ) : base_type_t{ mf } {}
};

//! Transform mutable message instance into immutable.
/*!
 * \attention
 * Moves all content from \a msg into new object and leaves \a msg object in
 * nullptr state.
 *
 * Usage example:
 * \code
	void on_hello_mut( mhood_t< so_5::mutable_msg< hello > > cmd )
	{
		cmd->m_content = "bye";
		send( another_mbox, to_immutable( std::move(cmd) ) );
	}
 * \endcode
 *
 * \since
 * v.5.5.19
 */
template< typename M >
typename std::enable_if< !is_signal<M>::value, mhood_t< immutable_msg< M > > >::type
to_immutable( mhood_t< mutable_msg<M> > msg )
	{
		message_ref_t ref{ msg.make_reference() }; 
		change_message_mutability( ref, message_mutability_t::immutable_message );

		return { ref };
	}

template< typename M >
typename std::enable_if< is_signal<M>::value, mhood_t< immutable_msg< M > > >::type
to_immutable( mhood_t< mutable_msg<M> > /*msg*/ )
	{
		message_ref_t ref; 
		return { ref };
	}

/*!
 * \since
 * v.5.5.14
 *
 * \brief A special case for %mhood_t to prevent usage of %mhood_t as wrapper
 * for %user_type_message_t<T>.
 */
template< typename M >
class mhood_t< user_type_message_t< M > >;

/*!
 * \brief A short name for message hood for mutable message.
 *
 * \since
 * v.5.5.19
 */
template< typename M >
using mutable_mhood_t = mhood_t< mutable_msg<M> >;

} /* namespace so_5 */

