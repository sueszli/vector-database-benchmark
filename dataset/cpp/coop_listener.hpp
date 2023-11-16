/*
	SObjectizer 5.
*/

/*!
	\file
	\brief Interface for the cooperation listener definition.
*/

#pragma once

#include <string>
#include <memory>

#include <so_5/declspec.hpp>

#include <so_5/fwd.hpp>

namespace so_5
{

//
// coop_listener_t
//

//! Interface for the cooperation listener.
/*!
 * Cooperation listener is intended for observation moments of
 * cooperation registrations and deregistrations.
 *
 * \attention SObjectizer doesn't synchronize calls to the 
 * on_registered() and on_deregistered(). If this is a problem
 * then programmer should take care about the object's thread safety.
 */
class SO_5_TYPE coop_listener_t
{
		// Note: clang-3.9 requires this on Windows platform.
		coop_listener_t( const coop_listener_t & ) = delete;
		coop_listener_t( coop_listener_t && ) = delete;
		coop_listener_t & operator=( const coop_listener_t & ) = delete;
		coop_listener_t & operator=( coop_listener_t && ) = delete;

	public:
		coop_listener_t() = default;
		virtual ~coop_listener_t() noexcept = default;

		//! Hook for the cooperation registration event.
		/*!
		 * Method will be called right after the successful 
		 * cooperation registration.
		 *
		 * \note
		 * Since v.5.6.0 this method is noexcept!
		 */
		virtual void
		on_registered(
			//! SObjectizer Environment.
			environment_t & so_env,
			//! Cooperation which was registered.
			const coop_handle_t & coop ) noexcept = 0;

		//! Hook for the cooperation deregistration event.
		/*!
		 * Method will be called right after full cooperation deregistration.
		 *
		 * \note
		 * Since v.5.6.0 this method is noexcept!
		 */
		virtual void
		on_deregistered(
			//! SObjectizer Environment.
			environment_t & so_env,
			//! Cooperation which was registered.
			const coop_handle_t & coop,
			//! Reason of deregistration.
			const coop_dereg_reason_t & reason ) noexcept = 0;
};

//! Typedef for the coop_listener autopointer.
using coop_listener_unique_ptr_t = std::unique_ptr< coop_listener_t >;

} /* namespace so_5 */
