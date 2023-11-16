/*
	SObjectizer 5.
*/

/*!
	\file
	\brief SObjectizer Environment definition.
*/

#pragma once

#include <so_5/compiler_features.hpp>
#include <so_5/coop.hpp>
#include <so_5/coop_listener.hpp>
#include <so_5/custom_mbox.hpp>
#include <so_5/declspec.hpp>
#include <so_5/disp_binder.hpp>
#include <so_5/environment_infrastructure.hpp>
#include <so_5/error_logger.hpp>
#include <so_5/event_exception_logger.hpp>
#include <so_5/event_queue_hook.hpp>
#include <so_5/exception.hpp>
#include <so_5/mbox.hpp>
#include <so_5/mbox_namespace_name.hpp>
#include <so_5/mchain.hpp>
#include <so_5/message.hpp>
#include <so_5/msg_tracing.hpp>
#include <so_5/nonempty_name.hpp>
#include <so_5/queue_locks_defaults_manager.hpp>
#include <so_5/so_layer.hpp>
#include <so_5/stop_guard.hpp>
#include <so_5/timers.hpp>

#include <so_5/stats/controller.hpp>
#include <so_5/stats/repository.hpp>

#include <so_5/disp/one_thread/params.hpp>
#include <so_5/disp/nef_one_thread/params.hpp>
#include <so_5/disp/abstract_work_thread.hpp>

#include <chrono>
#include <functional>
#include <memory>
#include <type_traits>
#include <variant>

#if defined( SO_5_MSVC )
	#pragma warning(push)
	#pragma warning(disable: 4251)
#endif

namespace so_5
{

namespace low_level_api
{

struct schedule_timer_params_t
	{
		//! Message type.
		const std::type_index & m_msg_type;
		//! Message to be sent after timeout.
		const message_ref_t & m_msg;
		//! Mbox to which message will be delivered.
		const mbox_t & m_mbox;
		//! Timeout before the first delivery.
		std::chrono::steady_clock::duration m_pause;
		//! Period of the delivery repetition for periodic messages.
		std::chrono::steady_clock::duration m_period;
	};

struct single_timer_params_t
	{
		//! Message type.
		const std::type_index & m_msg_type;
		//! Message to be sent after timeout.
		const message_ref_t & m_msg;
		//! Mbox to which message will be delivered.
		const mbox_t & m_mbox;
		//! Timeout before the delivery.
		std::chrono::steady_clock::duration m_pause;
	};

} /* namespace low_level_api */

//
// environment_params_t
//

//! Parameters for the SObjectizer Environment initialization.
/*!
 * This class is used for setting SObjectizer Parameters.
 *
 * \see http://www.parashift.com/c++-faq/named-parameter-idiom.html
 */
class SO_5_TYPE environment_params_t
{
	public:
		/*!
		 * \brief A sum type for holding parameters for the default disp.
		 *
		 * \since v.5.8.0
		 */
		using default_disp_params_t = std::variant<
				disp::one_thread::disp_params_t,
				disp::nef_one_thread::disp_params_t
			>;

		/*!
		 * \brief Constructor.
		 *
		 * Sets default values for parameters.
		 */
		environment_params_t();
		/*!
		 * \brief Move constructor.
		 *
		 * \since v.5.2.3
		 */
		environment_params_t( environment_params_t && other );
		~environment_params_t();

		/*!
		 * \brief Move operator.
		 *
		 * \since v.5.2.3
		 */
		environment_params_t &
		operator=( environment_params_t && other ) noexcept;

		/*!
		 * \brief Swap operation.
		 *
		 * \since v.5.2.3
		 */
		friend SO_5_FUNC void
		swap( environment_params_t & a, environment_params_t & b ) noexcept;

		//! Set the timer_thread factory.
		/*!
		 * If \a factory is a null then the default timer thread
		 * will be used.
		 */
		environment_params_t &
		timer_thread(
			//! timer_thread factory to be used.
			so_5::timer_thread_factory_t factory );

		//! Add an additional layer to the SObjectizer Environment.
		/*!
		 * If this layer is already added it will be replaced by \a layer_ptr.
		 * 
		 * The method distinguishes layers from each other by the type SO_LAYER.
		*/
		template< class SO_Layer >
		environment_params_t &
		add_layer(
			//! A layer to be added.
			std::unique_ptr< SO_Layer > layer_ptr )
		{
			if( layer_ptr.get() )
			{
				layer_unique_ptr_t ptr( layer_ptr.release() );

				add_layer(
					std::type_index( typeid( SO_Layer ) ),
					std::move( ptr ) );
			}

			return *this;
		}

		//! Set cooperation listener object.
		environment_params_t &
		coop_listener(
			coop_listener_unique_ptr_t coop_listener );

		//! Set exception logger object.
		environment_params_t &
		event_exception_logger(
			event_exception_logger_unique_ptr_t logger );

		/*!
		 * \name Exception reaction flag management methods.
		 * \{
		 */
		/*!
		 * \brief Get exception reaction flag value.
		 *
		 * \note
		 * This method is noexcept since v.5.8.0.
		 *
		 * \since v.5.3.0
		 */
		inline exception_reaction_t
		exception_reaction() const noexcept
		{
			return m_exception_reaction;
		}

		/*!
		 * \brief Set exception reaction flag value.
		 *
		 * Usage example:
		 * \code
		 * so_5::launch([](so_5::environment_t & env) {...},
		 * 	[](so_5::environment_params_t & params) {
		 * 		params.exception_reaction(so_5::exception_reaction_t::shutdown_sobjectizer_on_exception);
		 * 		...
		 * 	});
		 * \endcode
		 *
		 * \note
		 * This method is noexcept since v.5.8.0.
		 *
		 * \since v.5.3.0
		 */
		environment_params_t &
		exception_reaction( exception_reaction_t value ) noexcept
		{
			m_exception_reaction = value;
			return *this;
		}
		/*!
		 * \}
		 */

		/*!
		 * \brief Do not shutdown SO Environment when it is becomes empty.
		 *
		 * \par Description
		 * Since v.5.4.0 SO Environment checks count of live cooperations
		 * after every cooperation deregistration. If there is no more
		 * live cooperations then SO Environment will be shutted down.
		 * If it is not appropriate then this method must be called.
		 * It disables autoshutdown of SO Environment. Event if there is
		 * no more live cooperations SO Environment will work until
		 * explicit call to environment_t::stop() method.
		 *
		 * \since v.5.4.0
		 */
		environment_params_t &
		disable_autoshutdown()
		{
			m_autoshutdown_disabled = true;
			return *this;
		}

		/*!
		 * \brief Is autoshutdown disabled?
		 *
		 * \see disable_autoshutdown()
		 *
		 * \since v.5.4.0
		 */
		[[nodiscard]]
		bool
		autoshutdown_disabled() const
		{
			return m_autoshutdown_disabled;
		}

		/*!
		 * \brief Set error logger for the environment.
		 *
		 * \since v.5.5.0
		 */
		environment_params_t &
		error_logger( error_logger_shptr_t logger )
		{
			m_error_logger = std::move(logger);
			return *this;
		}

		/*!
		 * \brief Set message delivery tracer for the environment.
		 *
		 * Usage example:
		 * \code
		 * so_5::launch( [](so_5::environment_t & env) { ... },
		 * 	[](so_5::environment_params_t & params) {
		 * 		params.message_delivery_tracer( so_5::msg_tracing::std_cout_tracer() );
		 * 		...
		 * 	} );
		 * \endcode
		 *
		 * \since v.5.5.9
		 */
		environment_params_t &
		message_delivery_tracer( so_5::msg_tracing::tracer_unique_ptr_t tracer )
		{
			m_message_delivery_tracer = std::move( tracer );
			return *this;
		}

		/*!
		 * \brief Set message tracer filter for the environment.
		 *
		 * \since v.5.5.22
		 */
		environment_params_t &
		message_delivery_tracer_filter(
			so_5::msg_tracing::filter_shptr_t filter )
		{
			m_message_delivery_tracer_filter = std::move( filter );
			return *this;
		}

		/*!
		 * \brief Set parameters for a case when one_thread-disp
		 * must be used as the default dispatcher.
		 *
		 * \par Usage example:
			\code
			so_5::launch( []( so_5::environment_t & env ) { ... },
				[]( so_5::environment_params_t & env_params ) {
					using namespace so_5::disp::one_thread;
					// Event queue for the default dispatcher must use mutex as lock.
					env_params.default_disp_params( disp_params_t{}.tune_queue_params(
						[]( queue_traits::queue_params_t & queue_params ) {
							queue_params.lock_factory( queue_traits::simple_lock_factory() );
						} ) );
				} );
			\endcode
		 *
		 * \note
		 * If some parameters have already been set, the old parameters will be
		 * replaced by new ones.
		 *
		 * \since v.5.5.10
		 */
		environment_params_t &
		default_disp_params( so_5::disp::one_thread::disp_params_t params )
		{
			m_default_disp_params = std::move(params);
			return *this;
		}

		/*!
		 * \brief Set parameters for a case when
		 * nef_one_thread-disp must be used as the default dispatcher.
		 *
		 * \par Usage example:
			\code
			so_5::launch( []( so_5::environment_t & env ) { ... },
				[]( so_5::environment_params_t & env_params ) {
					using namespace so_5::disp::nef_one_thread;
					// Event queue for the default dispatcher must use mutex as lock.
					env_params.default_disp_params( disp_params_t{}.tune_queue_params(
						[]( queue_traits::queue_params_t & queue_params ) {
							queue_params.lock_factory( queue_traits::simple_lock_factory() );
						} ) );
				} );
			\endcode
		 *
		 * \note
		 * If some parameters have already been set, the old parameters will be
		 * replaced by new ones.
		 *
		 * \since v.5.8.0
		 */
		environment_params_t &
		default_disp_params( so_5::disp::nef_one_thread::disp_params_t params )
		{
			m_default_disp_params = std::move(params);
			return *this;
		}

		/*!
		 * \brief Get the parameters for the default dispatcher.
		 *
		 * \attention
		 * The returned reference will be invalidated by any subsequent
		 * calls to default_disp_params-setters.
		 *
		 * \note
		 * Since v.5.8.0 it returns a sum type with parameters for different
		 * types of dispatchers. Therefore, the returned value has to be examined
		 * accordingly, for example:
		 * \code
		 * so_5::environment_params_t & params = ...
		 * const auto & disp_params = params.default_disp_params();
		 * if( const auto * one_thread =
		 * 	std::get_if< so_5::disp::one_thread::disp_params_t >( &disp_params ) )
		 * {
		 * 	... // Handling.
		 * }
		 * else if( const auto * nef_one_thread =
		 * 	std::get_if< so_5::disp::nef_one_thread::disp_params_t >( &disp_params ) )
		 * {
		 * 	... // Handling.
		 * }
		 * \endcode
		 *
		 * \since v.5.5.10
		 */
		const default_disp_params_t &
		default_disp_params() const
		{
			return m_default_disp_params;
		}

		/*!
		 * \brief Set activity tracking flag for the whole SObjectizer Environment.
		 *
		 * \since v.5.5.18
		 */
		environment_params_t &
		work_thread_activity_tracking(
			work_thread_activity_tracking_t flag )
		{
			m_work_thread_activity_tracking = flag;
			return *this;
		}

		/*!
		 * \brief Get activity tracking flag for the whole SObjectizer Environment.
		 *
		 * \since v.5.5.18
		 */
		work_thread_activity_tracking_t
		work_thread_activity_tracking() const
		{
			return m_work_thread_activity_tracking;
		}

		//! Helper for turning work thread activity tracking on.
		/*!
		 * \since v.5.5.18
		 */
		environment_params_t &
		turn_work_thread_activity_tracking_on()
			{
				return work_thread_activity_tracking(
						work_thread_activity_tracking_t::on );
			}

		//! Helper for turning work thread activity tracking off.
		/*!
		 * \since v.5.5.18
		 */
		environment_params_t &
		turn_work_thread_activity_tracking_off()
			{
				return work_thread_activity_tracking(
						work_thread_activity_tracking_t::off );
			}

		//! Set manager for queue locks defaults.
		/*!
		 * \since v.5.5.18
		 */
		environment_params_t &
		queue_locks_defaults_manager(
			queue_locks_defaults_manager_unique_ptr_t manager )
			{
				m_queue_locks_defaults_manager = std::move(manager);
				return *this;
			}

		//! Get the current environment infrastructure factory.
		/*!
		 * \since v.5.5.19
		 */
		const environment_infrastructure_factory_t &
		infrastructure_factory() const
			{
				return m_infrastructure_factory;
			}

		//! Set new environment infrastructure factory.
		/*!
		 * Usage example:
		 * \code
		 * so_5::launch( []( so_5::environment_t & env ) {
		 * 		... // Some initial actions.
		 * 	},
		 * 	[]( so_5::environment_params_t & params ) {
		 * 		// Set infrastructure factory to make simple not-thread-safe
		 * 		// single-threaded infrastructure.
		 * 		params.infrastructure_factory(
		 * 			so_5::env_infrastructures::simple_not_mtsafe::factory() );
		 * 	} );
		 * \endcode
		 *
		 * \since v.5.5.19
		 */
		environment_params_t &
		infrastructure_factory(
			environment_infrastructure_factory_t factory )
			{
				m_infrastructure_factory = std::move(factory);
				return *this;
			}

		/*!
		 * \brief Set event_queue_hook object.
		 *
		 * Since v.5.5.24 it is possible to use special event_queue_hook
		 * object. If it is used it should be set for SObjectizer
		 * Environment before the Environment will be started. This method
		 * allows to specify event_queue_hook object for a new Environment
		 * object.
		 *
		 * Usage example:
		 * \code
		 * so_5::launch(
		 * 	[](so_5::environment_t & env) {...}, // Some stating actions.
		 * 	[](so_5::environment_params_t & params) {
		 * 		// Set my own event_queue hook object.
		 * 		so_5::make_event_queue_hook<my_hook>(
		 * 			// Object is created dynamically and should be
		 * 			// destroyed the normal way.
		 * 			so_5::event_queue_hook_t::default_deleter,
		 * 			arg1, arg3, arg3 // and all other arguments for my_hook's constructor.
		 * 		);
		 * 	});
		 * \endcode
		 *
		 * \note
		 * The previous event_queue_hook object (if it was set earlier)
		 * will just be dropped.
		 *
		 * \since v.5.5.24
		 */
		void
		event_queue_hook(
			event_queue_hook_unique_ptr_t hook )
			{
				m_event_queue_hook = std::move(hook);
			}

		/*!
		 * \brief Set work thread factory to be used by default.
		 *
		 * Since v.5.7.3 it's possible to use custom worker threads instead of
		 * standard ones provided by SObjectizer. To do so it's required to
		 * provide an appropriate worker thread factory. It can be done by
		 * specifying a factory in parameters for a dispatcher. Or it can be done
		 * globally for the whole SObjectizer's Environment by this method.
		 *
		 * This method sets a global factory that will be used by default. It
		 * means that:
		 *
		 * - the global factory won't be used if there is a factory in
		 *   dispatcher's parameters (in that case dispatcher's factory has
		 *   higher priority and the default factory is ignored);
		 * - the global factory will be used if there is no factory specified in
		 *   dispatcher's parameters;
		 * - the global factory will be used for the default one_thread dispatcher
		 *   created by default multi-thread environment infrastructure.
		 *
		 * Usage example:
		 * \code
		 * class my_thread_factory final : public so_5::disp::abstract_work_thread_factory_t
		 * {
		 * 	...
		 * };
		 * ...
		 * so_5::launch( [](so_5::environment_t & env) {
		 * 		env.introduce_coop( [](so_5::coop_t & coop) {
		 * 			// This agent will be bound to the default dispatcher.
		 * 			// Worker thread for the default dispatcher will be provided
		 * 			// by default factory.
		 * 			coop.make_agent<my_agent>(...);
		 *
		 * 			// This agent will be bound to an instance of new one_thread
		 * 			// dispatcher. A worker thread for that new dispatcher will
		 * 			// be provided by the default factory.
		 * 			coop.make_agent_with_binder<my_agent>(
		 * 				so_5::disp::one_thread::make_dispatcher(coop.environment()).binder(),
		 * 				...);
		 *
		 * 			// This dispatcher will use another own instance of
		 * 			// thread factory.
		 * 			auto tp_disp = so_5::disp::thread_pool::make_dispatcher(
		 * 				coop.environment(),
		 * 				"my_pool_disp",
		 * 				so_5::disp::thread_pool::disp_params_t{}
		 * 					.thread_count(10)
		 * 					.work_thread_factory(std::make_shared<my_thread_factory>(...)) );
		 * 			// This agent will be bound to the new thread_pool dispatcher
		 * 			// and will work on a thread provided by a separate thread factory.
		 * 			coop.make_agent_with_binder<my_agent>(tp_disp.binder());
		 * 		} );
		 * 		...
		 * 	},
		 * 	[](so_5::environment_params_t & params) {
		 * 		params.work_thread_factory( std::make_shared<my_thread_factory>(...) );
		 * 	} );
		 * \endcode
		 *
		 * \since v.5.7.3
		 */
		environment_params_t &
		work_thread_factory(
			so_5::disp::abstract_work_thread_factory_shptr_t factory )
			{
				m_work_thread_factory = std::move(factory);
				return *this;
			}

		/*!
		 * \name Methods for internal use only.
		 * \{
		 */
		//! Get map of default SObjectizer's layers.
		[[nodiscard]]
		layer_map_t
		so5_giveout_layers_map()
		{
			return std::move( m_so_layers );
		}

		//! Get cooperation listener.
		coop_listener_unique_ptr_t
		so5_giveout_coop_listener()
		{
			return std::move( m_coop_listener );
		}

		//! Get exception logger.
		event_exception_logger_unique_ptr_t
		so5_giveout_event_exception_logger()
		{
			return std::move( m_event_exception_logger );
		}

		//! Get the timer_thread factory.
		so_5::timer_thread_factory_t
		so5_giveout_timer_thread_factory()
		{
			return std::move( m_timer_thread_factory );
		}

		//! Get error logger for the environment.
		const error_logger_shptr_t &
		so5_error_logger() const
		{
			return m_error_logger;
		}

		/*!
		 * \brief Get message delivery tracer for the environment.
		 *
		 * \since v.5.5.9
		 */
		so_5::msg_tracing::tracer_unique_ptr_t
		so5_giveout_message_delivery_tracer()
		{
			return std::move( m_message_delivery_tracer );
		}

		/*!
		 * \brief Get message delivery tracer filter for the environment.
		 *
		 * \since v.5.5.22
		 */
		so_5::msg_tracing::filter_shptr_t
		so5_giveout_message_delivery_tracer_filter()
		{
			return std::move( m_message_delivery_tracer_filter );
		}

		//! Take out queue locks defaults manager.
		/*!
		 * \since v.5.5.18
		 */
		queue_locks_defaults_manager_unique_ptr_t
		so5_giveout_queue_locks_defaults_manager()
			{
				return std::move( m_queue_locks_defaults_manager );
			}

		//! Take out event_queue_hook object.
		/*!
		 * \since v.5.5.24
		 */
		event_queue_hook_unique_ptr_t
		so5_giveout_event_queue_hook()
			{
				return std::move( m_event_queue_hook );
			}

		//! Take out work_thread_factory object.
		/*!
		 * \since v.5.7.3
		 */
		so_5::disp::abstract_work_thread_factory_shptr_t
		so5_giveout_work_thread_factory()
			{
				return std::move( m_work_thread_factory );
			}
		/*!
		 * \}
		 */

	private:
		//! Add an additional layer.
		/*!
		 * If this layer is already added it will be replaced by \a layer_ptr.
		 * 
		 * The method distinguishes layers from each other by the type SO_LAYER.
		 */
		void
		add_layer(
			//! Type identification for layer.
			const std::type_index & type,
			//! A layer to be added.
			layer_unique_ptr_t layer_ptr );

		//! Timer thread factory.
		so_5::timer_thread_factory_t m_timer_thread_factory;

		//! Additional layers.
		layer_map_t m_so_layers;

		//! Cooperation listener.
		coop_listener_unique_ptr_t m_coop_listener;

		//! Exception logger.
		event_exception_logger_unique_ptr_t m_event_exception_logger;

		/*!
		 * \brief Exception reaction flag for the whole SO Environment.
		 *
		 * \since v.5.3.0
		 */
		exception_reaction_t m_exception_reaction;

		/*!
		 * \brief Is autoshutdown when there is no more cooperation disabled?
		 *
		 * \see disable_autoshutdown()
		 *
		 * \since v.5.4.0
		 */
		bool m_autoshutdown_disabled;

		/*!
		 * \brief Error logger for the environment.
		 *
		 * \since v.5.5.0
		 */
		error_logger_shptr_t m_error_logger;

		/*!
		 * \brief Tracer for message delivery.
		 *
		 * \since v.5.5.9
		 */
		so_5::msg_tracing::tracer_unique_ptr_t m_message_delivery_tracer;

		/*!
		 * \brief Message delivery tracer filter to be used with environment.
		 *
		 * \since v.5.5.22
		 */
		so_5::msg_tracing::filter_shptr_t m_message_delivery_tracer_filter;

		/*!
		 * \brief Parameters for the default dispatcher.
		 *
		 * \since v.5.5.10
		 */
		default_disp_params_t m_default_disp_params;

		/*!
		 * \brief Work thread activity tracking for the whole Environment.
		 *
		 * \since v.5.5.18
		 */
		work_thread_activity_tracking_t m_work_thread_activity_tracking;

		/*!
		 * \brief Manager for defaults of queue locks.
		 *
		 * \since v.5.5.18
		 */
		queue_locks_defaults_manager_unique_ptr_t m_queue_locks_defaults_manager;

		/*!
		 * \brief A factory for environment infrastructure entity.
		 *
		 * \since v.5.5.19
		 */
		environment_infrastructure_factory_t m_infrastructure_factory;

		/*!
		 * \brief An event_queue_hook object.
		 *
		 * \note
		 * It can be a nullptr. It means that no event_queue_hook should
		 * be used.
		 *
		 * \since v.5.5.24
		 */
		event_queue_hook_unique_ptr_t m_event_queue_hook;

		/*!
		 * \brief Global factory for work threads.
		 *
		 * \note
		 * It can be a nullptr. It means that standard work thread factory
		 * has to be used as the global factory.
		 *
		 * \since v.5.7.3
		 */
		so_5::disp::abstract_work_thread_factory_shptr_t m_work_thread_factory;
};

//
// environment_t
//

//! SObjectizer Environment.
/*!
 * \section so_env__intro Basic information
 *
 * The SObjectizer Environment provides a basic infrastructure for
 * the SObjectizer Run-Time execution.
 *
 * The main method of starting SObjectizer Environment creates a
 * class derived from the environment_t and reimplementing the
 * environment_t::init() method.
 * This method should be used to define starting actions of
 * application. For example first application cooperations can
 * be registered here and starting messages can be sent to them.
 *
 * The SObjectizer Environment calls the environment_t::init() when
 * the SObjectizer Run-Time is successfully started. 
 * If something happened during the Run-Time startup then 
 * the method init() will not be called.
 *
 * The SObjectizer Run-Time is started by the environment_t::run().
 * This method blocks the caller thread until SObjectizer completely
 * finished its work.
 *
 * The SObjectizer Run-Time is finished by the environment_t::stop().
 * This method doesn't block the caller thread. Instead it sends a special
 * shutdown signal to the Run-Time. The SObjectizer Run-Time then 
 * informs agents about this and waits finish of agents work.
 * The SObjectizer Run-Time finishes if all agents are stopped and
 * all cooperations are deregistered.
 *
 * Methods of the SObjectizer Environment can be splitted into the
 * following groups:
 * - working with mboxes;
 * - working with dispatchers, exception loggers and handlers;
 * - working with cooperations;
 * - working with delayed and periodic messages;
 * - working with additional layers;
 * - initializing/running/stopping/waiting of the Run-Time.
 *
 * \section so_env__mbox_methods Methods for working with mboxes.
 *
 * SObjectizer Environment allows creation of named and anonymous mboxes.
 * Syncronization objects for these mboxes can be obtained from
 * common pools or assigned by a user during mbox creation.
 *
 * Mboxes are created by environment_t::create_mbox() methods.
 * All these methods return the mbox_t which is a smart reference 
 * to the mbox.
 *
 * An anonymous mbox is automatically destroyed when the last reference to it is
 * destroyed. So, to save the anonymous mbox, the mbox_ref from 
 * the create_mbox() should be stored somewhere.
 *
 * \section so_env__coop_methods Methods for working with cooperations.
 *
 * Cooperations can be created by environment_t::make_coop() methods.
 *
 * The method environment_t::register_coop() should be used for the 
 * cooperation registration.
 *
 * Method environment_t::deregister_coop() should be used for the 
 * cooperation deregistration.
 */
class SO_5_TYPE environment_t
{
	friend class so_5::impl::internal_env_iface_t;

		//! Auxiliary methods for getting reference to itself.
		/*!
		 * Could be used in constructors without compiler warnings.
		 */
		environment_t &
		self_ref();

	public:
		explicit environment_t(
			//! Initialization params.
			environment_params_t && so_environment_params );

		virtual ~environment_t();

		environment_t( const environment_t & ) = delete;
		environment_t &
		operator=( const environment_t & ) = delete;

		/*!
		 * \name Methods for working with mboxes.
		 * \{
		 */

		//! Create an anonymous MPMC mbox.
		/*!
		 * Usage example:
		 * \code
		 * class my_agent final : public so_5::agent_t {
		 * 	const so_5::mbox_t broadcast_mbox_;
		 * 	...
		 * public:
		 * 	my_agent(context_t ctx)
		 * 		: so_5::agent_t{std::move(ctx)}
		 * 		, broadcast_mbox_{so_environment().create_mbox()}
		 * 	{}
		 * 	...
		 * };
		 * \endcode
		 *
		 *	\note always creates a new mbox.
		 */
		[[nodiscard]]
		mbox_t
		create_mbox();

		//! Create named MPMC mbox.
		/*!
		 * If \a mbox_name is unique then a new mbox will be created.
		 * If not the reference to existing mbox will be returned.
		 *
		 * Usage example:
		 * \code
		 * class first_participant final : public so_5::agent_t {
		 * 	const so_5::mbox_t broadcast_mbox_;
		 * 	...
		 * public:
		 * 	first_participant(context_t ctx)
		 * 		: so_5::agent_t{std::move(ctx)}
		 * 		, broadcast_mbox_{so_environment().create_mbox("message-board")}
		 * 	{}
		 * 	...
		 * };
		 *
		 * class second_participant final : public so_5::agent_t {
		 * 	const so_5::mbox_t broadcast_mbox_;
		 * 	...
		 * public:
		 * 	second_participant(context_t ctx)
		 * 		: so_5::agent_t{std::move(ctx)}
		 * 		, broadcast_mbox_{so_environment().create_mbox("message-board")}
		 * 	{}
		 * 	...
		 * };
		 * \endcode
		 * In this example both agents will use the same mbox instance.
		 *
		 * \attention
		 * Mboxes created by this method live in a separate namespace,
		 * they are not mixed with named mboxes introduced via
		 * introduce_named_mbox() method.
		 */
		[[nodiscard]]
		mbox_t
		create_mbox(
			//! Mbox name.
			nonempty_name_t mbox_name );

		/*!
		 * \brief Introduce named mbox with user-provided factory.
		 *
		 * This method allows a user to create own named mbox.
		 *
		 * The create_mbox(nonempty_name_t) method always creates a standard
		 * MPMC mbox. This isn't always desirable. For example, a user may want
		 * to have a named unique_subscribers mbox. The introduce_named_mbox()
		 * allows to achieve this:
		 * \code
		 * class first_participant final : public so_5::agent_t {
		 * 	const so_5::mbox_t broadcast_mbox_;
		 * 	...
		 * public:
		 * 	first_participant(context_t ctx)
		 * 		: so_5::agent_t{std::move(ctx)}
		 * 		, broadcast_mbox_{so_environment().introduce_named_mbox(
		 * 				so_5::mbox_namespace_name_t{"demo"},
		 * 				"message-board",
		 * 				[this]() { return so_5::make_unique_subscribers_mbox(so_environment()); } )
		 * 			}
		 * 	{}
		 * 	...
		 * };
		 *
		 * class second_participant final : public so_5::agent_t {
		 * 	const so_5::mbox_t broadcast_mbox_;
		 * 	...
		 * public:
		 * 	second_participant(context_t ctx)
		 * 		: so_5::agent_t{std::move(ctx)}
		 * 		, broadcast_mbox_{so_environment().introduce_named_mbox(
		 * 				so_5::mbox_namespace_name_t{"demo"},
		 * 				"message-board",
		 * 				[this]() { return so_5::make_unique_subscribers_mbox(so_environment()); } )
		 * 			}
		 * 	{}
		 * 	...
		 * };
		 * \endcode
		 *
		 * The `introduce_named_mbox` work the following way:
		 *
		 *
		 * - it checks for the existence of a mbox with the given name in the
		 *   specified namespace. If such a mbox is found it will be returned and
		 *   \a mbox_factory is not called;
		 * - otherwise, the \a mbox_factory is called;
		 * - after the completion of \a mbox_factory the existence of a mbox with
		 *   the given name is checked again;
		 * - if there is still no such a mbox, then the mbox obtained from \a
		 *   mbox_factory is stored inside the SOEnv and returned as the result
		 *   of the `introduce_named_mbox` method;
		 * - but if a mbox is found (it may have been created a little earlier by
		 *   a parallel call to the `introduce_named_mbox`), then the the value
		 *   returned by \a mbox_factory is discarded and the existing named mbox
		 *   is returned.
		 *
		 * It means then the \a mbox_factory is used if there is no a mbox with
		 * the name specified yet. If such a mbox is already exists then the
		 * already created mbox will be returned without calling \a mbox_factory.
		 *
		 * Note that named mboxes created by this method live in separate
		 * namespaces and those namespaces aren't intersect with "the default"
		 * namespace used by the create_mbox(nonempty_name_t) method. It means
		 * that a user can create named mboxes with the same name in different
		 * namespaces and they will be different mboxes:
		 * \code
		 * so_5::environment_t & env = ...;
		 * auto std_mbox = env.create_mbox("alice");
		 * auto my_mbox1 = env.introduce_named_mbox(
		 * 	so_5::mbox_namespace_name_t{"a"},
		 * 	"alice",
		 * 	[&env]() { return env.create_mbox(); });
		 * auto my_mbox2 = env.introduce_named_mbox(
		 * 	so_5::mbox_namespace_name_t{"b"},
		 * 	"alice",
		 * 	[&env]() { return env.create_mbox(); });
		 * // NOTE: it's just a reference to my_mbox2.
		 * auto my_duplicate = env.introduce_named_mbox(
		 * 	so_5::mbox_namespace_name_t{"b"},
		 * 	"alice",
		 * 	[&env]() { return env.create_mbox(); });
		 *
		 * assert(std_mbox->id() != my_mbox1->id());
		 * assert(std_mbox->id() != my_mbox2->id());
		 * assert(my_mbox1->id() != my_mbox2->id());
		 * assert(my_mbox2->id() == my_duplicate->id());
		 * \endcode
		 *
		 * \attention
		 * The \a mbox_factory should return a valid so_5::mbox_t.
		 * If \a mbox_factory returns empty so_5::mbox_t (nullptr), then
		 * introduce_named_mbox() will throw an exception with
		 * so_5::rc_nullptr_as_result_of_user_mbox_factory error code.
		 * If \a mbox_factory can't create a valid so_5::mbox_t it
		 * has to thrown a user-defined exception. This exception
		 * will be let out from introduce_named_mbox(). For example:
		 * \code
		 * so_5::environment_t & env = ...;
		 * try
		 * {
		 * 	auto mbox = env.introduce_named_mbox(
		 * 			so_5::mbox_namespace_name_t{"b"},
		 * 			"alice",
		 * 			[&]() {
		 * 				auto mbox = try_to_get_new_mbox();
		 * 				if(!mbox)
		 * 					throw my_exception{...};
		 * 				return mbox;
		 * 			});
		 * }
		 * catch( const my_exception & x ) { ... }
		 * \endcode
		 *
		 * \note
		 * If several parallel calls to introduce_named_mbox() with the same
		 * parameters are made at the same time then \a mbox_factory can
		 * be called several times (at most once for every introduce_named_mbox()
		 * invocation). But only one result of those calls will be used,
		 * all other returned values will be discarded.
		 *
		 * \attention
		 * The \a mbox_factory can be called in parallel (so \a mbox_factory
		 * should support this behavior).
		 * The \a mbox_factory can safely call environment_t's methods
		 * like create_mbox() and so on.
		 *
		 * \since v.5.8.0
		 */
		[[nodiscard]]
		mbox_t
		introduce_named_mbox(
			//! Name of mbox_namespace for a new mbox.
			mbox_namespace_name_t mbox_namespace,
			//! Name for a new mbox.
			nonempty_name_t mbox_name,
			//! Factory for new mbox.
			const std::function< mbox_t() > & mbox_factory );

		/*!
		 * \}
		 */

		/*!
		 * \name Method for working with message chains.
		 * \{
		 */

		/*!
		 * \brief Create message chain.
		 *
		 * \par Usage examples:
			\code
			so_5::environment_t & env = ...;
			// Create mchain with size-unlimited queue.
			auto ch1 = env.create_mchain(
				so_5::make_unlimited_mchain_params() );
			// Create mchain with size-limited queue without a timeout
			// on attempt to push another message to full mchain...
			auto ch2 = env.create_mchain(
				so_5::make_limited_without_waiting_mchain_params(
					// ...maximum size of the chain.
					100,
					// ...memory for chain will be allocated and deallocated dynamically...
					so_5::mchain_props::memory_usage_t::dynamic,
					// ...an exception will be thrown on overflow.
					so_5::mchain_props::overflow_reaction_t::throw_exception ) );
			// Create mchain with size-limited queue with a timeout for 200ms
			// on attempt to push another message to full mchain...
			auto ch3 = env.create_mchain(
				so_5::make_limited_with_waiting_mchain_params(
					// ...maximum size of the chain.
					100,
					// ...memory for chain will be preallocated...
					so_5::mchain_props::memory_usage_t::preallocated,
					// ...an oldest message from mchain will be removed on overflow...
					so_5::mchain_props::overflow_reaction_t::remove_oldest,
					// ...timeout for waiting on attempt to push a message into full mchain.
					std::chrono::milliseconds(200) ) );
			// Create size-unlimited mchain with custom notificator for
			// 'not_empty' situations.
			auto ch4 = env.create_mchain(
				so_5::make_unlimited_mchain_params().not_empty_notificator(
					[&] { some_widget.send_notify(); } ) );
			\endcode
		 *
		 * \since v.5.5.13
		 */
		mchain_t
		create_mchain(
			//! Parameters for a new bag.
			const mchain_params_t & params );
		/*!
		 * \}
		 */

		/*!
		 * \name Method for working with dispatchers.
		 * \{
		 */

		//! Set up an exception logger.
		void
		install_exception_logger(
			event_exception_logger_unique_ptr_t logger );
		/*!
		 * \}
		 */

		/*!
		 * \name Methods for working with cooperations.
		 * \{
		 */

		//! Create a cooperation.
		/*!
		 * Usage example:
		 * \code
		 * so_5::environmet_t & env = ...;
		 * // A binder for the default dispatcher will be used.
		 * auto coop = env.make_coop();
		 * coop.make_agent<first_agent>(...);
		 * coop.make_agent<second_agent>(...);
		 * // Registration of the coop.
		 * env.register_coop(std::move(coop));
		 * \endcode
		 *
		 * \return A new cooperation. This cooperation
		 * will use default dispatcher binders.
		 *
		 * \since v.5.6.0
		 */
		[[nodiscard]]
		coop_unique_holder_t
		make_coop();

		//! Create a cooperation with specified dispatcher binder.
		/*!
		 * A binder \a disp_binder will be used for binding cooperation
		 * agents to the dispatcher. This binder will be default binder for
		 * this cooperation.
		 *
			\code
			so_5::environment_t & so_env = ...;
			so_5::coop_unique_holder_t coop = so_env.make_coop(
				so_5::disp::active_group::make_dispatcher( so_env ).binder(
					"some_active_group" ) );

			// That agent will be bound to the dispatcher "active_group"
			// and will be member of an active group with name
			// "some_active_group".
			coop->make_agent< a_some_agent_t >();
			\endcode
		 *
		 * \since v.5.6.0
		 */
		[[nodiscard]]
		coop_unique_holder_t
		make_coop(
			//! A default binder for this cooperation.
			disp_binder_shptr_t disp_binder );

		/*!
		 * \brief Create a new cooperation that will be a child for
		 * specified parent coop.
		 *
		 * The new cooperation will use the default dispatcher binder.
		 *
		 * Usage example:
		 * \code
		 * class parent_t final : public so_5::agent_t {
		 * 	void on_some_event(mhood_t<some_message>) {
		 * 		// We need to create a child coop.
		 * 		auto coop = so_environment().make_coop(
		 * 				// We as a parent.
		 * 				so_coop());
		 * 		...
		 * 	}
		 * };
		 * \endcode
		 *
		 * \since v.5.6.0
		 */
		[[nodiscard]]
		coop_unique_holder_t
		make_coop(
			//! Parent coop.
			coop_handle_t parent );

		/*!
		 * \brief Create a new cooperation that will be a child for
		 * specified parent coop.
		 *
		 * The new cooperation will use the specified dispatcher binder.
		 *
		 * Usage example:
		 * \code
		 * class parent_t final : public so_5::agent_t {
		 * 	void on_some_event(mhood_t<some_message>) {
		 * 		// We need to create a child coop.
		 * 		auto coop = so_environment().make_coop(
		 * 				// We as a parent.
		 * 				so_coop(),
		 * 				// The default dispatcher for the new coop.
		 * 				so_5::disp::active_obj::make_dispatcher(
		 * 						so_environment() ).binder() );
		 * 		...
		 * 	}
		 * };
		 * \endcode
		 *
		 * \since v.5.6.0
		 */
		[[nodiscard]]
		coop_unique_holder_t
		make_coop(
			//! Parent coop.
			coop_handle_t parent,
			//! A default binder for this cooperation.
			disp_binder_shptr_t disp_binder );

		//! Register a cooperation.
		/*!
		 * The cooperation registration includes following steps:
		 *
		 * - binding agents to the cooperation object;
		 * - checking uniques of the cooperation name. The cooperation will 
		 *   not be registered if its name isn't unique;
		 * - agent_t::so_define_agent() will be called for each agent
		 *   in the cooperation;
		 * - binding of each agent to the dispatcher.
		 *
		 * If all these actions are successful then the cooperation is
		 * marked as registered.
		 *
		 * \par Usage examples
		 *
		 * Very simple case.
		 * \code
		 * so_5::environment_t & env = ...;
		 *
		 * auto simple_coop = env.make_coop();
		 * simple_coop->make_agent<some_agent_type>(...);
		 * env.register_coop(std::move(simple_coop));
		 * \endcode
		 *
		 * More complex case with storing coop_handle and using it for
		 * coop deregistration:
		 * \code
		 * so_5::environment_t & env = ...;
		 * so_5::coop_handle_t coop_handle;
		 *
		 * auto simple_coop = env.make_coop();
		 * simple_coop->make_agent<some_agent_type>(...);
		 * coop_handle = env.register_coop(std::move(simple_coop));
		 * ...
		 * // Some time later.
		 * env.deregister_coop(coop_handle, so_5::dereg_reason::normal);
		 * \endcode
		 *
		 * A typical scenario for register_coop() when an instance of coop
		 * is created by a separate function:
		 * \code
		 * [[nodiscard]] so_5::coop_unique_holder_t create_coop(
		 * 		so_5::environment_t & env)
		 * {
		 * 	auto coop = env.make_coop();
		 * 	coop->make_agent<some_agent_type>(...);
		 * 	... // Some other actions like creation of additional agents.
		 *
		 * 	// Now the coop can be returned back.
		 * 	return coop;
		 * }
		 *
		 * so_5::environment_t & env = ...;
		 * ...
		 * env.register_coop(create_coop(env));
		 * \endcode
		 */
		coop_handle_t
		register_coop(
			//! Cooperation to be registered.
			coop_unique_holder_t agent_coop );

		/*!
		 * \brief Register single agent as a cooperation.
		 *
		 * It is just a helper methods for convience.
		 *
		 * Usage sample:
		 * \code
		   std::unique_ptr< my_agent > a( new my_agent(...) );
		   so_env.register_agent_as_coop( std::move(a) );
		 * \endcode
		 */
		template< class A >
		coop_handle_t
		register_agent_as_coop(
			std::unique_ptr< A > agent )
		{
			auto coop = make_coop();
			coop->add_agent( std::move( agent ) );
			return register_coop( std::move( coop ) );
		}

		/*!
		 * \brief Register single agent as a cooperation with specified
		 * dispatcher binder.
		 *
		 * It is just a helper methods for convience.
		 *
		 * Usage sample:
		 * \code
		 * so_5::environment_t & env = ...;
		 * env.register_agent_as_coop(
		 * 		std::make_unique<my_agent>(...),
		 * 		so_5::disp::active_group::make_dispatcher(env)
		 * 				.binder("some_active_group") );
		 * \endcode
		 *
		 * \since v.5.2.1
		 */
		template< class A >
		coop_handle_t
		register_agent_as_coop(
			std::unique_ptr< A > agent,
			disp_binder_shptr_t disp_binder )
		{
			auto coop = make_coop( std::move( disp_binder ) );
			coop->add_agent( std::move( agent ) );
			return register_coop( std::move( coop ) );
		}

		//! Deregister the cooperation.
		/*!
		 * Method searches the cooperation within registered cooperations and if
		 * it is found deregisters it.
		 *
		 * Deregistration can take some time.
		 *
		 * At first a special signal is sent to cooperation agents.
		 * By receiving these signal agents stop receiving new messages.
		 * When the local event queue for an agent becomes empty the 
		 * agent informs the cooperation about this. When the cooperation 
		 * receives all these signals from agents it informs 
		 * the SObjectizer Run-Time.
		 *
		 * Only after this the cooperation is deregistered on the special
		 * context.
		 *
		 * After the cooperation deregistration agents are unbound from
		 * dispatchers.
		 *
		 * Usage example:
		 * \code
		 * so_5::environment_t & env = ...;
		 * so_5::coop_handle_t coop_handle;
		 *
		 * auto simple_coop = env.make_coop();
		 * simple_coop->make_agent<some_agent_type>(...);
		 * coop_handle = env.register_coop(std::move(simple_coop));
		 * ...
		 * // Some time later.
		 * env.deregister_coop(coop_handle, so_5::dereg_reason::normal);
		 * \endcode
		 *
		 * \note
		 * This method is marked as noexcept because there is no way
		 * to recover if any exception is raised here.
		 */
		void
		deregister_coop(
			//! The coop to be deregistered.
			coop_handle_t coop,
			//! Deregistration reason.
			int reason ) noexcept
		{
			auto coop_shptr = low_level_api::to_shptr_noexcept( coop );
			if( coop_shptr )
				coop_shptr->deregister( reason );
		}
		/*!
		 * \}
		 */

		/*!
		 * \name Methods for working with layers.
		 * \{
		 */

		//! Get access to the layer without raising exception if layer
		//! is not found.
		template< class SO_Layer >
		SO_Layer *
		query_layer_noexcept() const
		{
			static_assert( std::is_base_of< layer_t, SO_Layer >::value,
					"SO_Layer must be derived from so_layer_t class" );

			return dynamic_cast< SO_Layer * >(
					query_layer( std::type_index( typeid( SO_Layer ) ) ) );
		}

		//! Get access to the layer with exception if layer is not found.
		template< class SO_Layer >
		SO_Layer *
		query_layer() const
		{
			auto layer = query_layer_noexcept< SO_Layer >();

			if( !layer )
				SO_5_THROW_EXCEPTION(
					rc_layer_does_not_exist,
					"layer does not exist" );

			return layer;
		}

		//! Add an additional layer.
		template< class SO_Layer >
		void
		add_extra_layer(
			std::unique_ptr< SO_Layer > layer_ptr )
		{
			add_extra_layer(
				std::type_index( typeid( SO_Layer ) ),
				layer_ref_t( layer_ptr.release() ) );
		}
		/*!
		 * \}
		 */

		/*!
		 * \name Methods for starting, initializing and stopping of the Run-Time.
		 * \{
		 */

		//! Run the SObjectizer Run-Time.
		void
		run();

		//! Initialization hook.
		/*!
		 * \attention A hang inside of this method will prevent the Run-Time
		 * from stopping. For example if a dialog with an application user
		 * is performed inside init() then SObjectizer cannot finish
		 * its work until this dialog is finished.
		 */
		virtual void
		init() = 0;

		//! Send a shutdown signal to the Run-Time.
		/*!
		 * \note
		 * This method is noexcept since v.5.8.0.
		 */
		void
		stop() noexcept;
		/*!
		 * \}
		 */

		/*!
		 * \brief Call event exception logger for logging an exception.
		 *
		 * \note
		 * Since v.5.6.0 this method is marked as noexcept.
		 *
		 * \since v.5.2.3
		 */
		void
		call_exception_logger(
			//! Exception caught.
			const std::exception & event_exception,
			//! A cooperation to which agent is belong.
			const coop_handle_t & coop ) noexcept;

		/*!
		 * \brief An exception reaction for the whole SO Environment.
		 *
		 * \note
		 * This method is noexcept since v.5.8.0.
		 *
		 * \since v.5.3.0
		 */
		exception_reaction_t
		exception_reaction() const noexcept;

		/*!
		 * \brief Get the error_logger object.
		 *
		 * \since v.5.5.0
		 */
		error_logger_t &
		error_logger() const;

		/*!
		 * \brief Helper method for simplification of agents creation.
		 *
		 * \since v.5.5.4
		 *
		 * \note Creates an instance of agent of type \a Agent by using
		 * environment_t::make_agent() template function and adds it to
		 * the cooperation. Uses the fact that most agent types use reference
		 * to the environment object as the first argument.
		 *
		 * \return unique pointer to the new agent.
		 *
		 * \tparam Agent type of agent to be created.
		 * \tparam Args type of parameters list for agent constructor.
		 *
		 * \par Usage sample:
		 \code
		 so_5::environment_t & env = ...;
		 // For the case of constructor like my_agent(environmen_t&).
		 auto a1 = env.make_agent< my_agent >(); 
		 // For the case of constructor like your_agent(environment_t&, std::string).
		 auto a2 = env.make_agent< your_agent >( "hello" );
		 // For the case of constructor like their_agent(environment_t&, std::string, mbox_t).
		 auto a3 = env.make_agent< their_agent >( "bye", a2->so_direct_mbox() );
		 \endcode
		 */
		template< class Agent, typename... Args >
		std::unique_ptr< Agent >
		make_agent( Args &&... args )
		{
			return std::unique_ptr< Agent >(
					new Agent( *this, std::forward<Args>(args)... ) );
		}

		/*!
		 * \brief Access to controller of run-time monitoring.
		 *
		 * \since v.5.5.4
		 */
		stats::controller_t &
		stats_controller();

		/*!
		 * \brief Access to repository of data sources for run-time monitoring.
		 *
		 * \since v.5.5.4
		 */
		stats::repository_t &
		stats_repository();

		/*!
		 * \brief Access to the global work thread factory.
		 *
		 * \since v.5.7.3
		 */
		[[nodiscard]]
		so_5::disp::abstract_work_thread_factory_shptr_t
		work_thread_factory() const noexcept;

		/*!
		 * \brief Helper method for simplification of cooperation creation
		 * and registration.
		 *
		 * \return The value returned from lambda-function. Or void if
		 * the lambda-function returns void.
		 *
		 * \since v.5.5.5
		 *
		 * \par Usage samples:
			\code
			// The default dispatcher will be used for binding.
			env.introduce_coop( []( so_5::coop_t & coop ) {
				coop.make_agent< first_agent >(...);
				coop.make_agent< second_agent >(...);
			});

			// For the case when dispatcher binder is specified.
			env.introduce_coop(
				so_5::disp::active_obj::make_dispatcher( env ).binder(),
				[]( so_5::coop_t & coop ) {
					coop.make_agent< first_agent >(...);
					coop.make_agent< second_agent >(...);
				} );

			// Usage of return value from the lambda function.
			so_5::mbox_t mbox = env.introduce_coop( [](so_5::coop_t & coop) {
					auto * a = coop.make_agent< first_agent >(...);
					coop.make_agent< second_agent >(...);

					return a->so_direct_mbox();
				});
			\endcode
		 */
		template< typename... Args >
		decltype(auto)
		introduce_coop( Args &&... args );

		/*!
		 * \brief Get activity tracking flag for the whole SObjectizer Environment.
		 *
		 * \since v.5.5.18
		 */
		work_thread_activity_tracking_t
		work_thread_activity_tracking() const;

		/*!
		 * \brief Get binding to the default dispatcher.
		 *
		 * \note
		 * This method is part of environment_t for possibility to
		 * write custom implementations of environment_infrastructure_t.
		 * Because of that this method can be changed or removed in 
		 * future versions of SObjectizer.
		 *
		 * \since v.5.5.19
		 */
		disp_binder_shptr_t
		so_make_default_disp_binder();

		/*!
		 * \brief Get autoshutdown_disabled flag.
		 *
		 * Autoshutdown feature is on by default. It can be turned off
		 * in environment_params_t. This methods returns <i>true</i> if
		 * autoshutdown is turned off.
		 *
		 * \since v.5.5.19
		 */
		bool
		autoshutdown_disabled() const;

		//! Schedule timer event.
		/*!
		 * \attention
		 * Values of \a pause and \a period should be non-negative.
		 *
		 * \note
		 * This method is a part of low-level SObjectizer's API.
		 * Because of that it can be changed or removed in some
		 * future version of SObjectizer without prior notice.
		 */
		so_5::timer_id_t
		so_schedule_timer(
			//! Parameters for new timer.
			const low_level_api::schedule_timer_params_t params );

		//! Schedule a single shot timer event.
		/*!
		 * \attention
		 * Value of \a pause should be non-negative.
		 *
		 * \note
		 * This method is a part of low-level SObjectizer's API.
		 * Because of that it can be changed or removed in some
		 * future version of SObjectizer without prior notice.
		 */
		void
		so_single_timer(
			//! Parameters for new timer.
			const low_level_api::single_timer_params_t params );

		//! Create a custom mbox.
		/*!
		 * \tparam Lambda type of actual lambda with all creation actions.
		 * The Lambda must be lambda-function or functional objects with
		 * the following format:
		 * \code
		 * so_5::mbox_t lambda(const so_5::mbox_creation_data_t &);
		 * \endcode
		 *
		 * \since v.5.5.19.2
		 */
		template< typename Lambda >
		mbox_t
		make_custom_mbox( Lambda && lambda )
			{
				using namespace custom_mbox_details;

				creator_template_t< Lambda > creator( std::forward<Lambda>(lambda) );
				return do_make_custom_mbox( creator );
			}

		/*!
		 * \name Methods for working with stop_guards.
		 * \{
		 */
		//! Set up a new stop_guard.
		/*!
		 * Usage examples:
		 * \code
		 * // Add a stop_guard.
		 * // Note: an exception can be thrown if stop is in progress
		 * class my_stop_guard
		 * 	: public so_5::stop_guard_t
		 * 	, public std::enable_shared_from_this< my_stop_guard >
		 * {...};
		 *
		 * class my_agent : public so_5::agent_t
		 * {
		 * 	...
		 * 	void on_some_event()
		 * 	{
		 * 		// We need a stop_guard here.
		 * 		m_my_guard = std::make_shared< my_stop_guard >(...);
		 * 		so_environment().setup_stop_guard( m_my_guard );
		 * 	}
		 * private :
		 * 	so_5::stop_guard_shptr_t m_my_guard;
		 * };
		 *
		 * //
		 * // Add a stop_guard without throwing an exception if stop is in progress
		 * //
		 * class my_stop_guard
		 * 	: public so_5::stop_guard_t
		 * 	, public std::enable_shared_from_this< my_stop_guard >
		 * {...};
		 *
		 * class my_agent : public so_5::agent_t
		 * {
		 * 	...
		 * 	void on_some_event()
		 * 	{
		 * 		// We need a stop_guard here.
		 * 		m_my_guard = std::make_shared< my_stop_guard >(...);
		 * 		const auto r = so_environment().setup_stop_guard(
		 * 				m_my_guard,
		 * 				so_5::stop_guard_t::what_if_stop_in_progress_t::return_negative_result );
		 *			if( so_5::stop_guard_t::setup_result_t::stop_already_in_progress  == r )
		 *				... // handle error here.
		 * 	}
		 * private :
		 * 	so_5::stop_guard_shptr_t m_my_guard;
		 * };
		 * \endcode
		 *
		 * \note
		 * Uniqueness of stop_guard is not checked. It means that
		 * it is possible to add the same stop_guard several times.
		 * But it seems to be useless.
		 *
		 * \since v.5.5.19.2
		 */
		stop_guard_t::setup_result_t
		setup_stop_guard(
			//! Stop guard to be set.
			//! Should not be nullptr.
			stop_guard_shptr_t guard,
			//! What to do is the stop operation is already in progress?
			stop_guard_t::what_if_stop_in_progress_t reaction_on_stop_in_progress
				= stop_guard_t::what_if_stop_in_progress_t::throw_exception );

		//! Remove stop_guard and complete the stop operation if necessary.
		/*!
		 * Every stop_guard which was added to the environment must be
		 * explicitely removed from the environment. It is done by this method.
		 * If there is no more stop_guard and the stop operation is in progress
		 * then the environment will complete the stop operation.
		 *
		 * Usage examples:
		 * \code
		 * // Note: an exception can be thrown if stop is in progress
		 * class my_stop_guard
		 * 	: public so_5::stop_guard_t
		 * 	, public std::enable_shared_from_this< my_stop_guard >
		 * {...};
		 *
		 * class my_agent : public so_5::agent_t
		 * {
		 * 	...
		 * 	void on_some_event()
		 * 	{
		 * 		// We need a stop_guard here.
		 * 		m_my_guard = std::make_shared< my_stop_guard >(...);
		 * 		so_environment().setup_stop_guard( m_my_guard );
		 * 	}
		 *
		 * 	void on_work_finished_signal()
		 * 	{
		 * 		// Stop_guard must be removed now.
		 * 		so_environment().remove_stop_guard( m_my_guard );
		 * 	}
		 * private :
		 * 	so_5::stop_guard_shptr_t m_my_guard;
		 * };
		 * \endcode
		 *
		 * \since v.5.5.19.2
		 */
		void
		remove_stop_guard(
			//! Stop guard to be removed.
			stop_guard_shptr_t guard );
		/*!
		 * \}
		 */

		/*!
		 * \name Methods for working with msg_tracing's filters.
		 * \{
		 */
		/*!
		 * \brief Change the current msg_tracing's filter to a new one.
		 *
		 * Usage example:
		 * \code
		 * so_5::launch([](so_5::environment_t & env) {...},
		 * 	[](so_5::environment_params_t & params) {
		 * 		// Turn message delivery tracing on.
		 * 		params.message_delivery_tracer(
		 * 			so_5::msg_tracing::std_cout_tracer());
		 * 		// Disable all trace messages.
		 * 		// It is expected that trace filter will be changed in the future.
		 * 		params.message_delivery_tracer_filter(
		 * 			so_5::msg_tracing::make_disable_all_filter());
		 * 		...
		 * 	} );
		 * ...
		 * void some_agent_t::turn_msg_tracing_on() {
		 * 	// Remove trace filter. As result all trace messages will be printed.
		 * 	so_environment().change_message_delivery_tracer_filter(
		 * 		so_5::msg_tracing::no_filter());
		 * 	...
		 * }
		 * \endcode
		 *
		 * \note
		 * It is possible that there are active calls to
		 * so_5::msg_tracing::filter_t::filter() methods at the time of
		 * invocation of change_message_delivery_tracer_filter(). In this
		 * case all active calls will be completed with the previous
		 * filter. This could lead to mixture of messages in the trace:
		 * some of them will be enabled by old filter and some of them
		 * will be enabled by new filter. And it is possible that
		 * messages enabled by new filter will precede messages enabled
		 * by old filter.
		 *
		 * \throw exception_t if message delivery tracing is disabled.
		 *
		 * \since v.5.5.22
		 */
		void
		change_message_delivery_tracer_filter(
			//! A new filter to be used.
			//! It can be an empty pointer. In this case all trace messages
			//! will be passed to tracer object.
			so_5::msg_tracing::filter_shptr_t filter );
		/*!
		 * \}
		 */

	private:
		//! Access to an additional layer.
		layer_t *
		query_layer(
			const std::type_index & type ) const;

		//! Add an additional layer.
		void
		add_extra_layer(
			const std::type_index & type,
			const layer_ref_t & layer );

		//! Remove an additional layer.
		void
		remove_extra_layer(
			const std::type_index & type );

		//! Actual creation of a custom mbox.
		/*!
		 * \since v.5.5.19.2
		 */
		mbox_t
		do_make_custom_mbox(
			custom_mbox_details::creator_iface_t & creator );

		struct internals_t;

		//! SObjectizer Environment internals.
		std::unique_ptr< internals_t > m_impl;

		/*!
		 * \name Implementation details related to run/stop functionality.
		 * \{
		 */
		/*!
		 * \brief Run controller for run-time monitoring
		 * and call next run stage.
		 *
		 * \since v.5.5.4
		 */
		void
		imp_run_stats_controller_and_go_further();

		/*!
		 * \brief Run layers and call next run stage.
		 */
		void
		imp_run_layers_and_go_further();

		/*!
		 * \brief Launch environment infrastructure and wait for finish.
		 *
		 * \since v.5.5.19
		 */
		void
		imp_run_infrastructure();
		/*!
		 * \}
		 */
};

namespace details
{

/*!
 * \brief Helper class for building and registering new cooperation.
 *
 * \since v.5.5.5
 */
class introduce_coop_helper_t
{
private :
	//! Environment for creation of cooperation.
	environment_t & m_env;
	//! Optional parent coop.
	coop_handle_t m_parent;

	template< typename Lambda >
	decltype(auto)
	build_and_register_coop(
		disp_binder_shptr_t binder,
		Lambda && lambda )
	{
		const auto coop_maker = [this, b = std::move(binder)] {
			if( m_parent )
				return m_env.make_coop( m_parent, std::move(b) );
			else
				return m_env.make_coop( std::move(b) );
		};

		auto coop = coop_maker();

		using return_type = std::invoke_result_t<Lambda, so_5::coop_t &>;

		if constexpr( std::is_void_v<return_type> )
		{
			lambda( *coop );
			m_env.register_coop( std::move( coop ) );
		}
		else if constexpr( std::is_reference_v<return_type> )
		{
			auto && ret_val = lambda( *coop );
			m_env.register_coop( std::move( coop ) );

			return std::forward<decltype(ret_val)>(ret_val);
		}
		else
		{
			auto ret_val = lambda( *coop );
			m_env.register_coop( std::move( coop ) );

			return ret_val;
		}
	}

public :
	//! Constructor for the case of creation a cooperation without parent.
	introduce_coop_helper_t( environment_t & env )
		:	m_env{ env }
	{}
	//! Constructor for the case of creation a cooperation with parent.
	introduce_coop_helper_t(
		environment_t & env,
		coop_handle_t parent )
		:	m_env{ env }
		,	m_parent{ std::move(parent) }
	{}

	/*!
	 * For the case:
	 * - default dispatcher is used.
	 */
	template< typename L >
	decltype(auto)
	introduce( L && lambda )
	{
		return build_and_register_coop(
				m_env.so_make_default_disp_binder(),
				std::forward< L >( lambda ) );
	}

	/*!
	 * For the case:
	 * - dispatcher builder is specified.
	 */
	template< typename L >
	decltype(auto)
	introduce( disp_binder_shptr_t binder, L && lambda )
	{
		return build_and_register_coop(
				std::move( binder ),
				std::forward< L >( lambda ) );
	}
};

} /* namespace details */

template< typename... Args >
decltype(auto)
environment_t::introduce_coop( Args &&... args )
{
	details::introduce_coop_helper_t helper{ *this };
	return helper.introduce( std::forward< Args >( args )... );
}

/*!
 * \brief A simple way for creating child cooperation.
 *
 * \since v.5.5.3
 *
 * \par Usage sample
	\code
	class owner : public so_5::agent_t
	{
	public :
		...
		virtual void
		so_evt_start() override
		{
			auto child = so_5::create_child_coop( *this );
			child->make_agent< worker >();
			...
			so_environment().register_coop( std::move( child ) );
		}
	};
	\endcode
 */
template< typename... Args >
[[nodiscard]]
coop_unique_holder_t
create_child_coop(
	//! Owner of the cooperation.
	agent_t & owner,
	//! Arguments for the environment_t::make_coop() method.
	Args&&... args )
{
	auto coop = owner.so_environment().make_coop(
			owner.so_coop(),
			std::forward< Args >(args)... );

	return coop;
}

/*!
 * \brief A simple way for creating child cooperation when there is
 * a reference to the parent cooperation object.
 *
 * \since v.5.5.8
 *
 * \par Usage sample
	\code
	class parent final : public so_5::agent_t {
		void on_some_event(mhood_t<some_message>) {
			auto child = so_5::create_child_coop(
					// We as the parent coop.
					so_coop(),
					// The default binder for the new coop.
					so_5::disp::one_thread::make_dispatcher(
							so_environment().binder() ) );
			...
			so_environment().register_coop( std::move(child) );
		}
		...
	};
	\endcode
 */
template< typename... Args >
[[nodiscard]]
coop_unique_holder_t
create_child_coop(
	//! Parent cooperation.
	coop_handle_t parent,
	//! Arguments for the environment_t::make_coop() method.
	Args&&... args )
{
	return low_level_api::to_shptr(parent)->environment().make_coop(
			parent,
			std::forward< Args >(args)... );
}

/*!
 * \brief A simple way for creating and registering child cooperation.
 *
 * \since v.5.5.5
 *
 * \par Usage sample
	\code
	class owner : public so_5::agent_t
	{
	public :
		...
		virtual void
		so_evt_start() override
		{
			so_5::introduce_child_coop( *this, []( so_5::coop_t & coop ) {
				coop.make_agent< worker >();
			} );
		}
	};
	\endcode

 * \note This function is just a tiny wrapper around
 * so_5::environment_t::introduce_coop() helper method. For more
 * examples with usage of introduce_coop() please see description of
 * that method.
 */
template< typename... Args >
decltype(auto)
introduce_child_coop(
	//! Owner of the cooperation.
	agent_t & owner,
	//! Arguments for the environment_t::introduce_coop() method.
	Args&&... args )
{
	return details::introduce_coop_helper_t{
					owner.so_environment(),
					owner.so_coop()
			}.introduce( std::forward< Args >(args)... );
}

/*!
 * \brief A simple way for creating and registering child cooperation
 * when there is a reference to parent coop.
 *
 * \since v.5.5.8
 *
 * \par Usage sample
	\code
	class owner : public so_5::agent_t
	{
	public :
		...
		virtual void
		so_evt_start() override
		{
			so_5::introduce_child_coop( so_coop(), []( so_5::coop_t & coop ) {
				coop.make_agent< worker >();
			} );
		}
	};
	\endcode

 * \note This function is just a tiny wrapper around
 * so_5::environment_t::introduce_coop() helper method. For more
 * examples with usage of introduce_coop() please see description of
 * that method.
 */
template< typename... Args >
decltype(auto)
introduce_child_coop(
	//! Parent cooperation.
	coop_handle_t parent,
	//! Arguments for the environment_t::introduce_coop() method.
	Args&&... args )
{
	return details::introduce_coop_helper_t{
					low_level_api::to_shptr(parent)->environment(),
					parent
			}.introduce( std::forward< Args >(args)... );
}

//
// make_default_disp_binder
//
/*!
 * \brief Create an instance of the default dispatcher binder.
 *
 * \note This function takes into account a possibility to have
 * different types of environment infrastructures (introduced in v.5.5.19)
 * and creates a default dispatcher binder with respect to the
 * actual environment infrastructure type.
 *
 * Usage example:
 * \code
 * so_5::launch( [](so_5::environment_t & env) {
 * 	env.introduce_coop(
 * 		// Agents from that coop will be bound to the default dispatcher.
 * 		so_5::make_default_disp_binder(env),
 * 		[](so_5::coop_t & coop) {
 * 			coop.make_agent<...>(...);
 * 		} );
 * } );
 * \endcode
 *
 * \since v.5.5.19
 */
inline disp_binder_shptr_t
make_default_disp_binder( environment_t & env )
	{
		return env.so_make_default_disp_binder();
	}

namespace low_level_api
{

//! Schedule periodic timer event.
/*!
 * \note
 * This function is a part of low-level SObjectizer's interface.
 * Because of that this function can be removed or changed in some
 * future version without prior notice.
 *
 * \attention
 * Values of \a pause and \a period should be non-negative.
 *
 * \since v.5.6.0
 */
[[nodiscard]] inline so_5::timer_id_t
schedule_timer(
	//! Message type for searching subscribers.
	const std::type_index & subscription_type,
	//! Message to be sent after timeout.
	message_ref_t msg,
	//! Mbox to which message will be delivered.
	const mbox_t & mbox,
	//! Timeout before the first delivery.
	std::chrono::steady_clock::duration pause,
	//! Period of the delivery repetition for periodic messages.
	/*! 
		\note Value 0 indicates that it's not periodic message 
			(will be delivered one time).
	*/
	std::chrono::steady_clock::duration period )
{
	return mbox->environment().so_schedule_timer(
			schedule_timer_params_t{
					std::cref(subscription_type),
					std::cref(msg),
					std::cref(mbox),
					pause,
					period } );
}

//! Schedule single timer event.
/*!
 * \note
 * This function is a part of low-level SObjectizer's interface.
 * Because of that this function can be removed or changed in some
 * future version without prior notice.
 *
 * \attention
 * Value \a period should be non-negative.
 *
 * \since v.5.6.0
 */
inline void
single_timer(
	//! Message type for searching subscribers.
	const std::type_index & subscription_type,
	//! Message to be sent after timeout.
	message_ref_t msg,
	//! Mbox to which message will be delivered.
	const mbox_t & mbox,
	//! Timeout before the delivery.
	std::chrono::steady_clock::duration pause )
{
	return mbox->environment().so_single_timer(
			single_timer_params_t{
					std::cref(subscription_type),
					std::cref(msg),
					std::cref(mbox),
					pause } );
}

} /* namespace low_level_api */

} /* namespace so_5 */

#if defined( SO_5_MSVC )
	#pragma warning(pop)
#endif

