/*
 * SObjectizer-5
 */

/*!
 * \since
 * v.5.5.0
 *
 * \file
 * \brief Tools for logging error messages inside SObjectizer core.
 */

#pragma once

#include <so_5/declspec.hpp>
#include <so_5/compiler_features.hpp>

#include <memory>
#include <sstream>

namespace so_5
{

//
// error_logger_t
//
/*!
 * \since
 * v.5.5.0
 *
 * \brief An interface for logging error messages.
 */
class SO_5_TYPE error_logger_t
	{
		error_logger_t( const error_logger_t & ) = delete;
		error_logger_t &
		operator=( error_logger_t & ) = delete;

	public :
		error_logger_t() = default;
		virtual ~error_logger_t() noexcept = default;

		//! A method for logging message.
		/*!
		 * \note
		 * This method is not noexcept. It means that logger can throws,
		 * but in most cases logger is called in cases when the application
		 * is prepared to be aborted due to some fatal errors. In those cases
		 * an exception from log() will just terminate the application a bit
		 * earlier. In other cases exceptions thrown by log() will be ignored.
		 */
		virtual void
		log(
			//! Source file name.
			const char * file_name,
			//! Line number inside source file.
			unsigned int line,
			//! Text to log.
			const std::string & message ) = 0;
	};

//
// error_logger_shptr_t
//
/*!
 * \since
 * v.5.5.0
 *
 * \brief An alias for shared_ptr to error_logger.
 */
using error_logger_shptr_t = std::shared_ptr< error_logger_t >;

//
// create_stderr_logger
//
/*!
 * \since
 * v.5.5.0
 *
 * \brief A factory for creating error_logger implemenation which
 * uses std::stderr as log stream.
 */
SO_5_FUNC error_logger_shptr_t
create_stderr_logger();

namespace log_msg_details
{

class conductor_t
	{
	public :
		inline
		conductor_t(
			error_logger_t & logger,
			const char * file,
			unsigned int line )
			:	m_logger( logger )
			,	m_file( file )
			,	m_line( line )
			{}

		template< class Env >
		conductor_t( const Env & env,
			const char * file,
			unsigned int line )
			:	m_logger( env.error_logger() )
			,	m_file( file )
			,	m_line( line )
			{}

		inline bool
		completed() const { return m_completed; }

		inline std::ostringstream &
		stream() { return m_stream; }

		inline void
		log_message()
		{
			m_completed = true;
			m_logger.log( m_file, m_line, m_stream.str() );
		}

	private :
		error_logger_t & m_logger;
		const char * m_file;
		unsigned int m_line;
		bool m_completed = false;
		std::ostringstream m_stream;
	};

} /* namespace log_msg_details */

} /* namespace so_5 */

//
// SO_5_LOG_ERROR
//
/*!
 * \since
 * v.5.5.0
 *
 * \brief An implementation for SO_5_LOG_ERROR macro.
 */
#define SO_5_LOG_ERROR_IMPL(logger, file, line, var_name) \
	for( so_5::log_msg_details::conductor_t conductor_( logger, file, line ); \
			!conductor_.completed(); ) \
		for( std::ostringstream & var_name = conductor_.stream(); \
				!conductor_.completed(); conductor_.log_message() )

//
// SO_5_LOG_ERROR
//
/*!
 * \since
 * v.5.5.0
 *
 * \brief A special macro for helping error logging.
 *
 */
#define SO_5_LOG_ERROR(logger, var_name) \
	SO_5_LOG_ERROR_IMPL(logger, __FILE__, __LINE__, var_name )

