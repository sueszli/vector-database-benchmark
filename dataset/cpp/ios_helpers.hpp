/*
 * SObjectizer-5
 */

/*!
 * \since
 * v.5.5.12
 *
 * \file
 * \brief Various helpers for working with C++ iostreams.
 */

#pragma once

#include <string_view>
#include <ostream>
#include <cstdint>

namespace so_5 {

namespace details {

namespace ios_helpers {

/*!
 * \since
 * v.5.5.4
 *
 * \brief Helper for showing only part of long string.
 */
struct length_limited_string
	{
		const std::string_view m_what;
		const std::size_t m_limit;

		length_limited_string(
			const std::string_view what,
			std::size_t limit )
			:	m_what( what )
			,	m_limit( limit )
			{}
	};

inline std::ostream &
operator<<( std::ostream & to, const length_limited_string & v )
	{
		if( v.m_what.size() > v.m_limit )
			{
				const std::size_t median = v.m_limit / 2;

				to << v.m_what.substr( 0, median )
					<< "..."
					<< v.m_what.substr( v.m_what.size() - median + 3 );
			}
		else
			to << v.m_what;

		return to;
	}

/*!
 * \since
 * v.5.5.4
 *
 * \brief Helper for showing pointer value.
 *
 * \note
 * It's safe to pass a nullptr to `pointer` instance.
 */
struct pointer
	{
		const void * m_what;

		pointer( const void * what ) : m_what{ what } {}
	};

inline std::ostream &
operator<<( std::ostream & to, const pointer & v )
	{
		const auto oldf = to.setf( std::ios_base::hex, std::ios_base::basefield );
		to << "0x" << reinterpret_cast<std::uintptr_t>(v.m_what);
		to.setf( oldf, std::ios_base::basefield );

		return to;
	}

} /* namespace ios_helpers */

} /* namespace details */

} /* namespace so_5 */

