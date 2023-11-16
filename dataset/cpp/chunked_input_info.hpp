/*
	restinio
*/

/*!
 * @file
 * @brief Information about chunked encoded body.
 * @since v.0.6.9
 */

#pragma once

#include <restinio/exception.hpp>
#include <restinio/http_headers.hpp>

#include <restinio/impl/include_fmtlib.hpp>

#include <vector>
#include <memory>

namespace restinio
{

//
// chunk_ext_param_t
//

/**
 * @brief Chunk extension parameter.
 *
 * @since v.0.7.0
 */
struct chunk_ext_param_t
{
	std::string m_name;
	std::string m_value;
};

using chunk_ext_params_t = std::vector< chunk_ext_param_t >;
using chunk_ext_params_unique_ptr_t = std::unique_ptr< chunk_ext_params_t >;

//
// chunk_info_t
//
/*!
 * @brief Information about one chunk in an incoming request with
 * chunked encoding.
 *
 * In RESTinio v.0.6.9 all chunks are concatenated into the one body.
 * The information about individual chunks preserved in the form
 * of vector of chunk_info_t objects. Every object contains
 * the offset from the begining of the concatenated body and the size
 * of the chunk. This information allows to extract the corresponding
 * fragment from the whole body.
 *
 * @since v.0.6.9
 */
class chunk_info_t
{
	std::size_t m_started_at;
	std::size_t m_size;

	/**
	 * @brief Storage of chunk extension parameters
	 *
	 * The instance will be allocated only if chunk has extension's parameters.
	 *
	 * @since v.0.7.0
	 */
	chunk_ext_params_unique_ptr_t m_ext_params;
public:
	//! Initializing constructor.
	chunk_info_t(
		std::size_t started_at,
		std::size_t size,
		chunk_ext_params_unique_ptr_t ext_params )
		:	m_started_at{ started_at }
		,	m_size{ size }
		,	m_ext_params{ std::move( ext_params ) }
	{}

	//! Get the starting offset of chunk.
	[[nodiscard]]
	std::size_t
	started_at() const noexcept { return m_started_at; }

	//! Get the size of chunk.
	[[nodiscard]]
	std::size_t
	size() const noexcept { return m_size; }

	//! Extract the chunk value from the whole body.
	/*!
	 * @attention
	 * This method doesn't check the possibility of the extraction.
	 * An attempt of extraction of chunk from a body that is too small
	 * is undefined behavior.
	 */
	[[nodiscard]]
	string_view_t
	make_string_view_nonchecked( string_view_t full_body ) const noexcept
	{
		return full_body.substr( m_started_at, m_size );
	}

	//! Extract the chunk value from the whole body.
	/*!
	 * A check of possibility of extraction is performed.
	 *
	 * @throw exception_t if @a full_body is too small to hold the chunk.
	 */
	[[nodiscard]]
	string_view_t
	make_string_view( string_view_t full_body ) const
	{
		if( m_started_at >= full_body.size() ||
				m_started_at + m_size > full_body.size() )
		{
			throw exception_t{
				fmt::format(
						RESTINIO_FMT_FORMAT_STRING(
							"unable to make a chunk (started_at:{}, size: {}) "
							"from a body with length:{}" ),
						m_started_at,
						m_size,
						full_body.size() )
			};
		}

		return make_string_view_nonchecked( full_body );
	}

	/**
	 * @brief Get a list of chunk extension's params.
	 *
	 * In case this chunk has extensions this function returns a valid pointer
	 * to a vector of ext parameters (name-value pairs).
	 *
	 * @code
	 * auto handle_request( restinio::request_t req ) {
	 *     const auto * chunked_input = req.chunked_input_info();
	 *     if( !chunked_input )
	 *     {
	 *         return restinio::request_rejected();
	 *     }
	 *     auto resp = req->create_response()
	 *         .append_header_date_field()
	 *         .append_header( "Content-Type", "text/plain; charset=utf-8" );
	 *
	 *     int i = 0;
	 *     for( const auto & ch : chunked_input->chunks() )
	 *     {
	 *         if( const auto * params = ch.ext_params(); params )
	 *         {
	 *             resp.append_body(
	 *                 fmt::format(
	 *                     FMT_STRING( "Chunk #{} has {} ext param(s)\r\n" ),
	 *                     i++,
	 *                     params->size() ) );
	 *         }
	 *         else
	 *         {
	 *             resp.append_body(
	 *                 fmt::format(
	 *                     FMT_STRING( "Chunk #{} has no ext params\r\n" ),
	 *                     i++ ) );
	 *         }
	 *     }
	 *     return resp.done();
	 * }
	 * @endcode
	 *
	 * @since v.0.7.0
	 */
	[[nodiscard]] nullable_pointer_t< const chunk_ext_params_t >
	ext_params() const noexcept
	{
		return m_ext_params.get();
	}
};

namespace impl
{

//
// chunked_input_info_block_t
//
/*!
 * @brief Bunch of data related to chunked input.
 *
 * @since v.0.6.9
 */
struct chunked_input_info_block_t
{
	//! All non-empty chunks from the input.
	std::vector< chunk_info_t > m_chunks;

	//! Trailing fields found in the input.
	/*!
	 * @note
	 * Can be empty.
	 */
	http_header_fields_t m_trailing_fields;
};

} /* namespace impl */

//
// chunked_input_info_t
//
/*!
 * @brief An information about chunks and trailing fields in the
 * incoming request.
 *
 * This information is collected if chunked encoding is used in the
 * incoming request.
 *
 * @since v.0.6.9
 */
class chunked_input_info_t
{
	//! Actual data.
	impl::chunked_input_info_block_t m_info;

public:
	//! Default constructor. Makes empty object.
	chunked_input_info_t() = default;
	//! Initializing constructor.
	/*!
	 * @note
	 * This constrictor is intended to be used inside RESTinio and
	 * can be changed in future versions without any notice.
	 */
	chunked_input_info_t(
		impl::chunked_input_info_block_t info )
		:	m_info{ std::move(info) }
	{}

	//! Get the count of chunks.
	/*!
	 * @retval 0 if there is no chunks in the incoming request.
	 */
	[[nodiscard]]
	std::size_t
	chunk_count() const noexcept { return m_info.m_chunks.size(); }

	//! Get reference to the description of a chunk by index.
	/*!
	 * @attention
	 * This method doesn't check the validity of @a index.
	 * An attempt to access non-existent chunk is undefined behavior.
	 */
	[[nodiscard]]
	const chunk_info_t &
	chunk_at_nochecked( std::size_t index ) const noexcept
	{
		return m_info.m_chunks[ index ];
	}

	//! Get reference to the description of a chunk by index.
	/*!
	 * @throw std::exception if @a index is invalid.
	 */
	[[nodiscard]]
	const chunk_info_t &
	chunk_at( std::size_t index ) const
	{
		return m_info.m_chunks.at( index );
	}

	//! Get access to the container with description of chunks.
	/*!
	 * @note
	 * The actual type of the container is not specified and can
	 * be changed from version to version. But this container
	 * can be sequentially enumerated from begin() to the end().
	 */
	[[nodiscard]]
	const auto &
	chunks() const noexcept
	{
		return m_info.m_chunks;
	}

	//! Get access to the container with trailing fields.
	/*!
	 * @note
	 * This can be an empty container if there is no trailing fields
	 * in the incoming request.
	 */
	[[nodiscard]]
	const http_header_fields_t &
	trailing_fields() const noexcept
	{
		return m_info.m_trailing_fields;
	}
};

//
// chunked_input_info_unique_ptr_t
//
/*!
 * @brief Alias of unique_ptr for chunked_input_info.
 *
 * @since v.0.6.9
 */
using chunked_input_info_unique_ptr_t =
	std::unique_ptr< chunked_input_info_t >;

} /* namespace restinio */

