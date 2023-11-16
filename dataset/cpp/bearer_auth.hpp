/*
 * RESTinio
 */

/*!
 * @file
 * @brief Helpers for dealing with bearer authentification.
 *
 * @since v.0.6.7.1
 */

#pragma once

#include <restinio/helpers/http_field_parsers/authorization.hpp>

#include <restinio/http_headers.hpp>
#include <restinio/request_handler.hpp>
#include <restinio/expected.hpp>

#include <iostream>

namespace restinio
{

namespace http_field_parsers
{

namespace bearer_auth
{

//
// params_t
//
/*!
 * @brief Parameters for bearer authentification.
 *
 * @since v.0.6.7.1
 */
struct params_t
{
	//! Access Token.
	/*!
	 * Can't be empty.
	 */
	std::string token;
};

//
// extraction_error_t
//
/*!
 * @brief Error codes for failures of extraction of bearer authentification
 * parameters.
 *
 * @since v.0.6.7.1
 */
enum class extraction_error_t
{
	//! There is no HTTP field with authentification parameters.
	no_auth_http_field,

	//! The HTTP field with authentification parameters can't be parsed.
	illegal_http_field_value,

	//! Different authentification scheme found.
	//! bearer authentification scheme is expected.
	not_bearer_auth_scheme,

	//! Invalid value of parameter for bearer authentification scheme.
	//! The single parameter in the form of b64token is expected.
	invalid_bearer_auth_param,
};

/*!
 * @brief Helper function to get a string name of extraction_error enum.
 *
 * @since v.0.6.9
 */
[[nodiscard]]
inline string_view_t
to_string_view( extraction_error_t what ) noexcept
{
	string_view_t result{ "<unknown>" };

	switch( what )
	{
	case extraction_error_t::no_auth_http_field:
		result = string_view_t{ "no_auth_http_field" };
	break;
	
	case extraction_error_t::illegal_http_field_value:
		result = string_view_t{ "illegal_http_field_value" };
	break;

	case extraction_error_t::not_bearer_auth_scheme:
		result = string_view_t{ "not_bearer_auth_scheme" };
	break;

	case extraction_error_t::invalid_bearer_auth_param:
		result = string_view_t{ "invalid_bearer_auth_param" };
	break;
	}

	return result;
}

//
// try_extract_params
//
/*!
 * @brief Helper function for getting parameters of bearer authentification
 * from an already parsed HTTP-field.
 *
 * @attention
 * This function doesn't check the content of
 * authorization_value_t::auth_scheme. It's expected that this field was
 * checked earlier.
 *
 * Usage example:
 * @code
 * auto on_request(restinio::request_handle_t & req) {
 * 	using namespace restinio::http_field_parsers;
 *		const auto opt_field = req.header().opt_value_of(
 * 			restinio::http_field::authorization);
 * 	if(opt_field) {
 * 		const auto parsed_field = authorization_value_t::try_parse(*opt_field);
 * 		if(parsed_field) {
 * 			if("basic" == parsed_field->auth_scheme) {
 * 				... // Dealing with Basic authentification scheme.
 * 			}
 * 			else if("bearer" == parsed_field->auth_scheme) {
 * 				using namespace restinio::http_field_parsers::bearer_auth;
 * 				const auto bearer_params = try_extract_params(*parsed_field);
 * 				if(bearer_params) {
 * 					const std::string & token = auth_params->token;
 * 					... // Do something with token.
 * 				}
 * 			}
 * 			else {
 * 				... // Other authentification schemes.
 * 			}
 * 		}
 * 	}
 * 	...
 * }
 * @endcode
 *
 * @since v.0.6.8
 */
[[nodiscard]]
inline expected_t< params_t, extraction_error_t >
try_extract_params(
	const authorization_value_t & http_field )
{
	const auto * b64token = std::get_if<authorization_value_t::token68_t>(
			&http_field.auth_param );
	if( !b64token )
		return make_unexpected( extraction_error_t::invalid_bearer_auth_param );

	return params_t{ b64token->value };
}

/*!
 * @brief Helper function for getting parameters of bearer authentification
 * from an already parsed HTTP-field.
 *
 * @attention
 * This function doesn't check the content of
 * authorization_value_t::auth_scheme. It's expected that this field was
 * checked earlier.
 *
 * @note
 * This function can be used if one wants to avoid memory allocation
 * and can reuse value of auth_params.
 *
 * Usage example (please note that `const` is not used in code when
 * authorization HTTP-field is parsed):
 * @code
 * auto on_request(restinio::request_handle_t & req) {
 * 	using namespace restinio::http_field_parsers;
 *		const auto opt_field = req.header().opt_value_of(
 * 			restinio::http_field::authorization);
 * 	if(opt_field) {
 * 		// parsed_field is a mutable object.
 * 		// The content of parsed_field->auth_param can be moved out.
 * 		auto parsed_field = authorization_value_t::try_parse(*opt_field);
 * 		if(parsed_field) {
 * 			if("basic" == parsed_field->auth_scheme) {
 * 				... // Dealing with Basic authentification scheme.
 * 			}
 * 			else if("bearer" == parsed_field->auth_scheme) {
 * 				using namespace restinio::http_field_parsers::bearer_auth;
 * 				const auto bearer_params =
 * 						// Please note the usage of std::move here.
 * 						try_extract_params(std::move(*parsed_field));
 * 				if(bearer_params) {
 * 					const std::string & token = auth_params->token;
 * 					... // Do something with token.
 * 				}
 * 			}
 * 			else {
 * 				... // Other authentification schemes.
 * 			}
 * 		}
 * 	}
 * 	...
 * }
 * @endcode
 *
 * @since v.0.6.8
 */
[[nodiscard]]
inline expected_t< params_t, extraction_error_t >
try_extract_params(
	authorization_value_t && http_field )
{
	auto * b64token = std::get_if<authorization_value_t::token68_t>(
			&http_field.auth_param );
	if( !b64token )
		return make_unexpected( extraction_error_t::invalid_bearer_auth_param );

	return params_t{ std::move(b64token->value) };
}

namespace impl
{

[[nodiscard]]
inline expected_t< params_t, extraction_error_t >
perform_extraction_attempt(
	const std::optional< string_view_t > opt_field_value )
{
	if( !opt_field_value )
		return make_unexpected( extraction_error_t::no_auth_http_field );

	auto field_value_parse_result = authorization_value_t::try_parse(
			*opt_field_value );
	if( !field_value_parse_result )
		return make_unexpected( extraction_error_t::illegal_http_field_value );

	auto & parsed_value = *field_value_parse_result;
	if( "bearer" != parsed_value.auth_scheme )
		return make_unexpected( extraction_error_t::not_bearer_auth_scheme );

	return try_extract_params( std::move(parsed_value) );
}

} /* namespace impl */

//
// try_extract_params
//
/*!
 * @brief Helper function for getting parameters of bearer authentification
 * from a set of HTTP-fields.
 *
 * This helper function is intended to be used for cases when authentification
 * parameters are stored inside a HTTP-field with a custom name. For example:
 * @code
 * auto check_authorization(const restinio::http_header_fields_t & fields) {
 * 	using namespace restinio::http_field_parsers::bearer_auth;
 * 	const auto auth_params = try_extract_params(fields, "X-My-Authorization");
 * 	if(auth_params) {
 * 		const std::string & token = auth_params->token;
 * 		... // Do something with token.
 * 	}
 * 	...
 * }
 * @endcode
 *
 * @since v.0.6.9
 */
[[nodiscard]]
inline expected_t< params_t, extraction_error_t >
try_extract_params(
	//! A set of HTTP-fields.
	const http_header_fields_t & fields,
	//! The name of a HTTP-field with authentification parameters.
	string_view_t auth_field_name )
{
	return impl::perform_extraction_attempt(
			fields.opt_value_of( auth_field_name ) );
}

/*!
 * @brief Helper function for getting parameters of bearer authentification
 * from a request.
 *
 * This helper function is intended to be used for cases when authentification
 * parameters are stored inside a HTTP-field with a custom name. For example:
 * @code
 * auto on_request(restinio::request_handle_t & req) {
 * 	using namespace restinio::http_field_parsers::bearer_auth;
 * 	const auto auth_params = try_extract_params(*req, "X-My-Authorization");
 * 	if(auth_params) {
 * 		const std::string & token = auth_params->token;
 * 		... // Do something with token.
 * 	}
 * 	...
 * }
 * @endcode
 *
 * @since v.0.6.7.1
 */
template< typename Extra_Data >
[[nodiscard]]
inline expected_t< params_t, extraction_error_t >
try_extract_params(
	//! A request that should hold a HTTP-field with authentification
	//! parameters.
	const generic_request_t< Extra_Data > & req,
	//! The name of a HTTP-field with authentification parameters.
	string_view_t auth_field_name )
{
	return try_extract_params( req.header(), auth_field_name );
}

/*!
 * @brief Helper function for getting parameters of bearer authentification
 * from a set of HTTP-fields.
 *
 * Usage example:
 * @code
 * auto check_authorization(const restinio::http_header_fields_t & fields) {
 * 	using namespace restinio::http_field_parsers::bearer_auth;
 * 	const auto auth_params = try_extract_params(
 * 			fields, restinio::http_field::authorization);
 * 	if(auth_params) {
 * 		const std::string & token = auth_params->token;
 * 		... // Do something with token.
 * 	}
 * 	...
 * }
 * @endcode
 *
 * @since v.0.6.9
 */
[[nodiscard]]
inline expected_t< params_t, extraction_error_t >
try_extract_params(
	//! A set of HTTP-fields.
	const http_header_fields_t & fields,
	//! The ID of a HTTP-field with authentification parameters.
	http_field_t auth_field_id )
{
	return impl::perform_extraction_attempt(
			fields.opt_value_of( auth_field_id ) );
}

/*!
 * @brief Helper function for getting parameters of bearer authentification
 * from a request.
 *
 * Usage example:
 * @code
 * auto on_request(restinio::request_handle_t & req) {
 * 	using namespace restinio::http_field_parsers::bearer_auth;
 * 	const auto auth_params = try_extract_params(
 * 			*req, restinio::http_field::authorization);
 * 	if(auth_params) {
 * 		const std::string & token = auth_params->token;
 * 		... // Do something with token.
 * 	}
 * 	...
 * }
 * @endcode
 *
 * @since v.0.6.7.1
 */
template< typename Extra_Data >
[[nodiscard]]
inline expected_t< params_t, extraction_error_t >
try_extract_params(
	//! A request that should hold a HTTP-field with authentification
	//! parameters.
	const generic_request_t< Extra_Data > & req,
	//! The ID of a HTTP-field with authentification parameters.
	http_field_t auth_field_id )
{
	return try_extract_params( req.header(), auth_field_id );
}

} /* namespace bearer_auth */

} /* namespace http_field_parsers */

} /* namespace restinio */

