#include "le_core.h"
#include "le_hash_util.h"

#include "le_log.h"
#include "le_console.h"

#include <algorithm>
#include <cstdio>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <deque>

#include <assert.h>
#include <errno.h>
#include <string.h>

#include <memory>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <map>

#include "private/le_console/le_console_types.h"
#include "private/le_console/le_console_server.h"
#include "private/le_console/le_char_tree.h"

#include "private/le_core/le_settings_private_types.inl"

// Translation between winsock and posix sockets
#ifdef _WIN32
#	define strtok_reentrant strtok_s
#else
#	define strtok_reentrant strtok_r
#endif

extern void le_console_server_register_api( void* api ); // in le_console_server

static constexpr auto ISL_TTY_COLOR = "\x1b[38;2;204;203;164m";

namespace telnet {
// Telnet control byte constants
constexpr auto IAC      = '\xff';
constexpr auto DO       = '\xfd';
constexpr auto DONT     = '\xfe';
constexpr auto WILL     = '\xfb';
constexpr auto WONT     = '\xfc';
constexpr auto LINEMODE = '\x22';
constexpr auto ECHO     = '\x01';
constexpr auto SB       = '\xfa';
constexpr auto SE       = '\xf0';
constexpr auto SLC      = '\x03'; // substitute local characters
} // namespace telnet

/*

Intent for this module:

A console is an interactive session which has its own state and its own history.
Each session is Modal, it can run as an interactive TTY-like session (once you enter "tty")

*/

static void**                  PP_CONSOLE_SINGLETON  = nullptr; // set via registration method
static le_console_server_api_t le_console_server_api = {};

static constexpr auto NO_CONSOLE_MSG = "Could not find console. You must create at least one console object.";

static le_console_o* produce_console() {
	return static_cast<le_console_o*>( *PP_CONSOLE_SINGLETON );
}

// ----------------------------------------------------------------------
// This method may not log via le::log, because otherwise there is a
// chance of a deadlock.
static void logger_callback( char const* chars, uint32_t num_chars, void* user_data ) {
	auto connection = ( le_console_o::connection_t* )user_data;
	connection->channel_out.post( "\r" + std::string( chars, num_chars ) + "\r\n" );
	connection->wants_redraw = true;
}

// ----------------------------------------------------------------------

// If we initialize an object from this and store it static,
// then the destructor will get called when this module is being unloaded
// this allows us to remove ourselves from listeners before the listener
// gets destroyed.
class LeLogSubscriber : NoCopy {
  public:
	explicit LeLogSubscriber( le_console_o::connection_t* connection )
	    : handle( le_log::api->add_subscriber( logger_callback, connection, connection->log_level_mask ) ) {
		static auto logger = le::Log( LOG_CHANNEL );
		logger.debug( "Adding Log subscriber for %s with mask 0x%x", connection->remote_ip.c_str(), connection->log_level_mask );
	}
	~LeLogSubscriber() {
		static auto logger = le::Log( LOG_CHANNEL );
		logger.debug( "Removing Log subscriber" );
		// we must remove the subscriber because it may get called before we have a chance to change the callback address -
		// even callback forwarding won't help, because the reloader will log- and this log event will happen while the
		// current module is not yet loaded, which means there is no valid code to run for the subscriber.
		le_log::api->remove_subscriber( handle );
	};

  private:
	uint64_t handle;
};

// ----------------------------------------------------------------------
// We use this convoluted construction of a unique_ptr around a RAII class,
// so that we can detect when a module is being unloaded - in which case
// the unique_ptr will call the destructor on LeLogSubscriber.
//
// We must un-register the subscriber in case that this module is reloaded,
// because the loader itself might log, and this logging call may trigger
// a call onto an address which has just been unloaded...
//
// Even callback-forwarding won't help, because the call will happen in the
// liminal stage where this module has been unloaded, and its next version
// has not yet been loaded.
//
static std::unordered_map<uint32_t, std::unique_ptr<LeLogSubscriber>>& le_console_produce_log_subscribers() {
	static std::unordered_map<uint32_t, std::unique_ptr<LeLogSubscriber>> LOG_SUBSCRIBER = {};

	return LOG_SUBSCRIBER;
}

// ----------------------------------------------------------------------

le_console_o::connection_t::~connection_t() {
	// Remove subscribers if there were any
	this->wants_log_subscriber = false;
	le_console_produce_log_subscribers()[ this->fd ].reset( nullptr );
}

// ----------------------------------------------------------------------

// We want to start the server on-demand.
// if the module gets unloaded, the server thread needs to be first stopped
// if the module gets reloaded, the server thread needs to be resumed
class ServerWatcher : NoCopy {
  public:
	ServerWatcher( le_console_o* console_ )
	    : console( console_ ) {
		if ( console && console->server ) {
			le_console_server_api.start_thread( console->server );
		}
	};
	~ServerWatcher() {
		if ( console && console->server ) {
			// we must stop server
			le_console_server_api.stop_thread( console->server );
		}
	};

  private:
	le_console_o* console = nullptr;
};

// ----------------------------------------------------------------------

static std::unique_ptr<ServerWatcher>& le_console_produce_server_watcher( le_console_o* console = nullptr ) {
	static std::unique_ptr<ServerWatcher> SERVER_WATCHER = {};

	if ( nullptr == SERVER_WATCHER.get() && console != nullptr ) {
		SERVER_WATCHER = std::make_unique<ServerWatcher>( console );
	}
	return SERVER_WATCHER;
}

// ----------------------------------------------------------------------

static bool le_console_server_start() {
	static auto logger = le::Log( LOG_CHANNEL );

	le_console_o* self = produce_console();

	if ( self == nullptr ) {
		logger.error( NO_CONSOLE_MSG );
		return false;
	}

	if ( nullptr == self->server ) {
		logger.info( "* Creating Server..." );
		self->server = le_console_server_api.create( self ); // destroy server
		logger.info( "* Starting Server..." );
		le_console_server_api.start( self->server ); // setup server
		le_console_produce_server_watcher( self );   // Implicitly starts server thread
	}

	return true;
}

// ----------------------------------------------------------------------

static bool le_console_server_stop() {
	static auto logger = le::Log( LOG_CHANNEL );

	le_console_o* self = produce_console();

	if ( self == nullptr ) {
		logger.error( NO_CONSOLE_MSG );
		return false;
	}

	le_console_produce_log_subscribers().clear();

	if ( self->server ) {
		logger.info( "* Stopping server..." );
		le_console_produce_server_watcher().reset( nullptr ); // explicitly call destructor on server watcher - this will join the server thread
		le_console_server_api.stop( self->server );           // stop server
		le_console_server_api.destroy( self->server );        // destroy server
		self->server = nullptr;
	}

	return true;
}

// ------------------------------------------------------------------------------------------
// Split a given string into tokens by replacing delimiters with \0 - and returning a vector
// of token c-strings from the string.
// returns false if no tokens could be found
static void tokenize_string( std::string& msg, std::vector<char const*>& tokens, char const* delim = "\n\r= " ) {

	char* context = nullptr;
	char* token   = strtok_reentrant( msg.data(), delim, &context );

	while ( token != nullptr ) {
		tokens.emplace_back( token );
		token = strtok_reentrant( nullptr, delim, &context );
	}
}

// ----------------------------------------------------------------------

// Will update stream_begin to point at one past the last character
// of the last command which was interpreted
std::string telnet_filter( le_console_o::connection_t*       connection,
                           std::string::const_iterator       stream_begin,
                           std::string::const_iterator const stream_end ) {

	static auto logger = le::Log( LOG_CHANNEL );

	if ( connection == nullptr ) {
		assert( false && "Must have valid connection" );
		return "";
	}
	// ----------| invariant: connection is not nullptr

	if ( stream_end - stream_begin <= 0 ) {
		// no characters to process.
		return "";
	}

	std::string result;
	result.reserve( uint32_t( stream_end - stream_begin ) ); // pre-allocate maximum possible chars that this operation might need

	// Find next command
	// Set iterator `it` to one past next IAC byte. starts search at iterator position `it`
	// return true if a command was found
	static auto find_next_command = []( std::string::const_iterator& it, std::string::const_iterator const& it_end ) -> bool {
		if ( it == it_end ) {
			return false;
		}
		while ( it != it_end ) {
			if ( *it == telnet::IAC ) {

				// if there is a lookahead character, and it is in
				// fact a IAC character, this means that IAC is meant
				// to be escaped and interpreted literally.
				if ( it + 1 != it_end && ( uint8_t( *( it + 1 ) ) == uint8_t( telnet::IAC ) ) ) {
					it = it + 2;
					continue;
				}

				++it;
				return true;
			}
			it++;
		}
		return false;
	};

	// if is_option, returns true and sets it_end to one past the last character that is part of the option
	// otherwise return false and leave it_end untouched.
	static auto is_option = []( std::string::const_iterator const it, std::string::const_iterator& it_end ) -> bool {
		if ( it == it_end ) {
			return false;
		}
		// ---------| invariant: we are not at the end of the stream

		if ( it + 1 == it_end ) {
			return false;
		}
		// ----------| invariant: there is a next byte available

		if ( uint8_t( *it ) >= uint8_t( telnet::WILL ) &&
		     uint8_t( *it ) <= uint8_t( telnet::DONT ) ) {
			it_end = it + 2;
			return true;
		}

		return false;
	};

	// if is_option, returns true and sets it_end to one past the last character that is part of the option
	// if is not a sub option, return false, don't touch it_end
	static auto is_sub_option = []( std::string::const_iterator const it, std::string::const_iterator& it_end ) -> bool {
		if ( it == it_end ) {
			return false;
		}
		// ---------| invariant: we are not at the end of the stream

		if ( uint8_t( *it ) != uint8_t( telnet::SB ) ) { // suboption start
			return false;
		}

		auto sub_option_end = it + 1; // start our search for suboption end command one byte past the current byte

		if ( !find_next_command( sub_option_end, it_end ) ) {
			return false;
		}

		// ---------| invariant: we managed to find the next IAC byte

		if ( sub_option_end == it_end ) {
			return false;
		}

		// ----------| invariant: there is a next byte available
		// look for "suboption end byte"
		if ( uint8_t( telnet::SE ) == uint8_t( *( sub_option_end ) ) ) {
			it_end = sub_option_end + 1;
			return true;
		}

		return false;
	};

	static auto process_sub_option = []( le_console_o::connection_t* connection, std::string::const_iterator it, std::string::const_iterator const it_end ) -> bool {
		it++; // Move past the SB Byte
		logger.info( "Suboption x%02x (%1$03u)", *it );

		if ( uint8_t( *it ) == 0x1f ) {
			logger.debug( "\t Suboption NAWS (Negotiate window size)" );

			it++; // Move past the suboption specifier

			// the next four bytes are width and height

			if ( it_end - it == 4 + 2 ) {
				connection->console_width = uint16_t( uint8_t( *it++ ) ) << 8; // msb first
				connection->console_width |= uint16_t( uint8_t( *it++ ) );
				connection->console_height = uint16_t( uint8_t( *it++ ) ) << 8; // msb first
				connection->console_height |= uint16_t( uint8_t( *it++ ) );
				logger.debug( "\t Setting Console window to %dx%d (w x h)", connection->console_width, connection->console_height );
				connection->wants_redraw = true;
			}
		}
		return false;
	};

	static auto process_option = []( le_console_o::connection_t* connection, std::string::const_iterator it, std::string::const_iterator const it_end ) -> bool {
		if ( it == it_end || it + 1 == it_end ) {
			return false;
		}

		// ----------| invariant: there is an option specifier available

		switch ( uint8_t( *it ) ) {
		case ( uint8_t( telnet::WILL ) ):
			logger.debug( "WILL x%02x (%1$03u)", *( it + 1 ) );

			break;
		case ( uint8_t( telnet::WONT ) ):
			logger.debug( "WONT x%02x (%1$03u)", *( it + 1 ) );
			break;
		case ( uint8_t( telnet::DO ) ):
			logger.debug( "DO   x%02x (%1$03u)", *( it + 1 ) );
			// client will ignore goahead
			// client requests something from us
			if ( uint8_t( *( it + 1 ) ) == '\x03' ) {
				// Do Suppress goahead
				connection->state = le_console_o::connection_t::State::eTTY;
				logger.debug( "We will suppress Goahead" );
			}
			break;
		case ( uint8_t( telnet::DONT ) ):
			logger.debug( "DONT x%02x (%1$03u)", *( it + 1 ) );
			if ( uint8_t( *( it + 1 ) ) == '\x03' ) {
				// Don't Suppress goahead
				connection->state = le_console_o::connection_t::State::ePlain;
				logger.debug( "We won't suppress Goahead" );
			}
			break;
		}

		return true;
	};

	// std::string test_str = "this is \xff\xff\xff\xff a test \xff\xff hello, well \xffhere";
	// stream_begin                                      = test_str.begin();
	// std::string::const_iterator const test_stream_end = test_str.end();

	while ( true ) {

		auto prev_stream_begin = stream_begin;
		find_next_command( stream_begin, stream_end );

		{
			// add all characters which are not commands to the result stream
			// until we hit stream_begin
			bool iac_flip_flop = false;
			for ( auto c = prev_stream_begin; c != stream_begin; c++ ) {
				if ( *c == '\xff' ) {
					iac_flip_flop ^= true;
				}
				if ( !iac_flip_flop ) {
					result.push_back( *c );
				}
			}
		}

		if ( stream_begin == stream_end ) {
			break;
		}

		auto sub_range_end = stream_end;
		// check for possible commands:
		if ( is_option( stream_begin, sub_range_end ) ) {
			process_option( connection, stream_begin, sub_range_end );
			stream_begin = sub_range_end; // move it to end of the current range
		} else if ( is_sub_option( stream_begin, sub_range_end ) ) {
			process_sub_option( connection, stream_begin, sub_range_end );
			stream_begin = sub_range_end;
			// do something with suboption
		} else {
			// this is neither a suboption not an option
		}
	}

	return result;
}
// --------------------------------------------------------------------------------

// find one-past the first next space that is followed by
// a non-space
static bool find_next_word_boundary(
    std::string::const_iterator&       it_begin,
    std::string::const_iterator const& it_end ) {

	auto it = it_begin;
	while ( true ) {
		if ( it != it_end &&
		     *it == ' ' &&
		     ( it + 1 != it_end ) &&
		     *( it + 1 ) != ' ' ) {
			// found first space after-a-non-space
			++it; // move cursor to first space after non-space
			break;
		} else if ( it != it_end ) {
			it++;
		} else {
			break;
		}
	}
	if ( it_begin != it ) {
		it_begin = it;
		return true;
	}
	return false;
}

// --------------------------------------------------------------------------------

// will set it_end to one past the previous word boundary if found,
// otherwise will leave it_end untouched and return false
static bool find_previous_word_boundary(
    std::string::const_iterator const it_begin,
    std::string::const_iterator&      it_end ) {

	auto it = it_end;
	it--;
	while ( true ) {
		if ( it != it_begin &&
		     *it != ' ' &&
		     ( it - 1 != it_begin ) &&
		     *( it - 1 ) == ' ' ) {
			// found first space-before-non-space
			break;
		} else if ( it != it_begin ) {
			it--;
		} else {
			break;
		}
	}
	if ( it_end != it ) {
		it_end = it;
		return true;
	}
	return false;
}
// --------------------------------------------------------------------------------

static void tty_clear_screen( le_console_o::connection_t* connection ) {
	std::ostringstream msg;
	msg << "\x1b[2J"    // clear screen
	    << "\x1b[\x48"; // position cursor to 1,1

	connection->channel_out.post( msg.str() );
	connection->wants_redraw = true;
}

// returns index of first char that is not equal
// or length of strings if both equal

static inline constexpr size_t first_diff( char const* a, char const* b ) {
	size_t result = 0;
	while ( *a != 0 && *b != 0 && *a++ == *b++ ) {
		result++;
	}
	return result;
}

// ffdecl.
static void cb_autocomplete_command( Command const* cmd, std::string const& str, std::vector<char const*> const& tokens, le_console_o::connection_t* connection );

// --------------------------------------------------------------------------------
// we want to be able to express a tree of command tokens
class Command {

  public:
	typedef void ( *cmd_cb )( Command const* cmd, std::string const& cmdline, std::vector<char const*> const& tokens, le_console_o::connection_t* connection ); // tokens from cmdline

  private:
	explicit Command( std::string const& name_, cmd_cb cb_execute_ = nullptr, cmd_cb cb_autocomplete_ = cb_autocomplete_command )
	    : autocomplete_command( cb_autocomplete_ )
	    , execute_command( cb_execute_ )
	    , name( name_ ) {
		// static auto logger = le::Log( LOG_CHANNEL );
		// logger.info( "+ ConsoleCommand '%s' %x", name.c_str(), this );
	}

  public:
	~Command() {
		for ( auto& c : commands ) {
			delete c.second;
		}
		// static auto logger = le::Log( LOG_CHANNEL );
		// logger.info( "- ConsoleCommand '%s' %x", name.c_str(), this );
	}

	Command()                                  = delete; // default constructor
	Command& operator=( const Command& other ) = delete; // copy assignment constructor
	Command& operator=( Command&& other )      = delete;
	Command( const Command& other )            = delete; // copy constructor
	Command( Command&& other )                 = delete; // move constructor

	// Factory function - you must use this to create a new command
	static Command* New( std::string const&& name_, cmd_cb&& execute_ = nullptr, cmd_cb&& autocomplete_ = cb_autocomplete_command ) {

		return new Command( name_, execute_, autocomplete_ );
	}

	// will take ownership of cmd
	Command* addSubCommand( Command* cmd ) {
		cmd->parent = this;
		auto it     = commands.emplace( cmd->name, cmd );
		if ( it.second == false ) {
			std::swap( cmd, it.first->second );
			delete cmd;
		}
		return this;
	}

	Command* endSubCommands() {
		if ( parent ) {
			return parent;
		} else {
			return this;
		}
	}

	bool find_subcommand( const char* token, Command** c ) {
		auto result = commands.find( token );
		if ( result != commands.end() ) {
			*c = ( result->second );
			return true;
		}
		return false;
	}

	bool find_suggestions( const char* token, size_t& num_matching_chars, std::vector<std::string>& suggestions ) const {

		suggestions.clear();
		size_t num_possible_suggestions = 0;

		auto found_node = autocomplete_cache->find_word( token, &num_matching_chars );

		if ( found_node ) {

			{
				// Calculate number of suggestions
				auto first_option_node = found_node->get_first_child();

				if ( first_option_node ) {
					num_possible_suggestions = 1 + first_option_node->count_siblings();
				}
			}

			for ( size_t i = 0; i != num_possible_suggestions; i++ ) {
				size_t suggestion_len = 0;
				found_node->get_suggestion_at( i, &suggestion_len, nullptr );

				std::string suggestion;
				suggestion.resize( suggestion_len );

				found_node->get_suggestion_at( i, &suggestion_len, suggestion.data() );
				suggestions.emplace_back( std::move( suggestion ) );
			}
		} else {
			return false;
		}

		return true;
	}

	std::string const& getName() {
		return name;
	}

	Command* getParent() const {
		return parent;
	}

	// Ivalidates and then regenerates the autocomplete cache for
	// a command and all its subcommands.
	void updateAutocompleteCache() {
		for ( auto& c : commands ) {
			c.second->updateAutocompleteCache();
		}

		// we must build the char tree

		std::vector<std::unique_ptr<std::string>> strings_container; // unique_ptr so that strings get auto-deleted
		std::vector<char const*>                  strings_p;         // points into strings_container, addresses are fixed, as strings are allocated on the heap and only the addressesd get

		for ( auto& e : commands ) {

			// Why do we laboriously allocate strings on the heap here so that we can point to them
			// if we could just as well point to them directly as keys of `commands`?
			// It's so that we can append a whitespace to each command before adding it to our
			// tree. The whitespace at the end makes it possible for autocomplete to stop at the
			// first point of difference when there are two possible commands that begin with the
			// same character sequence: "set" and "setting", for example. For prompt "se", we want to
			// be able to stop after the first 't', as both "set" and "setting" are possible choices.
			//
			// Having a whitespace at the end of each token adds an extra sibling at the first 't',
			// which has the effect of pausing the cursor there.

			strings_container.emplace_back( std::make_unique<std::string>( e.first ) );
			strings_container.back()->append( " " );
			strings_p.push_back( strings_container.back()->data() );
		}

		autocomplete_cache.reset( new le_char_tree::node_t() );
		autocomplete_cache->add_children( 0, strings_p.data(), strings_p.data() + strings_p.size() );
	}

	cmd_cb autocomplete_command; // callback for autocomplete
	cmd_cb execute_command;      // callback for execute

  private:
	std::unique_ptr<le_char_tree::node_t> autocomplete_cache;
	std::string const                     name;
	Command*                              parent = nullptr;
	std::map<std::string, Command*>       commands;
};

// --------------------------------------------------------------------------------

static void tab_pressed( le_console_o::connection_t* connection ) {
	// static auto logger = le::Log( LOG_CHANNEL );

	// Reduce the prompt to up to one char before the current cursor pos
	std::string              input{ connection->input_buffer.begin(), connection->input_buffer.begin() + connection->input_cursor_pos };
	std::vector<char const*> tokens;
	tokenize_string( input, tokens );

	uint32_t token_idx = 0;

	auto console = produce_console();

	Command* cmd = console->cmd.get();
	if ( cmd == nullptr ) {
		return;
	}

	// invariant: cmd exists

	while ( token_idx < tokens.size() && cmd->find_subcommand( tokens[ token_idx ], &cmd ) ) {
		token_idx++;
	}

	if ( token_idx == tokens.size() && cmd->getParent() ) {
		cmd = cmd->getParent();
	}

	if ( cmd->autocomplete_command ) {
		cmd->autocomplete_command( cmd, input, tokens, connection );
	}
}

static inline void check_cursor_pos( le_console_o::connection_t* connection ) {
	connection->input_cursor_pos = std::clamp<uint32_t>( connection->input_cursor_pos, 0, uint32_t( connection->input_buffer.size() ) );
}

// --------------------------------------------------------------------------------
// Fetch next or previous entry from session history, depending on direction.
// before switching entry, store current input, and its cursor position inline.
// After fetching next entry, restore its cursor position if it has been previously saved inline.
static void fetch_session_history_entry( le_console_o::connection_t* connection, int direction ) {

	constexpr auto CURSOR_POS_BYTE_COUNT               = sizeof( connection->input_cursor_pos ) + 2;
	char           cursor_pos[ CURSOR_POS_BYTE_COUNT ] = {};

	memcpy( cursor_pos + 2, &connection->input_cursor_pos, CURSOR_POS_BYTE_COUNT - 2 );

	// Store the current cursor position with the current history entry by
	// appending the cursor pos after two \0 bytes
	*connection->session_history_it = connection->input_buffer.append( cursor_pos, CURSOR_POS_BYTE_COUNT ); // append the cursor pos
	connection->session_history_it += direction;

	connection->input_buffer = *connection->session_history_it;

	if ( connection->input_buffer.size() >= CURSOR_POS_BYTE_COUNT &&
	     *( connection->input_buffer.end() - CURSOR_POS_BYTE_COUNT ) == 0x00 ) {
		// Retrieve a previous cursor position by parsing the last few bytes of a history entry
		// if it contains a cursor pos, it is after two \0 bytes towards the end of the string.
		memcpy( &connection->input_cursor_pos,
		        connection->input_buffer.data() + connection->input_buffer.size() - CURSOR_POS_BYTE_COUNT + 2,
		        CURSOR_POS_BYTE_COUNT - 2 );
		connection->input_buffer.resize( connection->input_buffer.size() - CURSOR_POS_BYTE_COUNT );
	} else {
		check_cursor_pos( connection );
	}

	connection->wants_redraw = true;
}

// --------------------------------------------------------------------------------

std::string process_tty_input( le_console_o::connection_t* connection, std::string const& msg ) {

	static auto logger = le::Log( LOG_CHANNEL );

	// Process virtual terminal control sequences - if we're in line mode we have to do these things
	// on the server-side.
	//
	// See: ECMA-48, <https://www.ecma-international.org/publications-and-standards/standards/ecma-48/>

	enum State {
		DATA,  // plain data
		ESC,   //
		CSI,   // ESC [
		ENTER, // '\r'
	};

	State state = State::DATA;

	union control_function_t {
		struct Bytes {
			char intro; // store in lower byte
			char final; // store in upper byte
		} bytes;
		uint16_t data = {};
	} control_function = {};

	// introducer and end byte of control function
	std::string parameters;
	bool        enter_user_input = false;

	static auto execute_control_function = []( le_console_o::connection_t* connection, control_function_t f, std::string const& parameters ) {
		// Execute control function on connection
		//
		// We encode the control function (which is identified by its start and end byte)
		// as a uint16_t which consists of (start_byte | end_byte << 8) so that we can
		// switch on it:
		//
		switch ( f.data ) {
		case ( '[' | 'A' << 8 ):
			// cursor up
			if ( connection->session_history_it != connection->session_history.end() &&
			     connection->session_history_it != connection->session_history.begin() ) {
				fetch_session_history_entry( connection, -1 );
			}
			break;
		case ( '[' | 'B' << 8 ):
			// cursor down
			if ( connection->session_history_it < connection->session_history.end() - 1 ) {
				fetch_session_history_entry( connection, +1 );
			}
			break;
		case ( '[' | 'C' << 8 ):
			if ( connection->input_cursor_pos < connection->input_buffer.size() ) {
				if ( parameters.size() == 3 && parameters[ 2 ] == '5' ) {
					// CTRL+RIGHT: ^[1;5D
					auto const                  it_cursor = connection->input_buffer.begin() + connection->input_cursor_pos;
					std::string::const_iterator it        = it_cursor;

					if ( find_next_word_boundary( it, connection->input_buffer.end() ) ) {
						connection->input_cursor_pos = connection->input_cursor_pos - ( it_cursor - it );
						connection->wants_redraw     = true;
					}
				} else {
					connection->input_cursor_pos++;
					connection->channel_out.post( "\x1b[C" ); // cursor right
				}
			}
			break;
		case ( '[' | 'D' << 8 ):
			if ( connection->input_cursor_pos > 0 ) {
				if ( parameters.size() == 3 && parameters[ 2 ] == '5' ) {
					// CTRL+LEFT: ^[1;5D
					auto const                  it_cursor = connection->input_buffer.begin() + connection->input_cursor_pos;
					std::string::const_iterator it        = it_cursor;

					if ( find_previous_word_boundary( connection->input_buffer.begin(), it ) ) {
						connection->input_cursor_pos = connection->input_cursor_pos - ( it_cursor - it );
						connection->wants_redraw     = true;
					}
				} else {
					// LEFT
					connection->input_cursor_pos--;
					connection->channel_out.post( "\x1b[D" ); // cursor left
				}
			}
			break;
		case ( '[' | '~' << 8 ): {
			if ( !parameters.empty() && parameters[ 0 ] == '3' ) {
				if ( connection->input_cursor_pos < connection->input_buffer.size() ) {
					connection->input_buffer.erase( connection->input_buffer.begin() + connection->input_cursor_pos );
					connection->wants_redraw = true;
				}
			}
			break;
		}
		default:
			logger.debug( "executing control function: 0x%02x ('%1$c'), with parameters: '%2$s' and final byte: 0x%3$02x ('%3$c')", f.bytes.intro, parameters.c_str(), f.bytes.final );
		}
	};

	for ( auto c : msg ) {

		switch ( state ) {
		case DATA:
			if ( c == '\x1b' ) {
				state = ESC;
			} else if ( c == '\r' ) {
				state = ENTER;
			} else {
				if ( c == '\x01' ) {
					// goto first char
					connection->input_cursor_pos = 0;
					connection->wants_redraw     = true;
				} else if ( c == '\x03' ) {
					// CTRL+C
					connection->input_buffer.clear();
					connection->input_cursor_pos = 0;
					connection->wants_redraw     = true;
				} else if ( c == '\x04' ) {
					// CTRL+D
					connection->wants_close = true;
				} else if ( c == '\x05' ) {
					// goto last char
					connection->input_cursor_pos = uint32_t( connection->input_buffer.size() );
					connection->wants_redraw     = true;
				} else if ( c == '\x09' ) {
					// logger.info( "tab character pressed." );
					tab_pressed( connection );
				} else if ( c == '\x0c' ) {
					tty_clear_screen( connection );
				} else if ( c == '\x17' && connection->input_cursor_pos > 0 ) {
					auto const                  it_cursor = connection->input_buffer.begin() + connection->input_cursor_pos;
					std::string::const_iterator it        = it_cursor;

					if ( find_previous_word_boundary( connection->input_buffer.begin(), it ) ) {
						connection->input_buffer.erase( it, it_cursor );
						connection->input_cursor_pos = connection->input_cursor_pos - ( it_cursor - it );
					}
					connection->wants_redraw = true;

				} else if ( c == '\x7f' ) { // delete
					if ( !connection->input_buffer.empty() ) {
						// remove last character if not empty
						if ( connection->input_cursor_pos > 0 ) {
							connection->input_buffer.erase( connection->input_buffer.begin() + --connection->input_cursor_pos );
							connection->wants_redraw = true;
						}
					}
				} else if ( c > '\x1f' ) {
					// PLAIN CHARACTER
					connection->input_buffer.insert( connection->input_buffer.begin() + connection->input_cursor_pos++, c );
					connection->wants_redraw = true;
				} else {
					logger.debug( "Unhandled character: 0x%02x ('%1$c')", c );
				}
			}
			break;
		case ENTER:
			if ( c == 0x00 || c == '\n' ) {
				enter_user_input = true;
				state            = DATA;
				connection->channel_out.post( "\r\n" ); // carriage-return+newline
			} else {
				connection->input_buffer.insert( connection->input_buffer.begin() + connection->input_cursor_pos++, '\r' );
				state = DATA;
			}
			break;
		case ESC:
			if ( c == '\x5b' || c == '\x9b' ) { // 7-bit or 8 bit representation of control sequence
				state                        = CSI;
				control_function.bytes.intro = c;
			} else {
				connection->input_buffer.insert( connection->input_buffer.begin() + connection->input_cursor_pos++, '\x1b' );
				state = DATA; // FIXME: is this correct? this is what we do if ESC is *not* followed by a control sequence character
			}
			break;
		case CSI:
			if ( c >= '\x30' && c <= '\x3f' ) { // parameter bytes
				parameters.push_back( c );
			} else if ( c >= '\x20' && c <= '\x2f' ) { // intermediary bytes (' ' to '/')
				// we want to add these to the parameter string that we capture
				parameters.push_back( c );
			} else if ( c >= '\x40' && c <= '\x7e' ) { // final byte of a control sequence
				control_function.bytes.final = c;
				execute_control_function( connection, control_function, parameters );
				state            = DATA;
				control_function = {};
			}
			break;
		}
		if ( enter_user_input ) {
			break;
		}
	}

	if ( enter_user_input ) {
		// submit
		std::string result( std::move( connection->input_buffer ) );
		connection->input_buffer.clear();
		connection->input_cursor_pos = 0;
		connection->wants_redraw     = true;

		if ( !result.empty() ) {
			// only add to history if there was a non-empty submission
			while ( connection->history.size() >= 20 ) {
				connection->history.pop_front();
			}
			connection->history.push_back( result );
			connection->session_history = connection->history;
			connection->session_history.push_back( connection->input_buffer );
			connection->session_history_it = connection->session_history.end() - 1;
		}

		return result;
	} else {
		return "";
	}
}

// Default Command Command callback for autocomplete -
// will find the best matching subcommand via upper_bound,
// taking into account any tokens up to the cursor
// and the first token including and following the cursor
//
static void cb_autocomplete_command( Command const* cmd, std::string const& str, std::vector<char const*> const& tokens, le_console_o::connection_t* connection ) {

	size_t num_parents = 0;
	{
		Command const* c = cmd;
		while ( true ) {
			c = c->getParent();
			if ( c ) {
				num_parents++;
			} else {
				break;
			}
		}
	}

	std::string last_token_complete;

	if ( num_parents < tokens.size() ) {
		last_token_complete = tokens.back();
	}

	if ( !tokens.empty() ) {
		last_token_complete = tokens.back();
	}

	std::vector<std::string> suggestions;
	size_t                   num_matching_chars = 0;
	if ( cmd->find_suggestions( last_token_complete.c_str(), num_matching_chars, suggestions ) ) {
		std::ostringstream ib;

		// rebuild the input prompt
		for ( uint32_t i = 0; i != num_parents; i++ ) {
			ib << tokens[ i ]
			   << " ";
		}

		ib << std::string( last_token_complete.begin(), last_token_complete.begin() + num_matching_chars );

		if ( suggestions.size() == 1 ) {

			last_token_complete = std::string( last_token_complete.begin(), last_token_complete.begin() + num_matching_chars );
			last_token_complete += suggestions.front();
			ib << suggestions.front();

			// see if we can find more suggestions - if there is more than one option, then we want to
			// see the options without having to press tab again.
			cmd->find_suggestions( last_token_complete.c_str(), num_matching_chars, suggestions );

			if ( suggestions.size() > 1 ) {
				std::vector<char const*> input_tokens;
				std::string              input_str = ib.str();
				tokenize_string( input_str, input_tokens );
				cb_autocomplete_command( cmd, input_str, input_tokens, connection );
				connection->input_cursor_pos = input_str.size();
			} else {
				connection->input_buffer     = ib.str();
				connection->input_cursor_pos = ib.str().size();
			}

			check_cursor_pos( connection );
			connection->wants_redraw = true;
			return;

		} else if ( suggestions.size() > 1 ) {

			std::ostringstream msg;
			msg
			    << "\x1b[0K" // erase from current position to end of the line
			    << "\x1b[7m" // color background set to dark yellowish gray
			    << suggestions[ 0 ]
			    << "\x1b[0m" // reset colors
			    ;
			size_t num_chars = suggestions[ 0 ].size();

			for ( size_t i = 1; i < suggestions.size(); i++ ) {
				msg
				    << std::string( num_chars, '\b' ) + "\n"
				    << "\x1b[2K" // erase all characters in line
				    << "\x1b[7m"
				    << suggestions[ i ]
				    << "\x1b[0m" // reset colors
				    ;
				num_chars = suggestions[ i ].size();
			}
			// msg << std::string( num_chars, '\b' );

			for ( size_t i = 0; i != suggestions.size() - 1; i++ ) {
				msg << "\x1bM"; // inverse of \n
			}

			connection->num_suggestion_lines = suggestions.size();
			connection->input_suggestion     = msg.str();
		} else {
			// if there was no suggestion, then add the last token back to input so
			// that it doesn't get deleted.
			ib << last_token_complete;
		}

		connection->input_buffer = ib.str();
		check_cursor_pos( connection );
		connection->wants_redraw = true;
	}
}

static void cb_get_setting_command( Command const* cmd, std::string const& str, std::vector<char const*> const& tokens, le_console_o::connection_t* connection ) {
	if ( tokens.size() == 2 ) {
		std::ostringstream msg;
		auto               setting_name  = tokens[ 1 ];
		auto               setting_value = tokens[ 2 ];
		auto               found_setting = le_core_get_setting_entry( setting_name );

		if ( found_setting != nullptr ) {
			void* setting = found_setting->p_opj;
			switch ( found_setting->type_hash ) {
			case ( SettingType::eConstBool ):
				msg << found_setting->name << " [ const bool ] == '" << ( ( *( const bool* )found_setting->p_opj ) ? "true" : "false" ) << "'\n\r";
				break;
			case ( SettingType::eBool ):
				msg << found_setting->name << " [ bool ] == '" << ( ( *( bool* )found_setting->p_opj ) ? "true" : "false" ) << "'\n\r";
				break;
			case ( SettingType::eInt32_t ):
				msg << found_setting->name << " [ int32_t ] == '" << ( *( int32_t* )found_setting->p_opj ) << "'\n\r";
				break;
			case ( SettingType::eUint32_t ):
				msg << found_setting->name << " [ uint32_t ] == '" << ( *( uint32_t* )found_setting->p_opj ) << "'\n\r";
				break;
			case ( SettingType::eInt ):
				msg << found_setting->name << " [ int ] == '" << ( *( int* )found_setting->p_opj ) << "'\n\r";
				break;
			case ( SettingType::eStdString ):
				msg << found_setting->name << " [ std::string ] == '" << ( *( std::string* )found_setting->p_opj ) << "'\n\r";
				break;
			default:
				msg << found_setting->name << " [ unknown ] == '" << std::hex << found_setting->p_opj << "'\n\r";
				break;
			}
		}
		connection->channel_out.post( msg.str() );
	}
};

static void cb_set_setting_command( Command const* cmd, std::string const& str, std::vector<char const*> const& tokens, le_console_o::connection_t* connection ) {
	static auto logger = le::Log( LOG_CHANNEL );
	if ( tokens.size() == 3 ) {
		auto setting_name  = tokens[ 1 ];
		auto setting_value = tokens[ 2 ];
		auto found_setting = le_core_get_setting_entry( setting_name );

		if ( found_setting != nullptr ) {
			std::ostringstream msg;
			void*              setting = found_setting->p_opj;
			switch ( found_setting->type_hash ) {
			case SettingType::eConstBool:
				logger.warn( "Cannot set value for setting: '%s'. Settings with type `const bool` "
				             "cannot be altered at runtime. They can only be set on startup, and "
				             "the first evaluated value of the setting remains the canonical "
				             "value for the duration of the program.",
				             found_setting->name.c_str() );
				break;

			case SettingType::eBool:
				*( bool* )( setting ) = bool( std::strtoul( setting_value, nullptr, 10 ) );
				break;
			case SettingType::eUint32_t:
				*( uint32_t* )( setting ) = uint32_t( strtoul( setting_value, nullptr, 10 ) );
				break;
			case SettingType::eInt32_t:
				*( int32_t* )( setting ) = int32_t( strtoul( setting_value, nullptr, 10 ) );
				break;
			case SettingType::eInt:
				*( int* )( setting ) = int( strtoul( setting_value, nullptr, 10 ) );
				break;
			case SettingType::eStdString:
				*( std::string* )( setting ) = std::string( setting_value );
				break;
			default:
				break;
			}
		}
	}
};

static void cb_show_help_command( Command const* cmd, std::string const& str, std::vector<char const*> const& tokens, le_console_o::connection_t* connection ) {

	connection->channel_out.post( "There is no help.\n\r" );
	connection->channel_out.post( "But there is autocomplete. Hit tab...\n\r" );

	check_cursor_pos( connection );
	connection->wants_redraw = true;
};

static void cb_list_settings_command( Command const* cmd, std::string const& str, std::vector<char const*> const& tokens, le_console_o::connection_t* connection ) {

	std::ostringstream msg;

	le_settings_map_t current_settings;
	le_core_copy_settings_entries( &current_settings, nullptr );
	for ( auto& s : current_settings.map ) {

		switch ( s.second.type_hash ) {
		case ( SettingType::eBool ):
			msg << s.second.name << " [ bool ] = '" << ( ( *( bool* )s.second.p_opj ) ? "true" : "false" ) << "'\n\r";
			break;
		case ( SettingType::eInt32_t ):
			msg << s.second.name << " [ int32_t ] = '" << ( *( int32_t* )s.second.p_opj ) << "'\n\r";
			break;
		case ( SettingType::eUint32_t ):
			msg << s.second.name << " [ uint32_t ] = '" << ( *( uint32_t* )s.second.p_opj ) << "'\n\r";
			break;
		case ( SettingType::eInt ):
			msg << s.second.name << " [ int ] = '" << ( *( int* )s.second.p_opj ) << "'\n\r";
			break;
		case ( SettingType::eStdString ):
			msg << s.second.name << " [ std::string ] = '" << ( *( std::string* )s.second.p_opj ) << "'\n\r";
			break;
		default:
			msg << s.second.name << " [ unknown ] = '" << std::hex << s.second.p_opj << "'\n\r";
			break;
		}
	}

	connection->channel_out.post( msg.str() );
};

static void cb_init_tty_command( Command const* cmd, std::string const& str, std::vector<char const*> const& tokens, le_console_o::connection_t* connection ) {
	using namespace telnet;
	std::ostringstream msg;

	msg
	    << IAC
	    << DONT
	    << ECHO
	    //
	    << IAC
	    << WILL
	    << ECHO
	    //
	    << IAC
	    << DO
	    << '\x1f' // negotiate about window size
	    //
	    << IAC
	    << WILL
	    << "\x03"; // suppress goahead

	connection->channel_out.post( msg.str() );
	msg.clear();
	msg << ISL_TTY_COLOR
	    << "Island Console.\r\nWelcome.\x1b[0m\r\n";
	connection->channel_out.post( msg.str() );
};

static void cb_log_command( Command const* cmd, std::string const& str, std::vector<char const*> const& tokens, le_console_o::connection_t* connection ) {
	static auto logger = le::Log( LOG_CHANNEL );
	// If you set log_level_mask to 0, this means that log messages will be mirrored to console
	// If you set log_level_mask to -1, this means that all log messages will be mirrored to console
	if ( tokens.size() == 2 ) {
		le_console_produce_log_subscribers().erase( uint32_t( connection->fd ) ); // Force the subscriber to be de-registered.
		connection->log_level_mask = uint32_t( strtol( tokens[ 1 ], nullptr, 0 ) );
		if ( connection->log_level_mask != 0 ) {
			le_console_produce_log_subscribers()[ uint32_t( connection->fd ) ] = std::make_unique<LeLogSubscriber>( connection ); // Force the subscriber to be re-registered.
			connection->wants_log_subscriber                                   = true;
		} else {
			// If we don't subscribe to any messages, we might as well remove the subscriber from the log
			le_console_produce_log_subscribers()[ uint32_t( connection->fd ) ].reset( nullptr );
			connection->wants_log_subscriber = false;
		}
		le_log::le_log_channel_i.info( logger.getChannel(), "Client %s updated console log level mask to 0x%x", connection->remote_ip.c_str(), connection->log_level_mask );
	} else {
		connection->channel_out.post( "Incorrect number of arguments.\n\rExpecting a single integer argument to specify log level mask.\n\r(E.g. -1 to capture all log levels.)\r\n" );
	}
}

static void cb_exit_command( Command const* cmd, std::string const& str, std::vector<char const*> const& tokens, le_console_o::connection_t* connection ) {
	using namespace telnet;
	std::ostringstream msg;
	msg

	    << IAC
	    << DO
	    << ECHO
	    //
	    << IAC
	    << WONT
	    << ECHO
	    //
	    << IAC
	    << WONT
	    << "\x03";
	connection->channel_out.post( msg.str() );
	connection->wants_redraw = false;
}

static void cb_cls_command( Command const* cmd, std::string const& str, std::vector<char const*> const& tokens, le_console_o::connection_t* connection ) {
	tty_clear_screen( connection );
}
// ------------------------------------------------------------------------------------------

static void le_console_setup_commands( le_console_o* self ) {

	static uint64_t settings_hash_cache = 0;
	uint64_t        settings_hash       = 0;

	le_core_copy_settings_entries( nullptr, &settings_hash );

	if ( settings_hash_cache != settings_hash ) {
		settings_hash_cache = settings_hash;
		self->cmd.release(); // force re-create for settings commands
	}

	if ( self->cmd.get() != nullptr ) {
		return;
	}

	// Root of commands Tree
	self->cmd.reset( Command::New( "" ) );

	// "set command" - used to manipulate settings
	Command* set_command = Command::New( "set" );
	{
		// Create a local copy of global settings
		le_settings_map_t current_settings;
		le_core_copy_settings_entries( &current_settings, nullptr );

		// add a subcommand for each settings entry
		for ( auto& e : current_settings.map ) {
			set_command->addSubCommand(
			    Command::New( std::string( e.second.name ), cb_set_setting_command ) );
		}
	}

	// "get command" - used to manipulate settings
	Command* get_command = Command::New( "get" );
	{
		// Create a local copy of global settings
		le_settings_map_t current_settings;
		le_core_copy_settings_entries( &current_settings, nullptr );

		// add a subcommand for each settings entry
		for ( auto& e : current_settings.map ) {
			get_command->addSubCommand(
			    Command::New( std::string( e.second.name ), cb_get_setting_command ) );
		}
	}

	self->cmd
	    ->addSubCommand( get_command )
	    ->addSubCommand( set_command )
	    ->addSubCommand( Command::New( "help", cb_show_help_command ) )
	    ->addSubCommand( Command::New( "settings", cb_list_settings_command ) )
	    ->addSubCommand( Command::New( "tty", cb_init_tty_command ) )
	    ->addSubCommand( Command::New( "cls", cb_cls_command ) )
	    ->addSubCommand( Command::New( "log", cb_log_command ) )
	    ->addSubCommand( Command::New( "exit", cb_exit_command ) );

	self->cmd->updateAutocompleteCache();
}

// ------------------------------------------------------------------------------------------

static void le_console_process_input() {
	static auto logger = le::Log( LOG_CHANNEL );

	static le_console_o* self = produce_console();

	if ( self == nullptr ) {
		logger.error( NO_CONSOLE_MSG );
		return;
	}

	auto connections_lock = std::scoped_lock( self->connections_mutex );

	for ( auto& c : self->connections ) {

		auto& connection = c.second;

		if ( connection->wants_close ) {
			// do no more processing if this connection wants to close
			continue;
		}

		std::string msg;

		connection->channel_in.fetch( msg );

		// redraw if requested
		if ( connection->wants_redraw && connection->state == le_console_o::connection_t::State::eTTY ) {
			char out_buf[ 2048 ]; // buffer for printf ops
			// clear line, reposition cursor

			std::string suggestion_erasor;

			for ( int i = 0; i != connection->num_suggestion_lines; i++ ) {
				suggestion_erasor.append( "\n\x1b[2K" ); // new line, erase all characters in line
			}
			for ( int i = 0; i != connection->num_suggestion_lines; i++ ) {
				suggestion_erasor.append( "\x1bM" ); // line up
			}

			int num_bytes = snprintf(
			    out_buf, sizeof( out_buf ),
			    "%s"      // suggestion erasor
			    "%s\r"    // tty color
			    "\x1b[0m" // color set to default
			    "\x1b[0K" // delete until the end of the line from current pos
			    "\x1b[1M" //
			    "\x1b[1m" // bold
			    ">"
			    "\x1b[0m "
			    "%s"
			    "%s"       // input suggestion string
			    "\x1b[%dG" // place the cursor at cursor pos
			    ,
			    //
			    suggestion_erasor.c_str(),
			    ISL_TTY_COLOR,
			    connection->input_buffer.c_str(),
			    connection->input_suggestion.c_str(), connection->input_cursor_pos + 3 // we add three for the prompt
			    )
			    //
			    ;
			if ( num_bytes > 1 ) {
				connection->channel_out.post( std::string( out_buf, out_buf + num_bytes ) );
			}

			if ( connection->input_suggestion.empty() ) {
				connection->num_suggestion_lines = 0;
			}

			connection->input_suggestion.clear();
			connection->wants_redraw = false;
		}

		if ( msg.empty() ) {
			continue;
		}

		// --------| invariant: msg is not empty

		// Apply the telnet protocol - this means interpreting (updating the connection's telnet state)
		// and removing telnet commands from the message stream, and unescaping double-\xff "IAC" bytes to \xff.
		//
		msg = telnet_filter( connection.get(), msg.begin(), msg.end() );

		if ( connection->wants_close ) {
			// Do no more processing if this connection wants to close
			continue;
		}

		// latest possible position to setup console commands - when there is input for the first time - or after a reload
		// we do this here so that autocomplete can function when processing tty input as it needs the command object first.
		le_console_setup_commands( self );

		if ( connection->state == le_console_o::connection_t::State::eTTY ) {
			// if we're supposed to process character-by-character, we do so here
			msg = process_tty_input( connection.get(), msg );
		}

		if ( msg.empty() ) {
			continue;
		}

		std::vector<char const*> tokens;
		tokenize_string( msg, tokens );

		uint32_t token_idx = 0;

		Command* cmd = self->cmd.get();
		if ( cmd == nullptr ) {
			return;
		}

		// invariant: cmd, that is the root command, exists

		while ( token_idx < tokens.size() && cmd->find_subcommand( tokens[ token_idx ], &cmd ) ) {
			token_idx++;
		}

		std::string hint;

		if ( token_idx != 0 ) {
			logger.info( "found command: %s", cmd->getName().c_str() );
			logger.info( "command parent: %s", cmd->getParent()->getName().c_str() );
			if ( cmd->execute_command ) {
				cmd->execute_command( cmd, msg, tokens, connection.get() );
			}
		} else {

			if ( !tokens.empty() ) {
				le_log::le_log_channel_i.warn( logger.getChannel(), "Did not recognise command: '%s'", tokens[ 0 ] );

				std::string msg = "Incorrect command: '";
				msg += tokens[ 0 ];
				msg += "'\n\r";

				connection->channel_out.post( msg.c_str() );
			} else {
				le_log::le_log_channel_i.warn( logger.getChannel(), "Empty command." );
			}
		}
	} // end for each connection
}

// ----------------------------------------------------------------------

static le_console_o* le_console_create() {
	static auto logger = le::Log( LOG_CHANNEL );
	logger.set_level( le::Log::Level::eDebug );
	auto self = new le_console_o();
	return self;
}

// ----------------------------------------------------------------------

static void le_console_destroy( le_console_o* self ) {
	static auto logger = le::Log( LOG_CHANNEL );

	// Tear-down and delete server in case there was a server
	le_console_server_stop();

	// We can do this because le_log will never be reloaded as it is part of the core.
	logger.info( "Destroying console..." );
	delete self;
}

// ----------------------------------------------------------------------

static void le_console_inc_use_count() {

	le_console_o* self = produce_console();

	if ( self == nullptr ) {
		*PP_CONSOLE_SINGLETON = le_console_create();
		self                  = produce_console();
	}

	self->use_count++;
}

// ----------------------------------------------------------------------

static void le_console_dec_use_count() {
	le_console_o* self = produce_console();
	if ( self ) {
		if ( --self->use_count == 0 ) {
			le_console_destroy( self );
			*PP_CONSOLE_SINGLETON = nullptr;
		};
	}
}

// ----------------------------------------------------------------------

LE_MODULE_REGISTER_IMPL( le_console, api_ ) {

	auto  api          = static_cast<le_console_api*>( api_ );
	auto& le_console_i = api->le_console_i;

	le_console_i.inc_use_count = le_console_inc_use_count;
	le_console_i.dec_use_count = le_console_dec_use_count;
	le_console_i.server_start  = le_console_server_start;
	le_console_i.server_stop   = le_console_server_stop;
	le_console_i.process_input = le_console_process_input;

	// Load function pointers for private server object
	le_console_server_register_api( &le_console_server_api );

	PP_CONSOLE_SINGLETON = le_core_produce_dictionary_entry( hash_64_fnv1a_const( "le_console_singleton" ) );

	if ( *PP_CONSOLE_SINGLETON ) {
		// if a console already exists, this is a sign that this module has been
		// reloaded - in which case we want to re-register the subscriber to the log.
		le_console_o* console = produce_console();
		console->cmd.reset( nullptr ); // delete commands - so that they get re-populated
		if ( console->server ) {
			le_console_produce_server_watcher( console );
		}
		if ( !console->connections.empty() ) {
			auto  lock        = std::scoped_lock( console->connections_mutex );
			auto& subscribers = le_console_produce_log_subscribers();
			for ( auto& c : console->connections ) {
				if ( c.second->wants_log_subscriber ) {
					subscribers[ c.first ] = std::make_unique<LeLogSubscriber>( c.second.get() );
				}
			}
		}
	}
}
