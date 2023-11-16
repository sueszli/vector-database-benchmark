#include <eepp/network/platform/platformimpl.hpp>
#include <eepp/network/tcplistener.hpp>
#include <eepp/network/tcpsocket.hpp>
#include <eepp/system/log.hpp>

namespace EE { namespace Network {

TcpListener::TcpListener() : Socket( Tcp ) {}

unsigned short TcpListener::getLocalPort() const {
	if ( getHandle() != Private::SocketImpl::invalidSocket() ) {
		// Retrieve informations about the local end of the socket
		sockaddr_in address;
		Private::SocketImpl::AddrLength size = sizeof( address );

		if ( getsockname( getHandle(), reinterpret_cast<sockaddr*>( &address ), &size ) != -1 ) {
			return ntohs( address.sin_port );
		}
	}

	// We failed to retrieve the port
	return 0;
}

Socket::Status TcpListener::listen( unsigned short port, const IpAddress& address ) {
	// Close the socket if it is already bound
	close();

	// Create the internal socket if it doesn't exist
	create();

	// Check if the address is valid
	if ( ( address == IpAddress::None ) || ( address == IpAddress::Broadcast ) )
		return Error;

	// Bind the socket to the specified port
	sockaddr_in addr = Private::SocketImpl::createAddress( address.toInteger(), port );
	if ( bind( getHandle(), reinterpret_cast<sockaddr*>( &addr ), sizeof( addr ) ) == -1 ) {
		// Not likely to happen, but...
		Log::error( "Failed to bind listener socket to port %d", port );
		return Error;
	}

	// Listen to the bound port
	if ( ::listen( getHandle(), SOMAXCONN ) == -1 ) {
		// Oops, socket is deaf
		Log::error( "Failed to Listen to port %d", port );
		return Error;
	}

	return Done;
}

void TcpListener::close() {
	// Simply close the socket
	Socket::close();
}

Socket::Status TcpListener::accept( TcpSocket& socket ) {
	// Make sure that we're listening
	if ( getHandle() == Private::SocketImpl::invalidSocket() ) {
		Log::error( "Failed to accept a new connection, the socket is not listening" );
		return Error;
	}

	// Accept a new connection
	sockaddr_in address;
	Private::SocketImpl::AddrLength length = sizeof( address );
	SocketHandle remote = ::accept( getHandle(), reinterpret_cast<sockaddr*>( &address ), &length );

	// Check for errors
	if ( remote == Private::SocketImpl::invalidSocket() )
		return Private::SocketImpl::getErrorStatus();

	// Initialize the new connected socket
	socket.close();
	socket.create( remote );

	return Done;
}

}} // namespace EE::Network
