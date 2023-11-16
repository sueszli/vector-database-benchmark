#pragma once

#include <elle/reactor/network/fwd.hh>
#include <elle/reactor/network/socket.hh>
#include <elle/reactor/signal.hh>

namespace elle
{
  namespace reactor
  {
    namespace network
    {
      /// XXX[doc].
      class UDPServerSocket
        : public Socket
      {
        /*---------.
        | Typedefs |
        `---------*/
        public:
          using Super = Socket;
          using EndPoint = boost::asio::ip::udp::endpoint;

        /*-------------.
        | Construction |
        `-------------*/
        public:
          UDPServerSocket(Scheduler& sched,
                          UDPServer* server,
                          EndPoint const& peer);
          virtual
          ~UDPServerSocket();

        /*-----.
        | Read |
        `-----*/
        public:
          void
          read(Buffer buffer,
               DurationOpt timeout = {},
               int* bytes_read = nullptr) override;
          Size
          read_some(Buffer buffer,
                    DurationOpt timeout = {},
                    int* bytes_read = nullptr) override;
        private:
          friend class UDPServer;
          UDPServer* _server;
          EndPoint _peer;
          Byte* _read_buffer;
          Size _read_buffer_capacity;
          Size _read_buffer_size;
          Signal _read_ready;

        /*------.
        | Write |
        `------*/
        public:
          virtual
          void
          write(Buffer buffer);
          using Super::write;

        /*----------------.
        | Pretty printing |
        `----------------*/
        public:
          void
          print(std::ostream& s) const override;
      };
    }
  }
}
