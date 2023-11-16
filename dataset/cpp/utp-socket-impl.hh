#pragma once

#include <elle/reactor/network/utp-socket.hh>

#include <elle/reactor/Barrier.hh>
#include <elle/reactor/MultiLockBarrier.hh>
#include <elle/reactor/mutex.hh>

#include <utp.h>

namespace elle
{
  namespace reactor
  {
    namespace network
    {
      class UTPSocket::Impl
      {
      /*------.
      | Types |
      `------*/
      public:
        /// Import from libutp/utp.h.
        using utp_socket = ::UTPSocket;
        /// From reactor/network/utp-socket.hh.
        friend class UTPSocket;

      /*-------------.
      | Construction |
      `-------------*/
      public:
        Impl(std::weak_ptr<UTPServer::Impl> server,
             utp_socket* socket, bool open);
        ~Impl();

      /*-----------.
      | Attributes |
      `-----------*/
      public:
        UTPSocket::EndPoint
        peer() const;
      private:
        ELLE_ATTRIBUTE(utp_socket*, socket);
        ELLE_ATTRIBUTE(elle::Buffer, read_buffer);
        ELLE_ATTRIBUTE(Barrier, read_barrier);
        ELLE_ATTRIBUTE(Barrier, write_barrier);
        ELLE_ATTRIBUTE(Mutex, write_mutex);
        ELLE_ATTRIBUTE(Barrier, connect_barrier);
        ELLE_ATTRIBUTE(Barrier, destroyed_barrier);
        ELLE_ATTRIBUTE_R(std::weak_ptr<UTPServer::Impl>, server);
        ELLE_ATTRIBUTE(elle::ConstWeakBuffer, write);
        ELLE_ATTRIBUTE(MultiLockBarrier, pending_operations);
        ELLE_ATTRIBUTE(int, write_pos);
        ELLE_ATTRIBUTE(bool, open);
        ELLE_ATTRIBUTE(bool, closing);

      /*----------.
      | Callbacks |
      `----------*/
      public:
        void
        on_connect();
        void
        on_close();
        void
        on_read(elle::ConstWeakBuffer const&);

      /*----------.
      | Operation |
      `----------*/
      public:
        void
        _destroyed();
        void
        _write_cont();
      private:
        void
        _read();
      };

      std::ostream&
      operator <<(std::ostream& output, UTPSocket::Impl const& impl);
    }
  }
}
