#include <boost/uuid/random_generator.hpp>
#include <boost/uuid/uuid_io.hpp>

#include <elle/reactor/network/rdv-socket.hh>
#include <elle/reactor/network/resolve.hh>
#include <elle/reactor/network/rdv.hh>
#include <elle/reactor/network/Error.hh>
#include <elle/reactor/scheduler.hh>

ELLE_LOG_COMPONENT("rdv.socket");

namespace elle
{
  namespace reactor
  {
    namespace network
    {
      using Endpoint = boost::asio::ip::udp::endpoint;

      RDVSocket::RDVSocket()
        : _server_reached(elle::sprintf("%s: server reached", *this))
        , _breacher("breacher", [this] { this->_loop_breach(); })
        , _keep_alive("keep-alive", [this]  { this->_loop_keep_alive(); })
        , _tasks(elle::sprintf("%s tasks barrier", this))
      {}

      RDVSocket::~RDVSocket()
      {
        this->_breacher.terminate_now();
        this->_keep_alive.terminate_now();
        for (auto& c: this->_contacts)
          c.second.barrier.open();
        ELLE_DEBUG("%s: waiting for tasks to terminate...", this);
        reactor::wait(this->_tasks);
        ELLE_DEBUG("%s: waiting done", this);
      }

      void
      RDVSocket::rdv_connect(std::string const& id,
                             std::string const& rdv_host, int rdv_port,
                             DurationOpt timeout)
      {
        this->_id = id;
        this->rdv_connect(id, resolve_udp(rdv_host, rdv_port)[0],
                          timeout);
      }

      void
      RDVSocket::rdv_connect(std::string const& id, Endpoint ep,
                             DurationOpt timeout)
      {
        this->_id = id;
        ELLE_TRACE_SCOPE("rdv_connect to %s as %s", ep, id);
        this->_server_reached.close();
        this->_server = ep;
        rdv::Message req;
        req.command = rdv::Command::ping;
        req.id = id;
        elle::Buffer buf = elle::serialization::json::serialize(req, false);
        auto now = Clock::now();
        while (true)
        {
          this->_send_to_failsafe(
            elle::ConstWeakBuffer(buf.contents(), buf.size()),
            ep);
          if (reactor::wait(_server_reached, 500ms))
            return;
          else if (timeout && Clock::now() - now > *timeout)
            throw TimeOut();
        }
      }

      Size
      RDVSocket::receive_from(elle::WeakBuffer buffer,
                              boost::asio::ip::udp::endpoint &endpoint,
                              DurationOpt timeout)
      {
        while (true)
        {
          bool set_endpoint = false;
          Size sz = UDPSocket::receive_from(buffer, endpoint, timeout);
          if (sz < 8)
            return sz;
          bool server_hit = (endpoint == _server);
          auto addr = endpoint.address();
          if (endpoint.port() == _server.port()
              && addr.is_v6()
              && addr.to_v6().is_v4_mapped()
              && addr.to_v6().to_v4() == _server.address())
            server_hit = true;
          if (!this->_server_reached.opened() &&  server_hit)
          {
            ELLE_TRACE("message from server, open reached");
            this->_server_reached.open();
            set_endpoint = true;
          }
          auto magic = std::string(buffer.contents(), buffer.contents() + 8);
          auto it = this->_readers.find(magic);
          if (it != this->_readers.end())
          {
            it->second(elle::WeakBuffer(buffer.mutable_contents(), sz),
                       endpoint);
          }
          else if (magic == rdv::rdv_magic)
          {
            rdv::Message repl =
              elle::serialization::json::deserialize<rdv::Message>(
                elle::Buffer(buffer.contents() + 8, sz - 8), false);
            if (set_endpoint && repl.source_endpoint)
            {
              this->_public_endpoint = *repl.source_endpoint;
            }
            ELLE_DEBUG("got message from %s, code %s", endpoint,
                       (int)repl.command);
            switch (repl.command)
            {
            case rdv::Command::ping:
              {
                rdv::Message reply;
                reply.id = this->_id;
                reply.command = rdv::Command::pong;
                reply.source_endpoint = endpoint;
                reply.target_address = repl.target_address;
                elle::Buffer buf = elle::serialization::json::serialize(reply,
                                                                        false);
                this->_send_with_magik(buf, endpoint);
              }
              break;
            case rdv::Command::pong:
              {
                ELLE_DEBUG("pong from '%s' (%s)", repl.id, repl.target_address ?
                  *repl.target_address : "");
                auto it = this->_contacts.find(repl.id);
                if (it != this->_contacts.end())
                {
                  ELLE_TRACE("opening result barrier");
                  it->second.set_result(endpoint);
                  it->second.barrier.open();
                }
                if (repl.target_address)
                {
                  auto it = this->_contacts.find(*repl.target_address);
                  if (it != this->_contacts.end())
                  {
                    ELLE_TRACE("opening result barrier");
                    it->second.set_result(endpoint);
                    it->second.barrier.open();
                  }
                }
              }
              break;
            case rdv::Command::connect:
              {
                ELLE_TRACE("connect result tgt=%s, peer=%s",
                           *repl.target_address, !!repl.target_endpoint);
                auto it = this->_contacts.find(*repl.target_address);
                if (it != this->_contacts.end() && !it->second.barrier.opened())
                {
                  if (repl.target_endpoint)
                  {
                    // set result but do not open barrier yet, so that
                    // contact() can retry pinging it
                    it->second.set_result(*repl.target_endpoint);
                    // give it a ping
                    this->_send_ping(*repl.target_endpoint);
                  }
                  else
                  { // nothing to do, contact() will resend periodically
                  }
                }
              }
              break;
            case rdv::Command::connect_requested:
              { // add to breach requests
                ELLE_ASSERT(repl.target_endpoint);
                ELLE_TRACE("connect_requested, id=%s, ep=%s",
                  repl.id, *repl.target_endpoint);
                auto it = std::find_if(
                  this->_breach_requests.begin(),
                  this->_breach_requests.end(),
                  [&](std::pair<Endpoint, int>const& b)
                  {
                    return b.first == *repl.target_endpoint;
                  });
                if (it != _breach_requests.end())
                  it->second += 5;
                else
                  this->_breach_requests.push_back(
                    std::make_pair(*repl.target_endpoint, 5));
              }
              break;
            case rdv::Command::error:
              break;
            }
          }
          else
            return sz;
        }
      }

      Endpoint
      RDVSocket::contact(std::string const& id,
                         std::vector<Endpoint> const& endpoints,
                         DurationOpt timeout)
      {
        auto task_lock = this->_tasks.lock();
        ELLE_TRACE_SCOPE("%s: contact %s", *this, id);
        std::string tempid;
        std::string contactid = id;
        if (id.empty())
        {
          tempid = to_string(
            boost::uuids::basic_random_generator<boost::mt19937>()());
          contactid = tempid;
        }
        auto ci = this->_contacts.emplace(
          std::piecewise_construct,
          std::forward_as_tuple(contactid),
          std::forward_as_tuple(*this, contactid));
        ++ci.first->second.waiters;
        elle::SafeFinally unregister_request(
          [&]
          {
            auto it = this->_contacts.find(contactid);
            ELLE_ASSERT(it != this->_contacts.end());
            if (--it->second.waiters <= 0)
              this->_contacts.erase(it);
          });
        auto now = Clock::now();
        while (true)
        {
          if (!endpoints.empty())
          {
            // try known endpoints
            ELLE_TRACE("pinging id=%s", contactid);
            for (auto const& ep: endpoints)
              this->_send_ping(ep, contactid);
          }
          // try establishing link through RDV.
          auto const& c = this->_contacts.at(contactid);
          if (!c.barrier.opened()
              && this->_server_reached.opened()
              && !id.empty())
          {
            if (c.result && Clock::now() - c.result_time < 10s)
            {
              // RDV gave us an enpoint, but we are not connected to it yet,
              // ping it.
              this->_send_ping(*c.result);
            }
            else
            {
              rdv::Message req;
              req.command = rdv::Command::connect;
              req.id = this->_id;
              req.target_address = id;
              auto buf = elle::serialization::json::serialize(req, false);
              this->_send_to_failsafe(buf, _server);
            }
          }
          if (reactor::wait(this->_contacts.at(contactid).barrier, 500ms))
          {
            auto& c = _contacts.at(contactid);
            if (c.result)
            {
              ELLE_TRACE("got result: %s", *c.result);
              return *c.result;
            }
            else
              throw elle::Error(elle::sprintf("contact(%s) aborted", id));
          }
          else if (timeout && Clock::now() - now > *timeout)
            throw TimeOut();
        }
      }

      void
      RDVSocket::_send_with_magik(elle::Buffer const& b, Endpoint peer)
      {
        elle::Buffer data;
        data.append(reactor::network::rdv::rdv_magic, 8);
        data.append(b.contents(), b.size());
        this->_send_to_failsafe(
          elle::ConstWeakBuffer(data.contents(), data.size()),
          peer);
      }

      void
      RDVSocket::_send_ping(Endpoint target, std::string const& tid)
      {
        ELLE_DEBUG("send ping to %s", target);
        rdv::Message ping;
        ping.command = rdv::Command::ping;
        ping.id = this->_id;
        ping.source_endpoint = target;
        ping.target_address = tid;
        elle::Buffer buf = elle::serialization::json::serialize(ping, false);
        this->_send_with_magik(buf, target);
      }

      void
      RDVSocket::_loop_breach()
      {
        while (true)
        {
          std::vector<Endpoint> to_ping;
          for (int i = 0; i < signed(_breach_requests.size()); ++i)
          {
            auto& b = this->_breach_requests[i];
            to_ping.push_back(b.first);
            if (!--b.second)
            {
              std::swap(
                this->_breach_requests[i],
                this->_breach_requests[this->_breach_requests.size()-1]);
              this->_breach_requests.pop_back();
              --i;
            }
          }
          for (auto const& ep: to_ping)
            this->_send_ping(ep);
          reactor::sleep(500ms);
        }
      }

      void
      RDVSocket::_loop_keep_alive()
      {
        reactor::wait(_server_reached);
        while (true)
        {
          this->_send_ping(_server);
          reactor::sleep(30s);
        }
      }

      bool
      RDVSocket::rdv_connected() const
      {
        return _server_reached.opened();
      }

      void
      RDVSocket::set_local_id(std::string const& id)
      {
        this->_id = id;
      }

      void
      RDVSocket::register_reader(std::string const& magic,
                                 Reader handler)
      {
        ELLE_ASSERT_EQ(signed(magic.size()), 8);
        this->_readers[magic] = handler;
      }

      void
      RDVSocket::unregister_reader(std::string const& magic)
      {
        this->_readers.erase(magic);
      }

      void
      RDVSocket::_send_to_failsafe(elle::ConstWeakBuffer buffer,
                                   Endpoint endpoint)
      {
        try
        {
          send_to(buffer, endpoint);
        }
        catch (Error const& e)
        {
          ELLE_DEBUG("send_to failed with %s", e);
        }
      }

      RDVSocket::ContactInfo::ContactInfo(RDVSocket const& owner,
                                          std::string const& id)
        : barrier(elle::sprintf("%s: contact %s", owner, id))
        , waiters()
      {}

      void
      RDVSocket::ContactInfo::set_result(Endpoint ep)
      {
        this->result = ep;
        this->result_time = Clock::now();
      }
    }
  }
}
