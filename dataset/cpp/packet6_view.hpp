
#pragma once

#include "packet_view.hpp"
#include <net/ip6/packet_ip6.hpp>

namespace net::tcp {

template <typename Ptr_type> class Packet6_v;
using Packet6_view = Packet6_v<net::Packet_ptr>;
using Packet6_view_raw = Packet6_v<net::Packet*>;

template <typename Ptr_type>
class Packet6_v : public Packet_v<Ptr_type> {
public:
  Packet6_v(Ptr_type ptr)
    : Packet_v<Ptr_type>(std::move(ptr))
  {
    Expects(packet().is_ipv6());
    this->set_header(packet().ip_data().data());
  }

  inline void init();

  uint16_t compute_tcp_checksum() const noexcept override
  { return calculate_checksum6(*this); }

  Protocol ipv() const noexcept override
  { return Protocol::IPv6; }

  ip6::Addr ip6_src() const noexcept
  { return packet().ip_src(); }

  ip6::Addr ip6_dst() const noexcept
  { return packet().ip_dst(); }

private:
  PacketIP6& packet() noexcept
  { return static_cast<PacketIP6&>(*this->pkt); }

  const PacketIP6& packet() const noexcept
  { return static_cast<PacketIP6&>(*this->pkt); }

  void set_ip_src(const net::Addr& addr) noexcept override
  { packet().set_ip_src(addr.v6()); }

  void set_ip_dst(const net::Addr& addr) noexcept override
  { packet().set_ip_dst(addr.v6()); }

  net::Addr ip_src() const noexcept override
  { return packet().ip_src(); }

  net::Addr ip_dst() const noexcept override
  { return packet().ip_dst(); }

  uint16_t ip_data_length() const noexcept override
  { return packet().ip_data_length(); }

  uint16_t ip_header_length() const noexcept override
  { return packet().ip_header_len(); }
};

template <typename Ptr_type>
inline void Packet6_v<Ptr_type>::init()
{
  // clear TCP header
  memset(this->header, 0, sizeof(tcp::Header));

  this->set_win(tcp::default_window_size);
  this->header->offset_flags.offset_reserved = (5 << 4);
  this->set_length();
}

}
