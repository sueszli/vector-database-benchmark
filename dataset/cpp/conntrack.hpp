
#pragma once
#ifndef NET_CONNTRACK_HPP
#define NET_CONNTRACK_HPP

#include <net/socket.hpp>
#include <net/ip4/packet_ip4.hpp>
#include <net/ip6/packet_ip6.hpp>
#include <vector>
#include <unordered_map>
#include <rtc>
#include <chrono>
#include <util/timer.hpp>

namespace net {

class Conntrack {
public:
  struct Entry;
  using Entry_ptr = const Entry*;
  /**
   * Custom handler for tracking packets in a certain way
   */
  using Packet_tracker = delegate<Entry*(Conntrack&, Quadruple, const PacketIP4&)>;
  using Packet_tracker6 = delegate<Entry*(Conntrack&, Quadruple, const PacketIP6&)>;

  using Entry_handler = delegate<void(Entry*)>;

  /**
   * @brief      Key for lookup tables
   */
  struct Quintuple {
    Quadruple quad;
    Protocol  proto;

    Quintuple(Quadruple q, const Protocol p)
      : quad(std::move(q)), proto(p)
    {}

    bool operator==(const Quintuple& other) const noexcept
    { return proto == other.proto and quad == other.quad; }

    bool operator<(const Quintuple& other) const noexcept {
      return proto < other.proto
        or (proto == other.proto and quad < other.quad);
    }
  };

  /**
   * @brief      Hasher for Quintuple
   */
  struct Quintuple_hasher
  {
    std::size_t operator()(const Quintuple& key) const noexcept
    {
      const auto h1 = std::hash<Quadruple>{}(key.quad);
      const auto h2 = std::hash<uint8_t>{}(static_cast<uint8_t>(key.proto));
      return h1 ^ h2;
    }
  };

  /**
   * @brief      The state of the connection.
   */
  enum class State : uint8_t {
    NEW,
    ESTABLISHED,
    RELATED,
    UNCONFIRMED // not sure about this one
  };

  enum class Flag : uint8_t {
    UNREPLIED   = 1 << 0,
    ASSURED     = 1 << 1
  };

  /**
   * @brief      A entry in the connection tracker (a Connection)
   */
  struct Entry {
    Quadruple         first;
    Quadruple         second;
    RTC::timestamp_t  timeout;
    Protocol          proto;
    State             state;
    uint8_t           flags{0x0};
    uint8_t           other{0x0}; // whoever can make whatever here
    Entry_handler     on_close;

    Entry(Quadruple quad, Protocol p)
      : first{std::move(quad)}, second{first.dst, first.src},
        proto(p), state(State::UNCONFIRMED), on_close(nullptr)
    {}

    Entry() = default;

    bool is_mirrored() const noexcept
    { return first.src == second.dst and first.dst == second.src; }

    std::string to_string() const;

    ~Entry();

    int deserialize_from(void*);
    void serialize_to(std::vector<char>&) const;

    void set_flag(const Flag f)
    { flags |= static_cast<uint8_t>(f); }

    void unset_flag(const Flag f)
    { flags &= ~static_cast<uint8_t>(f); }

    bool isset(const Flag f) const noexcept
    { return flags & static_cast<uint8_t>(f); }

  };

  using Timeout_duration = std::chrono::seconds;
  struct Timeout_settings {
    Timeout_duration tcp;
    Timeout_duration udp;
    Timeout_duration icmp;

    Timeout_duration get(const Protocol proto) const noexcept
    {
      switch(proto) {
        case Protocol::TCP: return tcp;
        case Protocol::UDP: return udp;
        case Protocol::ICMPv4: return icmp;
        default: return Timeout_duration{0};
      }
    }
  };

public:
  /** Maximum number of conntrack entries. */
  // 0 means unlimited. Every new connection result in 2 entries.
  size_t maximum_entries;

  struct {
    Timeout_settings unconfirmed{ .tcp  = Timeout_duration{10},
                                  .udp  = Timeout_duration{10},
                                  .icmp = Timeout_duration{10}};

    Timeout_settings confirmed  { .tcp  = Timeout_duration{30},
                                  .udp  = Timeout_duration{10},
                                  .icmp = Timeout_duration{10}};

    Timeout_settings established{ .tcp  = Timeout_duration{300},
                                  .udp  = Timeout_duration{10},
                                  .icmp = Timeout_duration{10}};
  } timeout;
  /**
   * @brief      Find the entry for the given packet
   *
   * @param[in]  pkt   The packet
   *
   * @return     A matching conntrack entry (nullptr if not found)
   */
  Entry* get(const PacketIP4& pkt) const;
  Entry* get(const PacketIP6& pkt) const;

  /**
   * @brief      Find the entry where the quadruple
   *             with the given protocol matches.
   *
   * @param[in]  quad   The quad
   * @param[in]  proto  The prototype
   *
   * @return     A matching conntrack entry (nullptr if not found)
   */
  Entry* get(const Quadruple& quad, const Protocol proto) const;

  /**
   * @brief      Track a packet, updating the state of the entry.
   *
   * @param[in]  pkt   The packet
   *
   * @return     The conntrack entry related to this packet.
   */
  Entry* in(const PacketIP4& pkt);
  Entry* in(const PacketIP6& pkt);

  /**
   * @brief      Confirms a connection, moving the entry to confirmed.
   *
   * @param[in]  pkt   The packet
   *
   * @return     The confirmed entry, if any
   */
  Entry* confirm(const PacketIP4& pkt);
  Entry* confirm(const PacketIP6& pkt);

  /**
   * @brief      Confirms a connection, moving the entry to confirmed
   *             and indexing it both ways.
   *
   * @param[in]  quad   The quad
   * @param[in]  proto  The prototype
   *
   * @return     The confirmed entry, if any
   */
  Entry* confirm(Quadruple quad, const Protocol proto);

  /**
   * @brief      Adds an entry as unconfirmed, mirroring the quadruple.
   *
   * @param[in]  quad   The quadruple
   * @param[in]  proto  The prototype
   * @param[in]  dir    The direction the packet is going
   *
   * @return     The created entry
   */
  Entry* add_entry(const Quadruple& quad, const Protocol proto);

  /**
   * @brief      Update one quadruple of a old entry (proto + oldq)
   *             to a new Quadruple. This changes the entry and updates the key.
   *
   * @param[in]  proto  The protocol
   * @param[in]  oldq   The old (current) quadruple
   * @param[in]  newq   The new quadruple
   */
  Entry* update_entry(const Protocol proto, const Quadruple& oldq, const Quadruple& newq);

  /**
   * @brief      Remove all expired entries, both confirmed and unconfirmed.
   */
  void remove_expired();

  /**
   * @brief      Number of entries currently tracked.
   *
   * @return     Number of entries.
   */
  size_t number_of_entries() const noexcept
  { return entries.size(); }

  /**
   * @brief      Call reserve on the underlying unordered_map
   *
   * @param[in]  count  The count
   */
  void reserve(size_t count)
  { entries.reserve(count); }

  /**
   * @brief      A very simple and unreliable way for tracking quintuples.
   *
   * @param[in]  quad   The quad
   * @param[in]  proto  The prototype
   *
   * @return     The conntrack entry related to quintuple.
   */
  Entry* simple_track_in(Quadruple quad, const Protocol proto);

  /**
   * @brief      Gets the quadruple from a IP packet.
   *             Assumes the packet has protocol specific payload.
   *
   * @param[in]  pkt   The packet
   *
   * @return     The quadruple.
   */
  template <typename IP_packet>
  static Quadruple get_quadruple(const IP_packet& pkt);

  /**
   * @brief      Gets the quadruple from a IP packet carrying
   *             ICMP payload
   *
   * @param[in]  pkt   The packet
   *
   * @return     The quadruple for ICMP.
   */
  template <typename IP_packet>
  static Quadruple get_quadruple_icmp(const IP_packet& pkt);

  /**
   * @brief      Construct a Conntrack with unlimited maximum entries.
   */
  Conntrack();

  /**
   * @brief      Construct a Conntrack with a given limit of entries.
   *
   * @param[in]  max_entries  The maximum number of entries
   */
  Conntrack(size_t max_entries);

  /** How often the flush timer should fire */
  std::chrono::seconds flush_interval {10};

  /** Custom TCP handler can (and should) be added here */
  Packet_tracker  tcp_in;
  Packet_tracker6 tcp6_in;

  int deserialize_from(void*);
  void serialize_to(std::vector<char>&) const;

private:
  using Entry_table = std::unordered_map<Quintuple, std::shared_ptr<Entry>, Quintuple_hasher>;
  Entry_table entries;
  Timer       flush_timer;

  inline void update_timeout(Entry& ent, const Timeout_settings& timeouts);

  void on_timeout();

};

template <typename IP_packet>
inline Quadruple Conntrack::get_quadruple(const IP_packet& pkt)
{
  Expects(pkt.ip_protocol() == Protocol::TCP or pkt.ip_protocol() == Protocol::UDP);

  const auto* ports = reinterpret_cast<const uint16_t*>(pkt.ip_data().data());
  uint16_t src_port = ntohs(*ports);
  uint16_t dst_port = ntohs(*(ports + 1));

  return {{pkt.ip_src(), src_port}, {pkt.ip_dst(), dst_port}};
}

template <typename IP_packet>
inline Quadruple Conntrack::get_quadruple_icmp(const IP_packet& pkt)
{
  Expects(pkt.ip_protocol() == Protocol::ICMPv4 or pkt.ip_protocol() == Protocol::ICMPv6);

  struct partial_header {
    uint16_t  type_code;
    uint16_t  checksum;
    uint16_t  id;
  };

  auto id = reinterpret_cast<const partial_header*>(pkt.ip_data().data())->id;

  return {{pkt.ip_src(), id}, {pkt.ip_dst(), id}};
}

inline void Conntrack::update_timeout(Entry& ent, const Timeout_settings& timeouts)
{
  ent.timeout = RTC::now() + timeouts.get(ent.proto).count();
}

}

#endif
