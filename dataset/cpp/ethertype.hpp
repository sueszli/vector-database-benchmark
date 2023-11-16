
#pragma once
#ifndef NET_ETHERTYPE_HPP
#define NET_ETHERTYPE_HPP


namespace net {

  /** Little-endian ethertypes. More entries here:
      http://www.iana.org/assignments/ieee-802-numbers/ieee-802-numbers.xhtml */
  enum class Ethertype : uint16_t {
    IP4           = 0x8,
    ARP           = 0x608,
    WOL           = 0x4208,
    IP6           = 0xdd86,
    FLOW          = 0x888,
    JUMBO         = 0x7088,
    VLAN          = 0x81,
    TRAILER_NEGO  = 0x0010,   // RFC 893 Trailer negotiation
    TRAILER_FIRST = 0x0110,   // RFC 893, 1122 Trailer encapsulation
    TRAILER_LAST  = 0x0f10
  };

}

#endif
