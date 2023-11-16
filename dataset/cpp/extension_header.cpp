
#include <net/ip6/extension_header.hpp>
#include <net/ip6/packet_ip6.hpp>

namespace net::ip6 {

  Protocol parse_upper_layer_proto(const uint8_t* reader, const uint8_t* end,  Protocol proto)
  {
    while (proto != Protocol::IPv6_NONXT)
    {
      // bounds check
      if (reader + sizeof(ip6::Extension_header) >= end)
      {
        // the packet is invalid
        return Protocol::IPv6_NONXT;
      }

      switch (proto)
      {
        // One of these should be the last one, and isn't a IP6 option.
        case Protocol::TCP:
        case Protocol::UDP:
        case Protocol::ICMPv6:
          return proto;

        // Currently just ignore and iterate to next one header
        default:
        {
          auto& ext = *(Extension_header*)reader;
          proto = ext.proto();
          reader += ext.size();
        }
      }
    }

    return proto;
  }

  uint16_t parse_extension_headers(const Extension_header* start, Protocol proto,
                                   Extension_header_inspector on_ext_hdr)
  {
    const auto* reader = (uint8_t*) start;
    uint16_t n = 0;

    // TODO: Verify options. If corrupt options, the loop will go forever.
    while(proto != Protocol::IPv6_NONXT)
    {
      auto* ext = (Extension_header*)reader;
      on_ext_hdr(ext);

      switch(proto)
      {
        // One of these should be the last one, and isn't a IP6 option.
        case Protocol::TCP:
        case Protocol::UDP:
        case Protocol::ICMPv6:
          return n;

        // Currently just ignore and iterate to next one header
        default:
        {
          const auto sz = ext->size();
          proto = ext->proto();
          reader += sz;
          n += sz;
        }
      }
    }

    return n;
  }
}
