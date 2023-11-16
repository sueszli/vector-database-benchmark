
#include <net/dns/response.hpp>

namespace net::dns {

  ip4::Addr Response::get_first_ipv4() const
  {
    for(auto& rec : answers)
    {
      if(rec.rtype == Record_type::A)
        return rec.get_ipv4();
    }
    return ip4::Addr::addr_any;
  }

  ip6::Addr Response::get_first_ipv6() const
  {
    for(auto& rec : answers)
    {
      if(rec.rtype == Record_type::AAAA)
        return rec.get_ipv6();
    }
    return ip6::Addr::addr_any;
  }

  net::Addr Response::get_first_addr() const
  {
    for(auto& rec : answers)
    {
      if(rec.is_addr())
        return rec.get_addr();
    }
    return {};
  }

  bool Response::has_addr() const
  {
    for(auto& rec : answers)
    {
      if(rec.is_addr())
        return true;
    }
    return false;
  }

  // TODO: Verify
  int Response::parse(const char* buffer, size_t len)
  {
    Expects(len >= sizeof(Header));

    const auto& hdr = *(const Header*) buffer;

    // move ahead of the dns header and the query field
    const char* reader = (char*)buffer + sizeof(Header);
    // Iterate past the question string we sent ...
    while (*reader) reader++;
    // .. and past the question data
    reader += sizeof(Question);

    if(UNLIKELY(reader > (buffer + len)))
      return -1;

    try {
      // parse answers
      for(int i = 0; i < ntohs(hdr.ans_count); i++)
        reader += answers.emplace_back().parse(reader, buffer, len);

      // parse authorities
      for (int i = 0; i < ntohs(hdr.auth_count); i++)
        reader += auth.emplace_back().parse(reader, buffer, len);

      // parse additional
      for (int i = 0; i < ntohs(hdr.add_count); i++)
        reader += addit.emplace_back().parse(reader, buffer, len);
    }
    catch (const std::runtime_error&) {
      // packet probably too short
    }

    return reader - buffer;
  }

}
