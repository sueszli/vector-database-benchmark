
#ifndef NET_IP4_ICMP_ERROR_HPP
#define NET_IP4_ICMP_ERROR_HPP

#include <net/error.hpp>
#include <net/ip4/icmp4_common.hpp>

namespace net {

/**
   *  An object of this error class is sent to UDP and TCP (via Inet) when an ICMP error message
   *  is received in ICMPv4::receive
   */
  class ICMP_error : public Error {

  public:
    using ICMP_type = icmp4::Type;
    using ICMP_code = uint8_t;    // Codes in icmp4_common.hpp in namespace icmp4::code
                                  // icmp4::code::Dest_unreachable::PORT f.ex.

    /**
     * @brief      Constructor
     *             Default: No error occurred
     */
    ICMP_error()
      : Error{}
    {}

    /**
     * @brief      Constructor
     *
     * @param[in]  icmp_type  The ICMP type
     * @param[in]  icmp_code  The ICMP code
     * @param[in]  pmtu       The Path MTU (Maximum Transmission Unit for the destination)
     *                        This is set in Inet, which asks the IP layer for the most recent
     *                        Path MTU value
     */
    ICMP_error(ICMP_type icmp_type, ICMP_code icmp_code, uint16_t pmtu = 0)
      : Error{Error::Type::ICMP, "ICMP error message received"},
        icmp_type_{icmp_type}, icmp_code_{icmp_code}, pmtu_{pmtu}
    {}

    ICMP_type icmp_type() const noexcept
    { return icmp_type_; }

    std::string icmp_type_str() const
    { return icmp4::get_type_string(icmp_type_); }

    void set_icmp_type(ICMP_type icmp_type) noexcept
    { icmp_type_ = icmp_type; }

    ICMP_code icmp_code() const noexcept
    { return icmp_code_; }

    std::string icmp_code_str() const
    { return icmp4::get_code_string(icmp_type_, icmp_code_); }

    void set_icmp_code(ICMP_code icmp_code) noexcept
    { icmp_code_ = icmp_code; }

    bool is_too_big() const noexcept {
      return icmp_type_ == ICMP_type::DEST_UNREACHABLE and
        icmp_code_ == (ICMP_code) icmp4::code::Dest_unreachable::FRAGMENTATION_NEEDED;
    }

    uint16_t pmtu() const noexcept
    { return pmtu_; }

    void set_pmtu(uint16_t pmtu) noexcept
    { pmtu_ = pmtu; }

    std::string to_string() const override
    { return "ICMP " + icmp_type_str() + ": " + icmp_code_str(); }

  private:
    ICMP_type icmp_type_{ICMP_type::NO_ERROR};
    ICMP_code icmp_code_{0};
    uint16_t pmtu_{0};   // Is set if packet sent received an ICMP too big message

  };  // < class ICMP_error

} //< namespace net

#endif  //< NET_IP4_ICMP_ERROR_HPP
