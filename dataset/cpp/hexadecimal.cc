# include <elle/format/hexadecimal.hh>
# include <elle/Buffer.hh>

/*----------------.
| Macro-functions |
`----------------*/

#define ELLE_FORMAT_HEXADECIMAL_TO_VALUE(c)                             \
  (c >= '0' && c <= '9' ?                                               \
   c - '0' :                                                            \
   (c >= 'a' && c <= 'f' ? c - 'a' + 10 : -1))

namespace elle
{
  namespace format
  {
    namespace hexadecimal
    {
      Buffer
      decode(std::string const& string)
      {
        Buffer buffer;

        decode(string, buffer);

        return (buffer);
      }

      void
      decode(std::string const& string,
             Buffer& buffer)
      {
        char const* src = string.c_str();
        size_t src_size = string.size();
        size_t dst_size = src_size / 2;
        ELLE_ASSERT(src_size % 2 == 0);
        uint8_t* dst = nullptr;
          {
            size_t old_size = buffer.size();
            buffer.size(old_size + dst_size);
            dst = buffer.mutable_contents() + old_size;
          }
        ELLE_ASSERT(dst != nullptr);

        size_t i = 0, j = 0;
        while (i < src_size)
          {
            char c1 = src[i++];
            char c2 = src[i++];

            char v1 = ELLE_FORMAT_HEXADECIMAL_TO_VALUE(c1);
            char v2 = ELLE_FORMAT_HEXADECIMAL_TO_VALUE(c2);
            if (v1 == -1 || v2 == -1)
              throw std::runtime_error{
                  "Invalid char found in hexadecimal stream"
              };
            dst[j++] = ((v1 << 4) & 0xf0) + (v2 & 0xf);
          }
      }

      std::string
      encode(ConstWeakBuffer buffer)
      {
        std::string string;
        encode(buffer, string);
        return (string);
      }

      void
      encode(ConstWeakBuffer buffer,
             std::string& string)
      {
        if (!buffer.empty())
        {
          static char const* chars = "0123456789abcdef";
          string.reserve(string.size() + buffer.size() * 2);
          for (auto c: buffer)
          {
            string.push_back(chars[(c >> 4) & 0xf]);
            string.push_back(chars[c & 0xf]);
          }
        }
      }
    }
  }
}
