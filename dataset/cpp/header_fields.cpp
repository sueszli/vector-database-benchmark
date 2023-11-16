
namespace http {
namespace header {
//------------------------------------------------
using Field = const char*;
//------------------------------------------------
// Request Fields
//------------------------------------------------
Field Accept              {"Accept"};
Field Accept_Charset      {"Accept-Charset"};
Field Accept_Encoding     {"Accept-Encoding"};
Field Accept_Language     {"Accept-Language"};
Field Authorization       {"Authorization"};
Field Cookie              {"Cookie"};
Field Connection          {"Connection"};
Field Expect              {"Expect"};
Field From                {"From"};
Field Host                {"Host"};
Field HTTP2_Settings      {"HTTP2-Settings"};
Field If_Match            {"If-Match"};
Field If_Modified_Since   {"If-Modified-Since"};
Field If_None_Match       {"If-None-Match"};
Field If_Range            {"If-Range"};
Field If_Unmodified_Since {"If-Unmodified-Since"};
Field Max_Forwards        {"Max-Forwards"};
Field Origin              {"Origin"};
Field Proxy_Authorization {"Proxy-Authorization"};
Field Range               {"Range"};
Field Referer             {"Referer"};
Field TE                  {"TE"};
Field Upgrade             {"Upgrade"};
Field User_Agent          {"User-Agent"};
//------------------------------------------------
// Response Fields
//------------------------------------------------
Field Accept_Ranges       {"Accept-Ranges"};
Field Age                 {"Age"};
Field Date                {"Date"};
Field ETag                {"ETag"};
Field Location            {"Location"};
Field Proxy_Authenticate  {"Proxy-Authenticate"};
Field Retry_After         {"Retry-After"};
Field Server              {"Server"};
Field Set_Cookie          {"Set-Cookie"};
Field Vary                {"Vary"};
Field WWW_Authenticate    {"WWW-Authenticate"};
//------------------------------------------------
// Entity Fields
//------------------------------------------------
Field Allow               {"Allow"};
Field Content_Encoding    {"Content-Encoding"};
Field Content_Language    {"Content-Language"};
Field Content_Length      {"Content-Length"};
Field Content_Location    {"Content-Location"};
Field Content_MD5         {"Content-MD5"};
Field Content_Range       {"Content-Range"};
Field Content_Type        {"Content-Type"};
Field Expires             {"Expires"};
Field Last_Modified       {"Last-Modified"};
//------------------------------------------------
//------------------------------------------------
} //< namespace header
} //< namespace http
