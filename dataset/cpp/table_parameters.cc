#include "engine/api/table_parameters.hpp"
#include "server/api/parameters_parser.hpp"

#include "util.hpp"

#include <iterator>
#include <string>

using osrm::server::api::parseParameters;
using osrm::engine::api::TableParameters;

extern "C" int LLVMFuzzerTestOneInput(const unsigned char *data, unsigned long size)
{
    std::string in(reinterpret_cast<const char *>(data), size);

    auto first = begin(in);
    const auto last = end(in);

    const auto param = parseParameters<TableParameters>(first, last);
    escape(&param);

    return 0;
}
