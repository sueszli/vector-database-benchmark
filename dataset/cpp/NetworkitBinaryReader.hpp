/*
 * NetworkitBinaryReader.hpp
 *
 *      Author: Charmaine Ndolo <charmaine.ndolo@hu-berlin.de>
 */

#ifndef NETWORKIT_IO_NETWORKIT_BINARY_READER_HPP_
#define NETWORKIT_IO_NETWORKIT_BINARY_READER_HPP_

#include <cstring>
#include <string>
#include <networkit/graph/Graph.hpp>
#include <networkit/io/GraphReader.hpp>
#include <networkit/io/MemoryMappedFile.hpp>

namespace NetworKit {

/**
 * @ingroup io
 *
 * Reads a graph written in the custom Networkit format documented in cpp/io/NetworkitGraph.md
 */

class NetworkitBinaryReader final : public GraphReader {

public:
    NetworkitBinaryReader(){};

    Graph read(const std::string &path) override;
    Graph readFromBuffer(const std::vector<uint8_t> &data);

private:
    count nodes;
    count chunks;
    bool directed;
    bool weighted;
    bool indexed;
    count version;

    template <class T>
    Graph readData(const T &source);
};
} // namespace NetworKit

#endif // NETWORKIT_IO_NETWORKIT_BINARY_READER_HPP_
