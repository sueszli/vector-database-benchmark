/*
 * Usage: realm-trawler <realm-file-name>
 *
 * This tool will dump the structure of a realm file and print out any inconsistencies it finds.
 *
 * First it will print out information found in the top group. If there are inconsistencies in the
 * free list, this will be reported.
 *
 * Next, it will go through all tables and print the name, type and primary structure of the columns
 * found in the table. The user data found in the tables will not be interpreted.
 *
 * Generally all references will be checked in the sense that they should point to something that has
 * a valid header, meaning that the header must have a valid signature. Also, references that point
 * to areas included in the free list will be considered invalid. References that are not valid
 * will not be followed. It is checked that an area is only referenced once.
 *
 * Lastly it is checked that all space is accounted for. The combination of the free list and the
 * table tree should cover the whole file. Any leaked areas are reported.
 */

#include <realm/array_direct.hpp>
#include <realm/alloc_slab.hpp>
#include <realm/keys.hpp>
#include <realm/array.hpp>
#include <realm/column_type.hpp>
#include <realm/data_type.hpp>
#include <realm/table.hpp>
#include <realm/impl/transact_log.hpp>

#include <iostream>
#include <fstream>
#include <sstream>
#include <cstring>

constexpr const int signature = 0x41414141;
uint64_t current_logical_file_size;

struct Header {
    uint64_t m_top_ref[2]; // 2 * 8 bytes
    // Info-block 8-bytes
    uint8_t m_mnemonic[4];    // "T-DB"
    uint8_t m_file_format[2]; // See `library_file_format`
    uint8_t m_reserved;
    // bit 0 of m_flags is used to select between the two top refs.
    uint8_t m_flags;
};

struct StreamingFooter {
    uint64_t m_top_ref;
    uint64_t m_magic_cookie;
};

template <class T>
void consolidate_lists(std::vector<T>& list, std::vector<T>& list2)
{
    list.insert(list.end(), list2.begin(), list2.end());
    list2.clear();
    if (list.size() > 1) {
        std::sort(begin(list), end(list), [](T& a, T& b) {
            return a.start < b.start;
        });

        auto prev = list.begin();
        for (auto it = list.begin() + 1; it != list.end(); ++it) {
            if (prev->start + prev->length != it->start) {
                if (prev->start + prev->length > it->start) {
                    std::cout << "*** Overlapping entries:" << std::endl;
                    std::cout << std::hex;
                    std::cout << "    0x" << prev->start << "..0x" << prev->start + prev->length << std::endl;
                    std::cout << "    0x" << it->start << "..0x" << it->start + it->length << std::endl;
                    std::cout << std::dec;
                    // Remove new entry from list
                    it->length = 0;
                }
                prev = it;
                continue;
            }

            prev->length += it->length;
            it->length = 0;
        }

        // Remove all of the now zero-size chunks from the free list
        list.erase(std::remove_if(begin(list), end(list),
                                  [](T& chunk) {
                                      return chunk.length == 0;
                                  }),
                   end(list));
    }
}

struct Entry {
    Entry()
        : Entry(0, 0)
    {
    }
    Entry(uint64_t s, uint64_t l)
        : start(s)
        , length(l)
    {
    }
    bool operator<(const Entry& rhs) const
    {
        return start < rhs.start;
    }
    uint64_t start;
    uint64_t length;
};

struct FreeListEntry : public Entry {
    FreeListEntry()
        : FreeListEntry(0, 0)
    {
    }
    FreeListEntry(uint64_t s, uint64_t l, uint64_t v = 0)
        : Entry(s, l)
        , version(v)
    {
    }
    uint64_t version;
};

class Node {
public:
    Node() {}
    Node(realm::Allocator& alloc, uint64_t ref)
    {
        init(alloc, ref);
    }
    void init(realm::Allocator& alloc, uint64_t ref);
    bool valid() const
    {
        return m_valid;
    }
    bool has_refs() const
    {
        return (m_header[4] & 0x40) != 0;
    }
    unsigned width() const
    {
        return (1 << (unsigned(m_header[4]) & 0x07)) >> 1;
    }
    unsigned size() const
    {
        return unsigned(m_size);
    }
    unsigned length() const
    {
        unsigned width_type = (unsigned(m_header[4]) & 0x18) >> 3;
        return calc_byte_size(width_type, m_size, width());
    }
    uint64_t ref() const
    {
        return m_ref;
    }
    uint64_t size_in_bytes() const
    {
        return 8 + length();
    }
    char* data()
    {
        return realm::Array::get_data_from_header(m_header);
    }

protected:
    uint64_t m_ref = 0;
    char* m_header;
    unsigned m_size = 0;
    bool m_valid = false;

    static unsigned calc_byte_size(unsigned wtype, unsigned size, unsigned width)
    {
        unsigned num_bytes = 0;
        switch (wtype) {
            case 0: {
                unsigned num_bits = size * width;
                num_bytes = (num_bits + 7) >> 3;
                break;
            }
            case 1: {
                num_bytes = size * width;
                break;
            }
            case 2:
                num_bytes = size;
                break;
        }

        // Ensure 8-byte alignment
        return (num_bytes + 7) & ~size_t(7);
    }
};

class Array : public Node {
public:
    Array() {}
    Array(realm::Allocator& alloc, uint64_t ref)
        : Node(alloc, ref)
    {
        init(alloc, ref);
    }
    bool is_inner_bptree_node() const
    {
        return realm::NodeHeader::get_is_inner_bptree_node_from_header(m_header);
    }
    void init(realm::Allocator& alloc, uint64_t ref)
    {
        Node::init(alloc, ref);
        m_data = data();
        m_has_refs = has_refs();
    }
    int64_t get_val(size_t ndx) const
    {
        int64_t val = realm::get_direct(m_data, width(), ndx);

        if (m_has_refs) {
            if (val & 1) {
                val >>= 1;
            }
        }
        return val;
    }
    uint64_t get_ref(size_t ndx) const
    {
        REALM_ASSERT(m_has_refs);
        int64_t val = realm::get_direct(m_data, width(), ndx);

        if (val & 1)
            return 0;

        uint64_t ref = uint64_t(val);
        if (ref > current_logical_file_size || (ref & 7)) {
            std::cout << "*** Invalid ref: 0x" << std::hex << ref << std::dec << std::endl;
            return 0;
        }

        return ref;
    }
    std::string get_string(size_t ndx) const
    {
        std::string str;
        if (valid()) {
            auto sz = size_in_bytes();
            auto w = width();
            REALM_ASSERT(ndx * w < sz);
            str = m_data + ndx * w;
        }
        return str;
    }

    uint64_t mem_usage(realm::Allocator& alloc) const
    {
        uint64_t mem = 0;
        _mem_usage(alloc, mem);
        return mem;
    }

private:
    char* m_data;
    bool m_has_refs = false;

    void _mem_usage(realm::Allocator& alloc, uint64_t& mem) const
    {
        if (m_has_refs) {
            for (size_t i = 0; i < m_size; ++i) {
                if (uint64_t ref = get_ref(i)) {
                    Array subarray(alloc, ref);
                    subarray._mem_usage(alloc, mem);
                }
            }
        }
        mem += size_in_bytes();
    }
};

class Group;
class Table : public Array {
public:
    Table() = default;
    Table(realm::Allocator& alloc, uint64_t ref)
        : Array(alloc, ref)
    {
        if (valid()) {
            Array spec(alloc, get_ref(0));
            m_column_types.init(alloc, spec.get_ref(0));
            m_column_names.init(alloc, spec.get_ref(1));
            m_column_attributes.init(alloc, spec.get_ref(2));
            if (spec.size() > 5) {
                // Must be a Core-6 file.
                m_enum_keys.init(alloc, spec.get_ref(4));
                m_column_colkeys.init(alloc, spec.get_ref(5));

                size_t num_spec_cols = m_column_types.size();

                for (size_t spec_ndx = 0; spec_ndx < num_spec_cols; ++spec_ndx) {
                    realm::ColKey col_key{m_column_colkeys.get_val(spec_ndx)};
                    unsigned leaf_ndx = col_key.get_index().val;
                    if (leaf_ndx >= m_leaf_ndx2spec_ndx.size()) {
                        m_leaf_ndx2spec_ndx.resize(leaf_ndx + 1, -1);
                    }
                    m_leaf_ndx2spec_ndx[leaf_ndx] = spec_ndx;
                }
            }
            else if (spec.size() > 3) {
                // In pre Core-6, the subspecs array is optional
                // Not used in Core-6
                m_column_subspecs.init(alloc, spec.get_ref(3));
            }
            if (size() > 7) {
                // Must be a Core-6 file.
                m_clusters.init(alloc, get_ref(2));
                m_opposite_table.init(alloc, get_ref(7));
            }
            if (size() > 11) {
                auto pk_col = get_val(11);
                if (pk_col)
                    m_pk_col = realm::ColKey(pk_col);
            }
            if (size() > 12) {
                auto flags = get_val(12);
                m_table_type = static_cast<realm::Table::Type>(flags & 0x3);
            }
        }
    }
    std::string get_column_name(realm::ColKey col_key) const
    {
        return m_column_names.get_string(m_leaf_ndx2spec_ndx[col_key.get_index().val]);
    }
    void print_columns(const Group&) const;
    size_t get_size(realm::Allocator& alloc) const
    {
        size_t ret = 0;
        if (m_clusters.valid()) {
            if (m_clusters.is_inner_bptree_node()) {
                ret = size_t(m_clusters.get_val(2));
            }
            else {
                if (uint64_t key_ref = m_clusters.get_ref(0)) {
                    auto header = alloc.translate(realm::to_ref(key_ref));
                    ret = realm::NodeHeader::get_size_from_header(header);
                }
                else {
                    ret = (size_t)m_clusters.get_val(0);
                }
            }
        }
        return ret;
    }

private:
    size_t get_subspec_ndx_after(size_t column_ndx) const noexcept
    {
        REALM_ASSERT(column_ndx <= m_column_names.size());
        // The m_subspecs array only keep info for subtables so we need to
        // count up to it's position
        size_t subspec_ndx = 0;
        for (size_t i = 0; i != column_ndx; ++i) {
            auto type = realm::ColumnType(m_column_types.get_val(i));
            if (type == realm::col_type_Link || type == realm::col_type_LinkList) {
                subspec_ndx += 1; // index of dest column
            }
            else if (type == realm::col_type_BackLink) {
                subspec_ndx += 2; // index of table and index of linked column
            }
        }
        return subspec_ndx;
    }

    Array m_column_types;
    Array m_column_names;
    Array m_column_attributes;
    Array m_enum_keys;
    Array m_column_subspecs;
    Array m_column_colkeys;
    Array m_opposite_table;
    Array m_clusters;
    realm::ColKey m_pk_col;
    realm::Table::Type m_table_type = realm::Table::Type::TopLevel;
    std::vector<size_t> m_leaf_ndx2spec_ndx;
};

class Group : public Array {
public:
    Group(realm::Allocator& alloc, uint64_t ref)
        : Array(alloc, ref)
        , m_alloc(alloc)
    {
        m_valid &= (size() <= 12);
        if (valid()) {
            m_file_size = get_val(2);
            current_logical_file_size = m_file_size;
            m_table_names.init(alloc, get_ref(0));
            m_table_refs.init(alloc, get_ref(1));
            if (size() > 3) {
                m_free_list_positions.init(alloc, get_ref(3));
                m_free_list_sizes.init(alloc, get_ref(4));
                m_free_list_versions.init(alloc, get_ref(5));
            }
            if (size() > 8) {
                m_history.init(alloc, get_ref(8));
            }
            if (size() > 11) {
                auto ref = get_ref(11);
                if (ref)
                    m_evacuation_info.init(alloc, ref);
            }
        }
    }
    uint64_t get_file_size() const
    {
        return m_file_size;
    }
    uint64_t get_free_space_size() const
    {
        uint64_t sz = 0;
        for (size_t i = 0; i < m_free_list_sizes.size(); i++) {
            sz += m_free_list_sizes.get_val(i);
        }
        return sz;
    }
    int get_current_version() const
    {
        return int(get_val(6));
    }
    std::string get_history_type() const
    {
        switch (int(get_val(7))) {
            case 0:
                return "None";
            case 1:
                return "OutOfRealm";
            case 2:
                return "InRealm";
            case 3:
                return "SyncClient";
            case 4:
                return "SyncServer";
        }
        return "Unknown";
    }
    std::vector<realm::BinaryData> get_changesets()
    {
        std::vector<realm::BinaryData> ret;
        if (int(get_val(7)) == 2) {
            for (size_t n = 0; n < m_history.size(); n++) {
                auto ref = m_history.get_ref(n);
                Node node(m_alloc, ref);
                ret.emplace_back(node.data(), node.size());
            }
        }
        if (int(get_val(7)) == 3) {
            auto ref = m_history.get_ref(0); // ct history
            m_history.init(m_alloc, ref);
            for (size_t n = 0; n < m_history.size(); n++) {
                ref = m_history.get_ref(n);
                Node node(m_alloc, ref);
                ret.emplace_back(node.data(), node.size());
            }
        }
        return ret;
    }
    int get_history_schema_version() const
    {
        return int(get_val(9));
    }
    int get_file_ident() const
    {
        return int(get_val(10));
    }
    void print_evacuation_info(std::ostream& ostr) const
    {
        if (m_evacuation_info.valid()) {
            ostr << "Evacuation limit: " << size_t(m_evacuation_info.get_val(0));
            if (m_evacuation_info.get_val(1)) {
                ostr << " Scan done" << std::endl;
            }
            else {
                ostr << " Progress: [";
                for (size_t i = 2; i < m_evacuation_info.size(); i++) {
                    if (i > 2)
                        ostr << ',';
                    ostr << m_evacuation_info.get_val(i);
                }
                ostr << "]" << std::endl;
            }
        }
    }
    unsigned get_nb_tables() const
    {
        return m_table_names.size();
    }
    std::string get_table_name(size_t i) const
    {
        return m_table_names.get_string(i);
    }
    Table* get_table(size_t i) const
    {
        auto& ret = m_tables[i];
        if (!ret) {
            ret = new Table(m_alloc, m_table_refs.get_ref(i));
        }
        return ret;
    }
    std::vector<Entry> get_allocated_nodes() const;
    std::vector<FreeListEntry> get_free_list() const;
    void print_schema() const;

private:
    friend std::ostream& operator<<(std::ostream& ostr, const Group& g);
    realm::Allocator& m_alloc;
    uint64_t m_file_size;
    Array m_table_names;
    Array m_table_refs;
    Array m_free_list_positions;
    Array m_free_list_sizes;
    Array m_free_list_versions;
    Array m_evacuation_info;
    Array m_history;
    mutable std::map<size_t, Table*> m_tables;
};

class RealmFile {
public:
    RealmFile(const std::string& file_path, const char* encryption_key, uint64_t top_ref = 0);
    // Walk the file and check that it consists of valid nodes
    void node_scan();
    void schema_info();
    void memory_leaks();
    void free_list_info() const;
    void changes() const;

private:
    uint64_t m_top_ref;
    uint64_t m_start_pos;
    int m_file_format_version;
    std::unique_ptr<Group> m_group;
    realm::SlabAlloc m_alloc;
};

static std::string human_readable(uint64_t val)
{
    std::ostringstream out;
    out.precision(3);
    if (val < 1024) {
        out << val;
    }
    else if (val < 1024 * 1024) {
        out << (double(val) / 1024) << "K";
    }
    else if (val < 1024 * 1024 * 1024) {
        out << (double(val) / (1024 * 1024)) << "M";
    }
    else {
        out << (double(val) / (1024 * 1024 * 1024)) << "G";
    }
    return out.str();
}

static uint64_t get_size(const std::vector<Entry>& list)
{
    uint64_t sz = 0;
    std::for_each(list.begin(), list.end(), [&](const Entry& e) {
        sz += e.length;
    });
    return sz;
}

std::ostream& operator<<(std::ostream& ostr, const Group& g)
{
    if (g.valid()) {
        ostr << "Logical file size: " << human_readable(g.get_file_size()) << std::endl;
        if (g.size() > 11) {
            g.print_evacuation_info(ostr);
        }
        if (g.size() > 6) {
            ostr << "Current version: " << g.get_current_version() << std::endl;
            ostr << "Free list size: " << g.m_free_list_positions.size() << std::endl;
            ostr << "Free space size: " << human_readable(g.get_free_space_size()) << std::endl;
        }
        if (g.size() > 8) {
            ostr << "History type: " << g.get_history_type() << std::endl;
            ostr << "History schema version: " << g.get_history_schema_version() << std::endl;
            ostr << "History size: " << human_readable(g.m_history.mem_usage(g.m_alloc)) << std::endl;
        }
        if (g.size() > 10) {
            ostr << "File ident: " << g.get_file_ident() << std::endl;
        }
    }
    else {
        ostr << "*** Invalid group ***" << std::endl;
    }
    return ostr;
}

void Table::print_columns(const Group& group) const
{
    std::cout << "        <" << m_table_type << ">" << std::endl;
    for (unsigned i = 0; i < m_column_names.size(); i++) {
        auto type = realm::ColumnType(m_column_types.get_val(i) & 0xFFFF);
        auto attr = realm::ColumnAttr(m_column_attributes.get_val(i));
        std::string type_str;
        realm::ColKey col_key;
        if (this->m_column_colkeys.valid()) {
            // core6
            col_key = realm::ColKey(m_column_colkeys.get_val(i));
        }

        if (type == realm::col_type_Link || type == realm::col_type_LinkList) {
            size_t target_table_ndx;
            if (col_key) {
                // core6
                realm::TableKey opposite_table_key(uint32_t(m_opposite_table.get_val(col_key.get_index().val)));
                target_table_ndx = opposite_table_key.value & 0xFFFF;
            }
            else {
                target_table_ndx = size_t(m_column_subspecs.get_val(get_subspec_ndx_after(i)));
            }
            type_str += group.get_table_name(target_table_ndx);
            if (!col_key && type == realm::col_type_LinkList) {
                type_str += "[]";
            }
        }
        else {
            type_str = get_data_type_name(realm::DataType(type));
        }
        if (col_key) {
            if (col_key.is_list())
                type_str += "[]";
            if (col_key.is_set())
                type_str += "{}";
            if (col_key.is_dictionary()) {
                auto key_type = realm::DataType(int(m_column_types.get_val(i)) >> 16);
                type_str = std::string("{") + get_data_type_name(key_type) + ", " + type_str + "}";
            }
        }
        if (attr & realm::col_attr_Nullable)
            type_str += "?";
        if (attr & realm::col_attr_Indexed)
            type_str += " (indexed)";
        if (m_enum_keys.valid() && m_enum_keys.get_val(i)) {
            type_str += " (enumerated)";
        }
        std::string star = (m_pk_col && (m_pk_col == col_key)) ? "*" : "";
        std::cout << "        " << i << ": " << star << m_column_names.get_string(i) << ": " << type_str << std::endl;
    }
}

void Group::print_schema() const
{
    if (valid()) {
        std::cout << "Tables: " << std::endl;

        for (unsigned i = 0; i < get_nb_tables(); i++) {
            Table* table = get_table(i);
            std::cout << "    " << i << ": " << get_table_name(i) << " - size: " << table->get_size(m_alloc)
                      << " datasize: " << human_readable(table->mem_usage(m_alloc)) << std::endl;
            table->print_columns(*this);
        }
    }
}

void Node::init(realm::Allocator& alloc, uint64_t ref)
{
    m_ref = ref;
    m_header = alloc.translate(realm::to_ref(ref));

    if (memcmp(m_header, &signature, 4)) {
    }
    else {
        unsigned char* u = reinterpret_cast<unsigned char*>(m_header);
        m_size = (u[5] << 16) + (u[6] << 8) + u[7];
        m_valid = true;
    }
}

std::vector<unsigned> path;

static std::string print_path()
{
    std::string ret = "[" + std::to_string(path[0]);
    for (auto it = path.begin() + 1; it != path.end(); ++it) {
        ret += ", ";
        ret += std::to_string(*it);
    }
    return ret + "]";
}

static std::vector<Entry> get_nodes(realm::Allocator& alloc, uint64_t ref)
{
    std::vector<Entry> nodes;
    if (ref != 0) {
        Array arr(alloc, ref);
        if (!arr.valid()) {
            std::cout << "Not and array: 0x" << std::hex << ref << std::dec << ", path: " << print_path()
                      << std::endl;
            return {};
        }
        nodes.emplace_back(ref, arr.size_in_bytes());
        if (arr.has_refs()) {
            auto sz = arr.size();
            path.push_back(0);
            for (unsigned i = 0; i < sz; i++) {
                uint64_t r = arr.get_ref(i);
                if (r) {
                    path.back() = i;
                    auto sub_nodes = get_nodes(alloc, r);
                    consolidate_lists(nodes, sub_nodes);
                }
            }
            path.pop_back();
        }
    }
    return nodes;
}

std::vector<Entry> Group::get_allocated_nodes() const
{
    std::vector<Entry> all_nodes;
    all_nodes.emplace_back(0, 24);                  // Header area
    all_nodes.emplace_back(m_ref, size_in_bytes()); // Top array itself

    path.push_back(0);
    auto table_name_nodes = get_nodes(m_alloc, get_ref(0)); // Table names
    consolidate_lists(all_nodes, table_name_nodes);
    path.back() = 1;
    auto table_nodes = get_nodes(m_alloc, get_ref(1)); // Tables
    consolidate_lists(all_nodes, table_nodes);
    std::cout << "State size: " << human_readable(get_size(all_nodes)) << std::endl;

    if (size() > 3) {
        std::vector<Entry> free_lists;
        free_lists.emplace_back(m_free_list_positions.ref(), m_free_list_positions.size_in_bytes());
        free_lists.emplace_back(m_free_list_sizes.ref(), m_free_list_sizes.size_in_bytes());
        free_lists.emplace_back(m_free_list_versions.ref(), m_free_list_versions.size_in_bytes());
        consolidate_lists(all_nodes, free_lists);
    }

    if (size() > 8) {
        std::vector<Entry> history;
        history = get_nodes(m_alloc, get_ref(8));
        std::cout << "History size: " << human_readable(get_size(history)) << std::endl;
        consolidate_lists(all_nodes, history);
    }

    if (size() > 11) {
        std::vector<Entry> evac_info;
        evac_info = get_nodes(m_alloc, get_ref(11));
        consolidate_lists(all_nodes, evac_info);
    }

    return all_nodes;
}

std::vector<FreeListEntry> Group::get_free_list() const
{
    std::vector<FreeListEntry> list;
    if (valid()) {
        unsigned sz = m_free_list_positions.size();
        if (sz != m_free_list_sizes.size()) {
            std::cout << "FreeList positions size: " << sz << " FreeList sizes size: " << m_free_list_sizes.size()
                      << std::endl;
            return list;
        }
        if (sz != m_free_list_versions.size()) {
            std::cout << "FreeList positions size: " << sz
                      << " FreeList versions size: " << m_free_list_versions.size() << std::endl;
            return list;
        }
        for (unsigned i = 0; i < sz; i++) {
            int64_t pos = m_free_list_positions.get_val(i);
            int64_t size = m_free_list_sizes.get_val(i);
            int64_t version = m_free_list_versions.get_val(i);
            list.emplace_back(pos, size, version);
        }
    }
    return list;
}

RealmFile::RealmFile(const std::string& file_path, const char* encryption_key, uint64_t top_ref)
{
    realm::SlabAlloc::Config config;
    config.encryption_key = encryption_key;
    config.read_only = true;
    config.no_create = true;
    m_top_ref = m_alloc.attach_file(file_path, config);
    if (top_ref) {
        m_top_ref = top_ref;
        std::cout << "Using old top ref: 0x" << std::hex << m_top_ref << std::dec << std::endl;
    }
    else {
        std::cout << "Current top ref: 0x" << std::hex << m_top_ref << std::dec << std::endl;
    }
    m_start_pos = 24;
    m_group = std::make_unique<Group>(m_alloc, m_top_ref);
    m_file_format_version = m_alloc.get_committed_file_format_version();
    std::cout << "File format version: " << m_file_format_version << std::endl;
    std::cout << "File size: " << m_alloc.get_baseline() << std::endl;
    std::cout << *m_group;
}

void RealmFile::node_scan()
{
    std::map<uint64_t, unsigned> sizes;
    uint64_t ref = m_start_pos;
    auto free_list = m_group->get_free_list();
    auto free_entry = free_list.begin();
    auto end = m_alloc.get_baseline();
    if (m_alloc.is_file_on_streaming_form()) {
        end -= 16; // sizeof(StreamingFooter)
    }
    uint64_t bad_ref = 0;
    if (free_list.empty()) {
        std::cout << "*** No free list - results may be unreliable ***" << std::endl;
    }
    std::cout << std::hex;
    while (ref < end) {
        if (free_entry != free_list.end() && ref == free_entry->start) {
            ref += free_entry->length;
            ++free_entry;
        }
        else {
            while (free_entry != free_list.end() && ref > free_entry->start) {
                std::cout << "*** Bad free list entry: "
                          << "Start: 0x" << free_entry->start << "..0x" << free_entry->start + free_entry->length
                          << std::endl;
                ++free_entry;
            }
            Node n(m_alloc, ref);
            if (n.valid()) {
                if (bad_ref) {
                    std::cout << "*** Unaccounted space: "
                              << "Start: 0x" << bad_ref << "..0x" << ref << std::endl;
                    bad_ref = 0;
                }
                auto size_in_bytes = n.size_in_bytes();
                sizes[size_in_bytes]++;
                ref += size_in_bytes;
            }
            else {
                if (!bad_ref) {
                    bad_ref = ref;
                }
                ref += 8;
            }
        }
    }
    if (bad_ref) {
        std::cout << "*** Unaccounted space: "
                  << "Start: 0x" << bad_ref << "..0x" << end << std::endl;
    }
    std::cout << std::dec;
    std::cout << "Allocated space:" << std::endl;
    for (auto s : sizes) {
        std::cout << "    Size: " << s.first << " count: " << s.second << std::endl;
    }
}

void RealmFile::schema_info()
{
    m_group->print_schema();
}

void RealmFile::memory_leaks()
{
    if (m_group->valid()) {
        auto nodes = m_group->get_allocated_nodes();
        auto free_list = m_group->get_free_list();
        std::vector<Entry> free_blocks;
        for (auto& entry : free_list) {
            free_blocks.emplace_back(entry.start, entry.length);
        }
        consolidate_lists(nodes, free_blocks);
        auto it = nodes.begin();
        if (nodes.size() > 1) {
            std::cout << "Memory leaked:" << std::endl;
            auto prev = it;
            ++it;
            while (it != nodes.end()) {
                auto leak_start = prev->start + prev->length;
                std::cout << "    0x" << std::hex << leak_start << "..0x" << it->start << std::dec << std::endl;
                prev = it;
                ++it;
            }
        }
        else {
            REALM_ASSERT(it->length == m_group->get_file_size());
            std::cout << "No memory leaks" << std::endl;
        }
    }
}

void RealmFile::free_list_info() const
{
    std::map<uint64_t, unsigned> free_sizes;
    std::map<uint64_t, unsigned> pinned_sizes;
    std::cout << "Free space:" << std::endl;
    auto free_list = m_group->get_free_list();
    uint64_t pinned_free_list_size = 0;
    uint64_t total_free_list_size = 0;
    auto it = free_list.begin();
    auto end = free_list.end();
    while (it != end) {
        std::cout << "    0x" << std::hex << it->start << "..0x" << it->start + it->length << ", " << std::dec
                  << it->length << ", " << it->version << std::endl;
        total_free_list_size += it->length;
        if (it->version != 0) {
            pinned_free_list_size += it->length;
            pinned_sizes[it->length]++;
        }
        else {
            free_sizes[it->length]++;
        }

        ++it;
    }
    std::cout << "Free space sizes:" << std::endl;
    for (auto s : free_sizes) {
        std::cout << "    Size: " << s.first << " count: " << s.second << std::endl;
    }
    std::cout << "Pinned sizes:" << std::endl;
    for (auto s : pinned_sizes) {
        std::cout << "    Size: " << s.first << " count: " << s.second << std::endl;
    }
    std::cout << "Total free space size:  " << total_free_list_size << std::endl;
    std::cout << "Pinned free space size: " << pinned_free_list_size << std::endl;
}

class HistoryLogger {
public:
    HistoryLogger(Group* g)
        : m_group(g)
    {
    }
    bool select_table(realm::TableKey key)
    {
        std::cout << "Select table: " << m_group->get_table_name(key.value) << std::endl;
        m_table = m_group->get_table(key.value);
        return true;
    }

    bool insert_group_level_table(realm::TableKey key)
    {
        std::cout << "Create table: " << m_group->get_table_name(key.value) << std::endl;
        return true;
    }

    bool erase_class(realm::TableKey)
    {
        return true;
    }

    bool rename_class(realm::TableKey)
    {
        return true;
    }

    bool create_object(realm::ObjKey key)
    {
        std::cout << "Create object: " << key << std::endl;
        return true;
    }

    bool remove_object(realm::ObjKey key)
    {
        std::cout << "Remove object: " << key << std::endl;
        return true;
    }

    bool modify_object(realm::ColKey col_key, realm::ObjKey key)
    {
        std::cout << "Modify object: " << m_table->get_column_name(col_key) << " on " << key << std::endl;
        return true;
    }

    bool collection_set(size_t ndx)
    {
        std::cout << "Collection set at " << ndx << std::endl;
        return true;
    }

    bool collection_insert(size_t ndx)
    {
        std::cout << "Collection insert at " << ndx << std::endl;
        return true;
    }

    bool set_link_type(realm::ColKey)
    {
        return true;
    }

    bool insert_column(realm::ColKey col_key)
    {
        std::cout << "Add column: " << m_table->get_column_name(col_key) << std::endl;
        return true;
    }

    bool erase_column(realm::ColKey)
    {
        return true;
    }

    bool rename_column(realm::ColKey)
    {
        return true;
    }

    bool select_collection(realm::ColKey col_key, realm::ObjKey key)
    {
        std::cout << "Select collection: " << m_table->get_column_name(col_key) << " on " << key << std::endl;
        return true;
    }

    bool collection_move(size_t from_link_ndx, size_t to_link_ndx)
    {
        std::cout << "List move from " << from_link_ndx << " to " << to_link_ndx << std::endl;
        return true;
    }

    bool collection_erase(size_t ndx)
    {
        std::cout << "Collection erase at " << ndx << std::endl;
        return true;
    }

    bool collection_clear(size_t old_list_size)
    {
        std::cout << "Collection clear. Old size: " << old_list_size << std::endl;
        return true;
    }

    bool typed_link_change(realm::ColKey, realm::TableKey)
    {
        return true;
    }

private:
    Group* m_group;
    Table* m_table;
};

void RealmFile::changes() const
{
    realm::_impl::TransactLogParser parser;
    HistoryLogger logger(m_group.get());

    auto changesets = m_group->get_changesets();

    for (auto c : changesets) {
        realm::util::SimpleInputStream stream(c);
        parser.parse(stream, logger);
        std::cout << "--------------------------------------------" << std::endl;
    }
}

static unsigned int hex_char_to_bin(char c)
{
    if (c >= '0' && c <= '9')
        return c - '0';
    if (c >= 'a' && c <= 'f')
        return c - 'a' + 10;
    if (c >= 'A' && c <= 'F')
        return c - 'A' + 10;
    throw std::invalid_argument("Illegal key (not a hex digit)");
}

static unsigned int hex_to_bin(char first, char second)
{
    return (hex_char_to_bin(first) << 4) | hex_char_to_bin(second);
}


int main(int argc, const char* argv[])
{
    if (argc > 1) {
        try {
            bool free_list_info = false;
            bool memory_leaks = false;
            bool schema_info = false;
            bool node_scan = false;
            bool changes = false;
            uint64_t alternate_top = 0;
            const char* key_ptr = nullptr;
            char key[64];
            for (int curr_arg = 1; curr_arg < argc; curr_arg++) {
                if (strcmp(argv[curr_arg], "--keyfile") == 0) {
                    std::ifstream key_file(argv[curr_arg + 1]);
                    key_file.read(key, sizeof(key));
                    key_ptr = key;
                    curr_arg++;
                }
                else if (strcmp(argv[curr_arg], "--hexkey") == 0) {
                    curr_arg++;
                    const char* chars = argv[curr_arg];
                    if (strlen(chars) != 128) {
                        throw std::invalid_argument("Key string must be 128 chars long");
                    }
                    for (int idx = 0; idx < 64; ++idx) {
                        key[idx] = hex_to_bin(chars[idx * 2], chars[idx * 2 + 1]);
                    }
                    key_ptr = key;
                }
                else if (strcmp(argv[curr_arg], "--top") == 0) {
                    char* end;
                    curr_arg++;
                    alternate_top = strtol(argv[curr_arg], &end, 0);
                    if (*end != '\0' || (alternate_top & 7)) {
                        std::cout << "Not a ref: " << argv[curr_arg] << std::endl;
                        alternate_top = 0;
                    }
                }
                else if (argv[curr_arg][0] == '-') {
                    for (const char* command = argv[curr_arg] + 1; *command != '\0'; command++) {
                        switch (*command) {
                            case 'c':
                                changes = true;
                                break;
                            case 'f':
                                free_list_info = true;
                                break;
                            case 'm':
                                memory_leaks = true;
                                break;
                            case 's':
                                schema_info = true;
                                break;
                            case 'w':
                                node_scan = true;
                                break;
                        }
                    }
                }
                else {
                    std::cout << "File name: " << argv[curr_arg] << std::endl;
                    RealmFile rf(argv[curr_arg], key_ptr, alternate_top);
                    if (free_list_info) {
                        rf.free_list_info();
                    }
                    if (memory_leaks) {
                        rf.memory_leaks();
                    }
                    if (schema_info) {
                        rf.schema_info();
                    }
                    if (node_scan) {
                        rf.node_scan();
                    }
                    if (changes) {
                        rf.changes();
                    }
                    std::cout << std::endl;
                }
            }
        }
        catch (const std::exception& e) {
            std::cout << e.what() << std::endl;
        }
    }
    else {
        std::cout << "Usage: realm-trawler [-afmsw] [--keyfile file-with-binary-crypt-key] [--hexkey "
                     "crypt-key-in-hex] [--top "
                     "top_ref] <realmfile>"
                  << std::endl;
        std::cout << "   c : dump changelog" << std::endl;
        std::cout << "   f : free list analysis" << std::endl;
        std::cout << "   m : memory leak check" << std::endl;
        std::cout << "   s : schema dump" << std::endl;
        std::cout << "   w : node walk" << std::endl;
    }

    return 0;
}
