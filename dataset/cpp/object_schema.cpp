////////////////////////////////////////////////////////////////////////////
//
// Copyright 2015 Realm Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
////////////////////////////////////////////////////////////////////////////

#include <realm/object-store/object_schema.hpp>

#include <realm/object-store/feature_checks.hpp>
#include <realm/object-store/object_store.hpp>
#include <realm/object-store/property.hpp>
#include <realm/object-store/schema.hpp>

#include <realm/data_type.hpp>
#include <realm/group.hpp>
#include <realm/table.hpp>

using namespace realm;

static_assert(uint8_t(ObjectSchema::ObjectType::TopLevel) == uint8_t(Table::Type::TopLevel) &&
                  uint8_t(ObjectSchema::ObjectType::Embedded) == uint8_t(Table::Type::Embedded) &&
                  uint8_t(ObjectSchema::ObjectType::TopLevelAsymmetric) == uint8_t(Table::Type::TopLevelAsymmetric),
              "Values of 'ObjectSchema::ObjectType' and 'Table::Type' enums don't match");

ObjectSchema::ObjectSchema() = default;
ObjectSchema::~ObjectSchema() = default;

ObjectSchema::ObjectSchema(std::string name, std::initializer_list<Property> persisted_properties)
    : ObjectSchema(std::move(name), persisted_properties, {})
{
}

ObjectSchema::ObjectSchema(std::string name, ObjectType table_type,
                           std::initializer_list<Property> persisted_properties)
    : ObjectSchema(std::move(name), table_type, persisted_properties, {})
{
}

ObjectSchema::ObjectSchema(std::string name, std::initializer_list<Property> persisted_properties,
                           std::initializer_list<Property> computed_properties, std::string name_alias)
    : ObjectSchema(std::move(name), ObjectType::TopLevel, persisted_properties, computed_properties, name_alias)
{
}

ObjectSchema::ObjectSchema(std::string name, ObjectType table_type,
                           std::initializer_list<Property> persisted_properties,
                           std::initializer_list<Property> computed_properties, std::string name_alias)
    : name(std::move(name))
    , persisted_properties(persisted_properties)
    , computed_properties(computed_properties)
    , table_type(table_type)
    , alias(name_alias)
{
    for (auto const& prop : persisted_properties) {
        if (prop.is_primary) {
            primary_key = prop.name;
            break;
        }
    }
}

PropertyType ObjectSchema::from_core_type(ColumnType type)
{
    switch (type) {
        case col_type_Int:
            return PropertyType::Int;
        case col_type_Float:
            return PropertyType::Float;
        case col_type_Double:
            return PropertyType::Double;
        case col_type_Bool:
            return PropertyType::Bool;
        case col_type_String:
            return PropertyType::String;
        case col_type_Binary:
            return PropertyType::Data;
        case col_type_Timestamp:
            return PropertyType::Date;
        case col_type_Mixed:
            return PropertyType::Mixed;
        case col_type_ObjectId:
            return PropertyType::ObjectId;
        case col_type_Decimal:
            return PropertyType::Decimal;
        case col_type_UUID:
            return PropertyType::UUID;
        case col_type_Link:
        case col_type_TypedLink:
            return PropertyType::Object;
        case col_type_LinkList:
            return PropertyType::Object | PropertyType::Array;
        default:
            REALM_UNREACHABLE();
    }
    return PropertyType::Int;
}

PropertyType ObjectSchema::from_core_type(ColKey col)
{
    PropertyType prop_type = from_core_type(col.get_type());

    auto attr = col.get_attrs();
    if (attr.test(col_attr_Nullable))
        prop_type |= PropertyType::Nullable;
    if (attr.test(col_attr_List))
        prop_type |= PropertyType::Array;
    else if (attr.test(col_attr_Set))
        prop_type |= PropertyType::Set;
    else if (attr.test(col_attr_Dictionary))
        prop_type |= PropertyType::Dictionary;

    return prop_type;
}

ObjectSchema::ObjectSchema(Group const& group, StringData name, TableKey key)
    : name(name)
{
    ConstTableRef table;
    if (key) {
        table = group.get_table(key);
    }
    else {
        table = ObjectStore::table_for_object_type(group, name);
    }
    table_key = table->get_key();
    table_type = static_cast<ObjectSchema::ObjectType>(table->get_table_type());

    size_t count = table->get_column_count();
    ColKey pk_col = table->get_primary_key_column();
    persisted_properties.reserve(count);

    for (auto col_key : table->get_column_keys()) {
        StringData column_name = table->get_column_name(col_key);

        Property property;
        property.name = column_name;
        property.type = ObjectSchema::from_core_type(col_key);
        property.is_indexed = table->search_index_type(col_key) == IndexType::General;
        property.is_fulltext_indexed = table->search_index_type(col_key) == IndexType::Fulltext;
        property.column_key = col_key;

        if (property.type == PropertyType::Object) {
            // set link type for objects and arrays
            ConstTableRef linkTable = table->get_link_target(col_key);
            property.object_type = ObjectStore::object_type_for_table_name(linkTable->get_name().data());
        }
        persisted_properties.push_back(std::move(property));
    }

    if (pk_col)
        primary_key = table->get_column_name(pk_col);
    set_primary_key_property();
}

Property* ObjectSchema::property_for_name(StringData name) noexcept
{
    for (auto& prop : persisted_properties) {
        if (StringData(prop.name) == name) {
            return &prop;
        }
    }
    for (auto& prop : computed_properties) {
        if (StringData(prop.name) == name) {
            return &prop;
        }
    }
    return nullptr;
}

Property* ObjectSchema::property_for_public_name(StringData public_name) noexcept
{
    // If no `public_name` is defined, the internal `name` is also considered the public name.
    for (auto& prop : persisted_properties) {
        if (prop.public_name == public_name || (prop.public_name.empty() && prop.name == public_name))
            return &prop;
    }

    // Computed properties are not persisted, so creating a public name for such properties
    // are a bit pointless since the internal name is already the "public name", but since
    // this distinction isn't visible in the Property struct we allow it anyway.
    for (auto& prop : computed_properties) {
        if (StringData(prop.public_name.empty() ? prop.name : prop.public_name) == public_name)
            return &prop;
    }
    return nullptr;
}

const Property* ObjectSchema::property_for_public_name(StringData public_name) const noexcept
{
    return const_cast<ObjectSchema*>(this)->property_for_public_name(public_name);
}

const Property* ObjectSchema::property_for_name(StringData name) const noexcept
{
    return const_cast<ObjectSchema*>(this)->property_for_name(name);
}

bool ObjectSchema::property_is_computed(Property const& property) const noexcept
{
    auto end = computed_properties.end();
    return std::find(computed_properties.begin(), end, property) != end;
}

void ObjectSchema::set_primary_key_property() noexcept
{
    if (primary_key.length()) {
        if (auto primary_key_prop = primary_key_property()) {
            primary_key_prop->is_primary = true;
        }
    }
}

static void validate_property(Schema const& schema, ObjectSchema const& parent_object_schema, Property const& prop,
                              Property const** primary, std::vector<ObjectSchemaValidationException>& exceptions)
{
    auto& object_name = parent_object_schema.name;

    if (prop.type == PropertyType::LinkingObjects && !is_array(prop.type)) {
        exceptions.emplace_back("Linking Objects property '%1.%2' must be an array.", object_name, prop.name);
    }

    // check nullablity
    if (is_nullable(prop.type) && !prop.type_is_nullable()) {
        exceptions.emplace_back("Property '%1.%2' of type '%3' cannot be nullable.", object_name, prop.name,
                                string_for_property_type(prop.type));
    }
    else if (prop.type == PropertyType::Object && !is_nullable(prop.type) &&
             !(is_array(prop.type) || is_set(prop.type))) {
        exceptions.emplace_back("Property '%1.%2' of type 'object' must be nullable.", object_name, prop.name);
    }
    else if (prop.type == PropertyType::Mixed && !is_nullable(prop.type)) {
        exceptions.emplace_back("Property '%1.%2' of type 'Mixed' must be nullable.", object_name, prop.name);
    }

    // check primary keys
    if (prop.is_primary) {
        if (prop.type != PropertyType::Int && prop.type != PropertyType::String &&
            prop.type != PropertyType::ObjectId && prop.type != PropertyType::UUID) {
            exceptions.emplace_back("Property '%1.%2' of type '%3' cannot be made the primary key.", object_name,
                                    prop.name, string_for_property_type(prop.type));
        }
        if (*primary) {
            exceptions.emplace_back("Properties '%1' and '%2' are both marked as the primary key of '%3'.", prop.name,
                                    (*primary)->name, object_name);
        }
        *primary = &prop;
    }

    // check indexable
    if (prop.is_indexed && !prop.type_is_indexable()) {
        exceptions.emplace_back("Property '%1.%2' of type '%3' cannot be indexed.", object_name, prop.name,
                                string_for_property_type(prop.type));
    }

    // check that only link properties have object types
    if (prop.type != PropertyType::LinkingObjects && !prop.link_origin_property_name.empty()) {
        exceptions.emplace_back("Property '%1.%2' of type '%3' cannot have an origin property name.", object_name,
                                prop.name, string_for_property_type(prop.type));
    }
    else if (prop.type == PropertyType::LinkingObjects && prop.link_origin_property_name.empty()) {
        exceptions.emplace_back("Property '%1.%2' of type '%3' must have an origin property name.", object_name,
                                prop.name, string_for_property_type(prop.type));
    }

    if (prop.type != PropertyType::Object && prop.type != PropertyType::LinkingObjects) {
        if (!prop.object_type.empty()) {
            exceptions.emplace_back("Property '%1.%2' of type '%3' cannot have an object type.", object_name,
                                    prop.name, prop.type_string());
        }
        return;
    }


    // check that the object_type is valid for link properties
    auto it = schema.find(prop.object_type);
    if (it == schema.end()) {
        exceptions.emplace_back("Property '%1.%2' of type '%3' has unknown object type '%4'", object_name, prop.name,
                                string_for_property_type(prop.type), prop.object_type);
        return;
    }
    if (is_set(prop.type) && it->table_type == ObjectSchema::ObjectType::Embedded) {
        exceptions.emplace_back("Set property '%1.%2' cannot contain embedded object type '%3'. Set semantics are "
                                "not applicable to embedded objects.",
                                object_name, prop.name, prop.object_type);
        return;
    }
    if (it->table_type == ObjectSchema::ObjectType::TopLevelAsymmetric) {
        exceptions.emplace_back("Property '%1.%2' of type '%3' cannot be a link to an asymmetric object.",
                                object_name, prop.name, string_for_property_type(prop.type));
        return;
    }
    if (prop.type != PropertyType::LinkingObjects)
        return;

    const Property* origin_property = it->property_for_name(prop.link_origin_property_name);
    if (!origin_property) {
        exceptions.emplace_back(
            "Property '%1.%2' declared as origin of linking objects property '%3.%4' does not exist",
            prop.object_type, prop.link_origin_property_name, object_name, prop.name);
    }
    else if (origin_property->type != PropertyType::Object) {
        exceptions.emplace_back(
            "Property '%1.%2' declared as origin of linking objects property '%3.%4' is not a link", prop.object_type,
            prop.link_origin_property_name, object_name, prop.name);
    }
    else if (origin_property->object_type != object_name) {
        exceptions.emplace_back(
            "Property '%1.%2' declared as origin of linking objects property '%3.%4' links to type '%5'",
            prop.object_type, prop.link_origin_property_name, object_name, prop.name, origin_property->object_type);
    }
}

void ObjectSchema::validate(Schema const& schema, std::vector<ObjectSchemaValidationException>& exceptions,
                            SchemaValidationMode validation_mode) const
{
    std::vector<StringData> public_property_names;
    std::vector<StringData> internal_property_names;
    internal_property_names.reserve(persisted_properties.size() + computed_properties.size());
    auto gather_names = [&](auto const& properties) {
        for (auto const& prop : properties) {
            internal_property_names.push_back(prop.name);
            if (!prop.public_name.empty())
                public_property_names.push_back(prop.public_name);
        }
    };
    gather_names(persisted_properties);
    gather_names(computed_properties);
    std::sort(public_property_names.begin(), public_property_names.end());
    std::sort(internal_property_names.begin(), internal_property_names.end());

    // Check that property names and aliases are unique
    auto for_each_duplicate = [](auto&& container, auto&& fn) {
        auto end = container.end();
        for (auto it = std::adjacent_find(container.begin(), end); it != end; it = std::adjacent_find(it + 2, end))
            fn(*it);
    };
    for_each_duplicate(public_property_names, [&](auto public_property_name) {
        exceptions.emplace_back("Alias '%1' appears more than once in the schema for type '%2'.",
                                public_property_name, name);
    });
    for_each_duplicate(internal_property_names, [&](auto internal_name) {
        exceptions.emplace_back("Property '%1' appears more than once in the schema for type '%2'.", internal_name,
                                name);
    });

    // Check that no aliases conflict with property names
    struct ErrorWriter {
        ObjectSchema const& os;
        std::vector<ObjectSchemaValidationException>& exceptions;
        ErrorWriter(ObjectSchema const& os, std::vector<ObjectSchemaValidationException>& exceptions)
            : os(os)
            , exceptions(exceptions)
        {
        }

        ErrorWriter& operator=(StringData name)
        {
            exceptions.emplace_back(
                "Property '%1.%2' has an alias '%3' that conflicts with a property of the same name.", os.name,
                os.property_for_public_name(name)->name, name);
            return *this;
        }

        ErrorWriter(ErrorWriter const&) = default;
        ErrorWriter& operator=(ErrorWriter const&)
        {
            return *this;
        }
        ErrorWriter& operator*()
        {
            return *this;
        }
        ErrorWriter& operator++()
        {
            return *this;
        }
        ErrorWriter& operator++(int)
        {
            return *this;
        }
    } writer{*this, exceptions};
    std::set_intersection(public_property_names.begin(), public_property_names.end(), internal_property_names.begin(),
                          internal_property_names.end(), writer);

    // Validate all properties
    const Property* primary = nullptr;
    for (auto const& prop : persisted_properties) {
        validate_property(schema, *this, prop, &primary, exceptions);
    }
    for (auto const& prop : computed_properties) {
        validate_property(schema, *this, prop, &primary, exceptions);
    }

    if (!primary_key.empty() && table_type == ObjectSchema::ObjectType::Embedded) {
        exceptions.emplace_back("Embedded object type '%1' cannot have a primary key.", name);
    }
    if (!primary_key.empty() && !primary && !primary_key_property()) {
        exceptions.emplace_back("Specified primary key '%1.%2' does not exist.", name, primary_key);
    }

    auto for_sync =
        (validation_mode & SchemaValidationMode::SyncPBS) || (validation_mode & SchemaValidationMode::SyncFLX);
    if (for_sync && table_type != ObjectSchema::ObjectType::Embedded) {
        if (primary_key.empty()) {
            exceptions.emplace_back(util::format("There must be a primary key property named '_id' on a synchronized "
                                                 "Realm but none was found for type '%1'",
                                                 name));
        }
        else if (primary_key != "_id") {
            exceptions.emplace_back(util::format(
                "The primary key property on a synchronized Realm must be named '_id' but found '%1' for type '%2'",
                primary_key, name));
        }
    }

    if (!for_sync && table_type == ObjectSchema::ObjectType::TopLevelAsymmetric) {
        exceptions.emplace_back(util::format("Asymmetric table '%1' not allowed in a local Realm", name));
    }

    auto pbs_sync =
        (validation_mode & SchemaValidationMode::SyncPBS) && !(validation_mode & SchemaValidationMode::SyncFLX);
    if (pbs_sync && table_type == ObjectSchema::ObjectType::TopLevelAsymmetric) {
        exceptions.emplace_back(util::format("Asymmetric table '%1' not allowed in partition based sync", name));
    }
}

namespace realm {
bool operator==(ObjectSchema const& a, ObjectSchema const& b) noexcept
{
    return std::tie(a.name, a.table_type, a.primary_key, a.persisted_properties, a.computed_properties) ==
           std::tie(b.name, b.table_type, b.primary_key, b.persisted_properties, b.computed_properties);
}

std::ostream& operator<<(std::ostream& o, ObjectSchema::ObjectType table_type)
{
    switch (table_type) {
        case ObjectSchema::ObjectType::TopLevel:
            return o << "TopLevel";
        case ObjectSchema::ObjectType::Embedded:
            return o << "Embedded";
        case ObjectSchema::ObjectType::TopLevelAsymmetric:
            return o << "TopLevelAsymmetric";
    }
    return o << "Invalid table type: " << uint8_t(table_type);
}
} // namespace realm
