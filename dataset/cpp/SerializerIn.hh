#pragma once

#include <elle/json/json.hh>
#include <elle/serialization/SerializerIn.hh>

namespace elle
{
  namespace serialization
  {
    namespace json
    {
      /// A specialized SerializerIn for JSON.
      ///
      /// Deserialize objects from their JSON representations.
      class ELLE_API SerializerIn
        : public serialization::SerializerIn
      {
      /*------.
      | Types |
      `------*/
      public:
        using Self = SerializerIn;
        using Super = serialization::SerializerIn;

      /*-------------.
      | Construction |
      `-------------*/
      public:
        /// Construct a SerializerIn for JSON.
        ///
        /// @see elle::serialization::SerializerIn.
        SerializerIn(std::istream& input, bool versioned = true);
        /// Construct a SerializerIn for JSON.
        ///
        /// @see elle::serialization::SerializerIn.
        SerializerIn(std::istream& input,
                     Versions versions, bool versioned = true);
        /// Construct a SerializerIn from a JSON object.
        ///
        /// @param input A json object.
        /// @param versioned Whether the Serializer will read the version of
        ///                  objects.
        SerializerIn(elle::json::Json input, bool versioned = true);

      /*--------------.
      | Configuration |
      `--------------*/
      public:
        ELLE_ATTRIBUTE_RW(bool, partial);

      /*--------------.
      | Serialization |
      `--------------*/
      protected:
        void
        _serialize(int64_t& v) override;
        void
        _serialize(uint64_t& v) override;
        void
        _serialize(ulong& v) override;
        void
        _serialize(int32_t& v) override;
        void
        _serialize(uint32_t& v) override;
        void
        _serialize(int16_t& v) override;
        void
        _serialize(uint16_t& v) override;
        void
        _serialize(int8_t& v) override;
        void
        _serialize(uint8_t& v) override;
        void
        _serialize(double& v) override;
        void
        _serialize(bool& v) override;
        void
        _serialize(std::string& v) override;
        void
        _serialize(elle::Buffer& v) override;
        void
        _serialize(boost::posix_time::ptime& v) override;
        void
        _serialize_time_duration(std::int64_t& ticks,
                                 std::int64_t& num,
                                 std::int64_t& denom) override;
        void
        _serialize_named_option(std::string const& name,
                                bool,
                                std::function<void ()> const& f) override;
        void
        _serialize_option(bool,
                          std::function<void ()> const& f) override;
        void
        _serialize_array(int size,
                         std::function<void ()> const& f) override;
        void
        _deserialize_dict_key(
          std::function<void (std::string const&)> const& f) override;
        bool
        _enter(std::string const& name) override;
        void
        _leave(std::string const& name) override;

        ELLE_ATTRIBUTE(elle::json::Json, json);
        ELLE_ATTRIBUTE(std::vector<elle::json::Json*>, current);
      private:
        elle::json::Json&
        _check_type(elle::json::Type t);
        template <typename T>
        void
        _serialize_int(T& v);
      };
    }
  }
}
