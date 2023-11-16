#pragma once

#include <type_traits>

#include <elle/das/model.hh>

namespace elle
{
  namespace das
  {
    /// Flatten objects as tuple.
    ///
    /// \code{.cc}
    ///
    /// namespace symbols
    /// {
    ///   ELLE_DAS_SYMBOL(foo);
    ///   ELLE_DAS_SYMBOL(id);
    ///   ELLE_DAS_SYMBOL(name);
    /// }
    ///
    /// struct Foo
    /// {
    ///   std::string name;
    ///
    ///   using Model = elle::das::Model<
    ///                   Foo,
    ///                   decltype(elle::meta::list(symbols::name))>;
    /// };
    ///
    /// struct Device
    /// {
    ///   Device(int id, std::string const& name)
    ///     : id(id)
    ///     , foo(name)
    ///   {}
    ///
    ///   int id;
    ///   Foo foo;
    ///
    ///   using Model = elle::das::Model<
    ///                   Device,
    ///                   decltype(elle::meta::list(symbols::id,
    ///                                             symbols::foo))>;
    /// };
    ///
    /// Device d(42, "towel");
    /// auto flat = elle::das::flatten(d);
    /// auto rflat = elle::das::flatten_ref(d);
    /// assert(std::get<0>(flat) == d.id);
    /// assert(std::get<0>(std::get<1>(flat)) == d.foo.name);
    /// std::get<0>(rflat) += 5;
    /// assert(std::get<0>(rflat) == d.id);
    /// assert(std::get<0>(rflat) == std::get<0>(flat) + 5);
    ///
    /// \endcode
    namespace
    {
      template <typename O, template <typename> class M>
      struct flatten_object;

      template <typename Model, typename T, template <typename> class M>
      typename Model::Fields::template map<
        flatten_object<T, M>::template flatten>
        ::type::template apply<std::tuple>
      _flatten(typename M<T>::object_type o);

      template <typename T>
      struct FlattenByValue
      {
        using object_type = T const&;
        using type = T;
        template <typename V>
        static
        V const&
        value(V const& v)
        {
          return v;
        }
      };

      template <typename T>
      struct FlattenByRef
      {
        using object_type = T&;
        using type = typename std::remove_reference<T>::type&;
        static
        type
        value(T& v)
        {
          return v;
        }
      };

      template <typename T>
      struct FlattenByRefWrapper
      {
        using object_type = T&;
        using type =
          std::reference_wrapper<typename std::remove_reference<T>::type>;
         static
         type
         value(T& v)
         {
          return std::ref(v);
         }
       };

      template <typename T>
      struct FlattenByConstRef
      {
        using object_type = T const&;
        using type = T const&;
        static
        type
        value(T const& v)
        {
          return v;
        }
      };

      template <typename T>
      struct FlattenByConstRefWrapper
      {
        using object_type = T const&;
        using type =
          std::reference_wrapper<typename std::remove_reference<T>::type const>;
        static
        type
        value(T const& v)
        {
          return std::ref(v);
        }
      };

      template <typename T, template <typename> class M>
      struct FlattenRecurse
      {
        using Model = typename DefaultModel<T>::type;
        using type =
          typename Model::Fields::template map<
          flatten_object<T, M>::template flatten>
          ::type::template apply<std::tuple>;
        static
        type
        value(typename M<T>::object_type o)
        {
          return _flatten<Model, T, M>(o);
        }
      };

      template <typename T, template <typename> class M>
      struct FlattenCompose
        : public std::conditional<model_has<T>(),
                                  FlattenRecurse<T, M>,
                                  M<T> >::type
      {};

      template <typename Model, typename T, template <typename> class M>
      typename Model::Fields::template map<
        flatten_object<T, M>::template flatten>
        ::type::template apply<std::tuple>
      _flatten(typename M<T>::object_type o)
      {
        return Model::Fields::template map<
          flatten_object<T, M>::template flatten>::value(o);
      }

      template <typename O, template <typename> class M>
      struct flatten_object
      {
        template <typename S>
        struct flatten
        {
          using Method = FlattenCompose<typename S::template attr_type<O>, M>;
          using type = typename Method::type;
          static
          type
          value(typename M<O>::object_type o)
          {
            return Method::value(S::attr_get(o));
          }
        };
      };
    }

    /// Flatten a structure: return it as a tuple of values.
    template <typename Model, typename T>
    typename Model::Fields::template map<
      flatten_object<T, FlattenByValue>::template flatten>
      ::type::template apply<std::tuple>
    flatten(T const& o)
    {
      return _flatten<Model, T, FlattenByValue>(o);
    }

    /// Flatten a structure: return it as a tuple of values.
    template <typename T>
    auto
    flatten(T const& o)
    {
      return flatten<typename DefaultModel<T>::type, T>(o);
    }

    // Flatten by ref
    template <typename Model, typename T>
    typename Model::Fields::template map<
      flatten_object<T, FlattenByRef>::template flatten>
      ::type::template apply<std::tuple>
    flatten_ref(T& o)
    {
      return _flatten<Model, T, FlattenByRef>(o);
    }

    template <typename T>
    auto
    flatten_ref(T& o)
    {
      return flatten_ref<typename DefaultModel<T>::type, T>(o);
    }

    // Flatten by const ref
    template <typename Model, typename T>
    typename Model::Fields::template map<
      flatten_object<T, FlattenByConstRef>::template flatten>
      ::type::template apply<std::tuple>
    flatten_ref(T const& o)
    {
      return _flatten<Model, T, FlattenByConstRef>(o);
    }

    template <typename T>
    auto
    flatten_ref(T const& o)
    {
      return flatten_ref<typename DefaultModel<T>::type, T>(o);
    }
  }
}
