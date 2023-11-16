// Copyright (c) 2015-2016 Vittorio Romeo
// License: Academic Free License ("AFL") v. 3.0
// AFL License page: http://opensource.org/licenses/AFL-3.0
// http://vittorioromeo.info | vittorio.romeo@outlook.com

#pragma once

#include <ecst/config.hpp>
#include <ecst/mp/core.hpp>
#include <boost/hana/ext/std/tuple.hpp>

ECST_MP_LIST_NAMESPACE
{
    namespace impl
    {
        template <typename T>
        struct list_unwrapper;

        template <typename... Ts>
        struct list_unwrapper<type_list<Ts...>>
        {
            using type = std::tuple<typename Ts::type...>;
        };

        template <typename T>
        struct list_bh_unwrapper;

        template <typename... Ts>
        struct list_bh_unwrapper<type_list<Ts...>>
        {
            using type = bh::tuple<typename Ts::type...>;
        };
    }

    /// @brief Unwraps a `type_list<type_c<xs>...>` into an `std::tuple<xs...>`.
    template <typename T>
    using unwrap_tuple = typename impl::list_unwrapper<T>::type;

    /// @brief Unwraps a `type_list<type_c<xs>...>` into a `bh::tuple<xs...>`.
    template <typename T>
    using unwrap_bh_tuple = typename impl::list_bh_unwrapper<T>::type;
}
ECST_MP_LIST_NAMESPACE_END
