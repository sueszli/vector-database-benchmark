#ifndef METAL_VALUE_SAME_HPP
#define METAL_VALUE_SAME_HPP

#include "../config.hpp"

namespace metal {
/// \cond
namespace detail {
    template <class... vals>
    struct _same;
}
/// \endcond

/// \ingroup value
///
/// ### Description
/// Checks whether all \values are identical.
///
/// ### Usage
/// For any \values `val_0, ..., val_n-1`
/// \code
///     using result = metal::same<val_0, ..., val_n-1>;
/// \endcode
///
/// \returns: \number
/// \semantics:
///     If at least two \values in `val_0, ..., val_n-1` are not identical
///     to each other, then
///     \code
///         using result = metal::false_;
///     \endcode
///     otherwise
///     \code
///         using result = metal::true_;
///     \endcode
///
/// ### Example
/// \snippet value.cpp same
///
/// ### See Also
/// \see distinct
template <class... vals>
using same = typename detail::_same<vals...>::type;
}

#include "../number/number.hpp"
#include "../value/value.hpp"

namespace metal {
/// \cond
namespace detail {
    template <class...>
    struct _same_impl : false_ {
    };

    template <template <class> class... _, class val>
    struct _same_impl<_<val>...> : true_ {
    };

    template <class... vals>
    struct _same : _same_impl<maybe<vals>...> {
    };

    template <class x, class y>
    struct _same<x, y> : false_ {
    };

    template <class x>
    struct _same<x, x> : true_ {
    };

    template <class x>
    struct _same<x> : true_ {
    };

    template <>
    struct _same<> : true_ {
    };
}
/// \endcond
}

#endif
