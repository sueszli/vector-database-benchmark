#pragma once

namespace steem { namespace protocol {

enum curve_id
{
   quadratic,
   bounded,
   linear,
   square_root,
   convergent_linear,
   convergent_square_root
};

} } // steem::utilities


FC_REFLECT_ENUM(
   steem::protocol::curve_id,
   (quadratic)
   (bounded)
   (linear)
   (square_root)
   (convergent_linear)
   (convergent_square_root)
)
