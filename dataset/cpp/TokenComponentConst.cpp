#include "TokenComponentConst.h"

std::shared_ptr<TokenComponent> TokenComponentConst::copy() const
{
	return std::make_shared<TokenComponentConst>(*this);
}
