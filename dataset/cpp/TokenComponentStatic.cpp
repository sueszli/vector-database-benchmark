#include "TokenComponentStatic.h"

std::shared_ptr<TokenComponent> TokenComponentStatic::copy() const
{
	return std::make_shared<TokenComponentStatic>(*this);
}
