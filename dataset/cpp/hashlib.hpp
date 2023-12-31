#ifndef __HASHLIB_HPP
#define __HASHLIB_HPP

#include "builtin.hpp"

using namespace __shedskin__;
namespace __hashlib__ {

class md5;


extern str *__name__;

extern class_ *cl_md5;
class md5 : public pyobj {
public:
    bytes *data;

    md5() {}
    md5(bytes *data) {
        this->__class__ = cl_md5;
        __init__(data);
    }
    str *hexdigest();
    void *__init__(bytes *data);
};

void __init();

} // module namespace
#endif
