#include "pocketpy/tuplelist.h"

namespace pkpy {

Tuple::Tuple(int n){
    if(n <= 3){
        this->_args = _inlined;
    }else{
        this->_args = (PyObject**)pool64_alloc(n * sizeof(void*));
    }
    this->_size = n;
}

Tuple::Tuple(const Tuple& other): Tuple(other._size){
    for(int i=0; i<_size; i++) _args[i] = other._args[i];
}

Tuple::Tuple(Tuple&& other) noexcept {
    _size = other._size;
    if(other.is_inlined()){
        _args = _inlined;
        for(int i=0; i<_size; i++) _args[i] = other._args[i];
    }else{
        _args = other._args;
        other._args = other._inlined;
        other._size = 0;
    }
}

Tuple::Tuple(List&& other) noexcept {
    _size = other.size();
    _args = other._data;
    other._data = nullptr;
}

Tuple::Tuple(std::initializer_list<PyObject*> list): Tuple(list.size()){
    int i = 0;
    for(PyObject* obj: list) _args[i++] = obj;
}

Tuple::~Tuple(){ if(!is_inlined()) pool64_dealloc(_args); }

List ArgsView::to_list() const{
    List ret(size());
    for(int i=0; i<size(); i++) ret[i] = _begin[i];
    return ret;
}

Tuple ArgsView::to_tuple() const{
    Tuple ret(size());
    for(int i=0; i<size(); i++) ret[i] = _begin[i];
    return ret;
}

}   // namespace pkpy