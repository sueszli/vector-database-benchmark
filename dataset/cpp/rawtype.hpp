/****************************************************************************
**
** This file is part of the Ponder library, formerly CAMP.
**
** The MIT License (MIT)
**
** Copyright (C) 2009-2014 TEGESO/TEGESOFT and/or its subsidiary(-ies) and mother company.
** Copyright (C) 2015-2020 Nick Trout.
**
** Permission is hereby granted, free of charge, to any person obtaining a copy
** of this software and associated documentation files (the "Software"), to deal
** in the Software without restriction, including without limitation the rights
** to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
** copies of the Software, and to permit persons to whom the Software is
** furnished to do so, subject to the following conditions:
** 
** The above copyright notice and this permission notice shall be included in
** all copies or substantial portions of the Software.
** 
** THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
** IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
** FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
** AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
** LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
** OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
** THE SOFTWARE.
**
****************************************************************************/

#pragma once
#ifndef PONDER_DETAIL_RAWTYPE_HPP
#define PONDER_DETAIL_RAWTYPE_HPP

#include <memory>

namespace ponder {
namespace detail {
    
/**
 * \brief Utility class which tells at compile-time if a type T is a smart pointer to a type U
 *
 * To detect a smart pointer type, we check using SFINAE if T implements an operator -> returning a U*
 */
template <typename T, typename U>
struct IsSmartPointer
{
    static constexpr bool value = false;
};

template <typename T, typename U>
struct IsSmartPointer<std::unique_ptr<T>, U>
{
    static constexpr bool value = true;
};

template <typename T, typename U>
struct IsSmartPointer<std::shared_ptr<T>, U>
{
    static constexpr bool value = true;
};
        
/**
 * \class DataType
 *
 * \brief Helper structure used to extract the raw type of a composed type
 *
 * DataType<T> recursively removes const, reference and pointer modifiers from the given type.
 * In other words:
 *
 * \li DataType<T>::Type == T
 * \li DataType<const T>::Type == DataType<T>::Type
 * \li DataType<T&>::Type == DataType<T>::Type
 * \li DataType<const T&>::Type == DataType<T>::Type
 * \li DataType<T*>::Type == DataType<T>::Type
 * \li DataType<const T*>::Type == DataType<T>::Type
 *
 * \remark DataType is able to detect smart pointers and properly extract the stored type
 */

// Generic version -- T doesn't match with any of our specialization, and is thus considered a raw data type
//  - int -> int, int[] -> int, int* -> int.
 template <typename T, typename E = void>
struct DataType
{
    typedef T Type;
};

// const
template <typename T>
struct DataType<const T> : public DataType<T> {};

template <typename T>
struct DataType<T&> : public DataType<T> {};

template <typename T>
struct DataType<T*> : public DataType<T> {};
    
template <typename T, size_t N>
struct DataType<T[N]> : public DataType<T> {};

// smart pointer
template <template <typename> class T, typename U>
struct DataType<T<U>, typename std::enable_if<IsSmartPointer<T<U>, U>::value >::type>
{
    typedef typename DataType<U>::Type Type;
};

} // namespace detail

    
template<class T>
T* get_pointer(T *p)
{
    return p;
}

template<class T>
T* get_pointer(std::unique_ptr<T> const& p)
{
    return p.get();
}

template<class T>
T* get_pointer(std::shared_ptr<T> const& p)
{
    return p.get();
}

} // namespace ponder

#endif // PONDER_DETAIL_RAWTYPE_HPP
