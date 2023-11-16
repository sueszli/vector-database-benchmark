/**
 * @file TupleType.hh
 * @author Giovanni Tedesco (giovanni@distributive.network)
 * @brief Struct for representing python tuples
 * @version 0.1
 * @date 2022-08-19
 *
 * @copyright Copyright (c) 2022
 *
 */

#ifndef PythonMonkey_TupleType_
#define PythonMonkey_TupleType_

#include "include/PyType.hh"
#include "include/TypeEnum.hh"

#include <Python.h>

/**
 * @brief A struct to represent the tuple type in python
 *
 */
struct TupleType : public PyType {

public:
  TupleType(PyObject *obj);
  const TYPE returnType = TYPE::TUPLE;

/**
 * @brief Gets the tuple item at the given index
 *
 * @param index The index of the item in question
 * @return PyType* Returns a pointer to the appropriate PyType object
 */
  PyType *get(int index) const;

/**
 * @brief
 *
 * @returns int length of the tuple
 */
  int len() const;
};
#endif