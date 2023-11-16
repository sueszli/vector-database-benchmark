#include "include/StrType.hh"

#include "include/PyType.hh"

#include <Python.h>

#include <jsapi.h>
#include <js/String.h>

#define PY_UNICODE_HAS_WSTR (PY_VERSION_HEX < 0x030c0000) // Python version is less than 3.12

#define HIGH_SURROGATE_START 0xD800
#define HIGH_SURROGATE_END 0xDBFF
#define LOW_SURROGATE_START 0xDC00
#define LOW_SURROGATE_END 0xDFFF

#define PY_ASCII_OBJECT_CAST(op) ((PyASCIIObject *)(op))
#define PY_COMPACT_UNICODE_OBJECT_CAST(op) ((PyCompactUnicodeObject *)(op))
#define PY_UNICODE_OBJECT_CAST(op) ((PyUnicodeObject *)(op))

// https://github.com/python/cpython/blob/8de607a/Objects/unicodeobject.c#L114-L154
#define PY_UNICODE_OBJECT_UTF8(op)        (PY_COMPACT_UNICODE_OBJECT_CAST(op)->utf8)
#define PY_UNICODE_OBJECT_UTF8_LENGTH(op) (PY_COMPACT_UNICODE_OBJECT_CAST(op)->utf8_length)
#define PY_UNICODE_OBJECT_DATA_ANY(op)    (PY_UNICODE_OBJECT_CAST(op)->data.any)
#define PY_UNICODE_OBJECT_DATA_UCS2(op)   (PY_UNICODE_OBJECT_CAST(op)->data.ucs2)
#define PY_UNICODE_OBJECT_HASH(op)        (PY_ASCII_OBJECT_CAST(op)->hash)
#define PY_UNICODE_OBJECT_STATE(op)       (PY_ASCII_OBJECT_CAST(op)->state)
#define PY_UNICODE_OBJECT_KIND(op)        (PY_ASCII_OBJECT_CAST(op)->state.kind)
#define PY_UNICODE_OBJECT_LENGTH(op)      (PY_ASCII_OBJECT_CAST(op)->length)
#if PY_UNICODE_HAS_WSTR
  #define PY_UNICODE_OBJECT_WSTR(op)        (PY_ASCII_OBJECT_CAST(op)->wstr)
  #define PY_UNICODE_OBJECT_WSTR_LENGTH(op) (PY_COMPACT_UNICODE_OBJECT_CAST(op)->wstr_length)
  #define PY_UNICODE_OBJECT_READY(op)       (PY_ASCII_OBJECT_CAST(op)->state.ready)
#endif

/**
 * @brief check if UTF-16 encoded `chars` contain a surrogate pair
 */
static bool containsSurrogatePair(const char16_t *chars, size_t length) {
  for (size_t i = 0; i < length; i++) {
    if (Py_UNICODE_IS_SURROGATE(chars[i])) {
      return true;
    }
  }
  return false;
}

StrType::StrType(PyObject *object) : PyType(object) {}

StrType::StrType(char *string) : PyType(Py_BuildValue("s", string)) {}

StrType::StrType(JSContext *cx, JSString *str) {
  JSLinearString *lstr = JS_EnsureLinearString(cx, str);
  JS::AutoCheckCannotGC nogc;
  PyObject *p;

  size_t length = JS::GetLinearStringLength(lstr);

  pyObject = (PyObject *)PyObject_New(PyUnicodeObject, &PyUnicode_Type); // new reference
  Py_INCREF(pyObject); // XXX: Why?

  // Initialize as legacy string (https://github.com/python/cpython/blob/v3.12.0b1/Include/cpython/unicodeobject.h#L78-L93)
  // see https://github.com/python/cpython/blob/v3.11.3/Objects/unicodeobject.c#L1230-L1245
  PY_UNICODE_OBJECT_HASH(pyObject) = -1;
  PY_UNICODE_OBJECT_STATE(pyObject).interned = 0;
  PY_UNICODE_OBJECT_STATE(pyObject).compact = 0;
  PY_UNICODE_OBJECT_STATE(pyObject).ascii = 0;
  PY_UNICODE_OBJECT_UTF8(pyObject) = NULL;
  PY_UNICODE_OBJECT_UTF8_LENGTH(pyObject) = 0;

  if (JS::LinearStringHasLatin1Chars(lstr)) { // latin1 spidermonkey, latin1 python
    const JS::Latin1Char *chars = JS::GetLatin1LinearStringChars(nogc, lstr);

    PY_UNICODE_OBJECT_DATA_ANY(pyObject) = (void *)chars;
    PY_UNICODE_OBJECT_KIND(pyObject) = PyUnicode_1BYTE_KIND;
    PY_UNICODE_OBJECT_LENGTH(pyObject) = length;
  #if PY_UNICODE_HAS_WSTR
    PY_UNICODE_OBJECT_WSTR(pyObject) = NULL;
    PY_UNICODE_OBJECT_WSTR_LENGTH(pyObject) = 0;
    PY_UNICODE_OBJECT_READY(pyObject) = 1;
  #endif
  }
  else { // utf16 spidermonkey, ucs2 python
    const char16_t *chars = JS::GetTwoByteLinearStringChars(nogc, lstr);

    PY_UNICODE_OBJECT_DATA_ANY(pyObject) = (void *)chars;
    PY_UNICODE_OBJECT_KIND(pyObject) = PyUnicode_2BYTE_KIND;
    PY_UNICODE_OBJECT_LENGTH(pyObject) = length;

  #if PY_UNICODE_HAS_WSTR
    // python unicode objects take advantage of a possible performance gain on systems where
    // sizeof(wchar_t) == 2, i.e. Windows systems if the string is using UCS2 encoding by setting the
    // wstr pointer to point to the same data as the data.any pointer.
    // On systems where sizeof(wchar_t) == 4, i.e. Unixy systems, a similar performance gain happens if the
    // string is using UCS4 encoding [this is automatically handled by asUCS4()]
    if (sizeof(wchar_t) == 2) {
      PY_UNICODE_OBJECT_WSTR(pyObject) = (wchar_t *)chars;
      PY_UNICODE_OBJECT_WSTR_LENGTH(pyObject) = length;
    }
    else {
      PY_UNICODE_OBJECT_WSTR(pyObject) = NULL;
      PY_UNICODE_OBJECT_WSTR_LENGTH(pyObject) = 0;
    }
    PY_UNICODE_OBJECT_READY(pyObject) = 1;
  #endif

    if (containsSurrogatePair(chars, length)) {
      // We must convert to UCS4 here because Python does not support decoding string containing surrogate pairs to bytes
      PyObject *ucs4Obj = asUCS4(pyObject); // convert to a new PyUnicodeObject with UCS4 data
      if (!ucs4Obj) {
        // conversion fails, keep the original `pyObject`
        return;
      }
      Py_DECREF(pyObject); // cleanup the old `pyObject`
      Py_INCREF(ucs4Obj); // XXX: Same as the above `Py_INCREF(pyObject);`. Why double freed on GC?
      pyObject = ucs4Obj;
    }
  }
}

const char *StrType::getValue() const {
  return PyUnicode_AsUTF8(pyObject);
}

/* static */
PyObject *StrType::asUCS4(PyObject *pyObject) {
  if (PyUnicode_KIND(pyObject) != PyUnicode_2BYTE_KIND) {
    // return a new reference to match the behaviour of `PyUnicode_FromKindAndData`
    Py_INCREF(pyObject);
    return pyObject;
  }

  uint16_t *chars = PY_UNICODE_OBJECT_DATA_UCS2(pyObject);
  size_t length = PY_UNICODE_OBJECT_LENGTH(pyObject);

  uint32_t ucs4String[length];
  size_t ucs4Length = 0;

  for (size_t i = 0; i < length; i++, ucs4Length++) {
    if (Py_UNICODE_IS_LOW_SURROGATE(chars[i])) { // character is an unpaired low surrogate
      return NULL;
    } else if (Py_UNICODE_IS_HIGH_SURROGATE(chars[i])) { // character is a high surrogate
      if ((i + 1 < length) && Py_UNICODE_IS_LOW_SURROGATE(chars[i+1])) { // next character is a low surrogate
        ucs4String[ucs4Length] = Py_UNICODE_JOIN_SURROGATES(chars[i], chars[i+1]);
        i++; // skip over low surrogate
      }
      else { // next character is not a low surrogate
        return NULL;
      }
    } else { // character is not a surrogate, and is in the BMP
      ucs4String[ucs4Length] = chars[i];
    }
  }

  return PyUnicode_FromKindAndData(PyUnicode_4BYTE_KIND, ucs4String, ucs4Length);
}