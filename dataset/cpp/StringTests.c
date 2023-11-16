#include <stdlib.h>
#include <memory.h>
#include <stdbool.h>
#include "comp.h"

__declspec(dllexport) LPCSTR GetString()
{
    return "Hello from C!";
}

__declspec(dllexport) LPCSTR GetAllocatedString()
{
    char* ptr = malloc(strlen(GetString()) + 1);
    strcpy(ptr, GetString());

    return ptr;
}

__declspec(dllexport) LPCSTR GetNullString()
{
    return NULL;
}

__declspec(dllexport) BCSTR EchoBStr(BCSTR string)
{
    return string;
}

__declspec(dllexport) LPTCSTR GetLPTString()
{
    return LPTSTR("Hello from C!");
}

__declspec(dllexport) LPWCSTR GetLPWString()
{
    return (LPWCSTR) u"Hello from C!";
}

__declspec(dllexport) size_t StringLength(LPCSTR value)
{
    return strlen(value);
}

__declspec(dllexport) size_t BStringLength(BCSTR value)
{
    size_t length = 0;
    memcpy(&length, (char*)value, 2);

    return (length / sizeof(char16_t)) - 1;
}

__declspec(dllexport) size_t LPWStringLength(LPWCSTR value)
{
    LPWCSTR start = value;
    LPWCSTR end = value;

    // Increment until end
    while (*end++);

    return (end - start) - 1;
}

__declspec(dllexport) size_t LPTStringLength(LPTCSTR value)
{
    #if UNICODE
        return LPWStringLength(value);
    #else
        return strlen(value);
    #endif
}

__declspec(dllexport) bool CheckIfStringIsNull(const char* value)
{
    return value == NULL;
}


__declspec(dllexport) LPUTF8STR GetLPUTF8String()
{
    return u8"Hello, 🦈!";
}

__declspec(dllexport) size_t LPUTF8StringLength(LPUTF8CSTR value)
{
    // taken from https://hashnode.com/post/utf-8-string-length-ciibz8f4a011tj3xtj9qvku8i

    size_t i = 0;
    size_t length = 0;
    while (value[i])
    {
        if (((unsigned char)value[i] & 0xc0u) != 0x80)
        {
            length++;
        }

        i++;
    }

    return length;
}
