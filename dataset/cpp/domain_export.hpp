#pragma once
#include <QtCore/QtGlobal>

#if defined(DOMAIN_LIBRARY)
    #define DOMAIN_EXPORT Q_DECL_EXPORT
#else
    #define DOMAIN_EXPORT Q_DECL_IMPORT
#endif
