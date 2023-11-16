/****************************************************************************
**
** https://www.qxorm.com/
** Copyright (C) 2013 Lionel Marty (contact@qxorm.com)
**
** This file is part of the QxOrm library
**
** This software is provided 'as-is', without any express or implied
** warranty. In no event will the authors be held liable for any
** damages arising from the use of this software
**
** Commercial Usage
** Licensees holding valid commercial QxOrm licenses may use this file in
** accordance with the commercial license agreement provided with the
** Software or, alternatively, in accordance with the terms contained in
** a written agreement between you and Lionel Marty
**
** GNU General Public License Usage
** Alternatively, this file may be used under the terms of the GNU
** General Public License version 3.0 as published by the Free Software
** Foundation and appearing in the file 'license.gpl3.txt' included in the
** packaging of this file. Please review the following information to
** ensure the GNU General Public License version 3.0 requirements will be
** met : http://www.gnu.org/copyleft/gpl.html
**
** If you are unsure which license is appropriate for your use, or
** if you have questions regarding the use of this file, please contact :
** contact@qxorm.com
**
****************************************************************************/

#include <QxPrecompiled.h>

#include <QxDao/QxSqlGenerator/QxSqlGenerator_PostgreSQL.h>

#include <QxDao/QxSqlDatabase.h>
#include <QxDao/IxDao_Helper.h>

#include <QxRegister/QxClassX.h>

#include <QxMemLeak/mem_leak.h>

namespace qx {
namespace dao {
namespace detail {

QxSqlGenerator_PostgreSQL::QxSqlGenerator_PostgreSQL() : QxSqlGenerator_Standard() { this->initSqlTypeByClassName(); }

QxSqlGenerator_PostgreSQL::~QxSqlGenerator_PostgreSQL() { ; }

void QxSqlGenerator_PostgreSQL::checkSqlInsert(IxDao_Helper * pDaoHelper, QString & sql) const
{
   if (! pDaoHelper) { qAssert(false); return; }
   if (! pDaoHelper->getDataId()) { return; }
   qx::IxDataMember * pId = pDaoHelper->getDataId();
   if (! pId->getAutoIncrement()) { return; }
   if (pId->getNameCount() > 1) { qAssert(false); return; }
   QString sqlToAdd = " RETURNING " + pId->getName();
   if (sql.right(sqlToAdd.size()) == sqlToAdd) { return; }
   sql += sqlToAdd;
   pDaoHelper->builder().setSqlQuery(sql);
}

void QxSqlGenerator_PostgreSQL::onAfterInsert(IxDao_Helper * pDaoHelper, void * pOwner) const
{
   if (! pDaoHelper || ! pOwner) { qAssert(false); return; }
   if (! pDaoHelper->getDataId()) { return; }
   qx::IxDataMember * pId = pDaoHelper->getDataId();
   if (! pId->getAutoIncrement()) { return; }
   if (pId->getNameCount() > 1) { qAssert(false); return; }
   if (! pDaoHelper->nextRecord()) { qAssert(false); return; }
   QVariant vId = pDaoHelper->query().value(0);
   pId->fromVariant(pOwner, vId, -1, qx::cvt::context::e_database);
}

void QxSqlGenerator_PostgreSQL::initSqlTypeByClassName() const
{
   QHash<QString, QString> * lstSqlType = qx::QxClassX::getAllSqlTypeByClassName();
   if (! lstSqlType) { qAssert(false); return; }

   lstSqlType->insert("bool", "BOOLEAN");
   lstSqlType->insert("qx_bool", "TEXT");
   lstSqlType->insert("short", "SMALLINT");
   lstSqlType->insert("int", "INTEGER");
   lstSqlType->insert("long", "INTEGER");
   lstSqlType->insert("long long", "BIGINT");
   lstSqlType->insert("float", "FLOAT");
   lstSqlType->insert("double", "FLOAT");
   lstSqlType->insert("long double", "FLOAT");
   lstSqlType->insert("unsigned short", "SMALLINT");
   lstSqlType->insert("unsigned int", "INTEGER");
   lstSqlType->insert("unsigned long", "INTEGER");
   lstSqlType->insert("unsigned long long", "BIGINT");
   lstSqlType->insert("std::string", "TEXT");
   lstSqlType->insert("std::wstring", "TEXT");
   lstSqlType->insert("QString", "TEXT");
   lstSqlType->insert("QVariant", "TEXT");
   lstSqlType->insert("QUuid", "TEXT");
   lstSqlType->insert("QDate", "DATE");
   lstSqlType->insert("QTime", "TIME");
   lstSqlType->insert("QDateTime", "TIMESTAMP");
   lstSqlType->insert("QByteArray", "BYTEA");
   lstSqlType->insert("qx::QxDateNeutral", "TEXT");
   lstSqlType->insert("qx::QxTimeNeutral", "TEXT");
   lstSqlType->insert("qx::QxDateTimeNeutral", "TEXT");
}

} // namespace detail
} // namespace dao
} // namespace qx
