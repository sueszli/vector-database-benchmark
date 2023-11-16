/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this program. If not, see <http://www.gnu.org/licenses/>.        *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
/*****************************************************************************
* User of this library should read the documentation
* in the messaging.h file.
******************************************************************************/

#include <sofa/helper/system/console.h>
#include <sofa/helper/logging/DefaultStyleMessageFormatter.h>
#include <sofa/helper/logging/Message.h>


namespace sofa::helper::logging
{

void DefaultStyleMessageFormatter::formatMessage(const Message & m, std::ostream & out)
{

    setColor(out, m.type()) << getPrefixText(m.type());

    if (!m.sender().empty())
        out << console::Foreground::Normal::Blue << "[" << m.componentInfo() << "] ";

    out << console::Foreground::Normal::Reset << m.message().str() << std::endl;
}
} // namespace sofa::helper::logging
