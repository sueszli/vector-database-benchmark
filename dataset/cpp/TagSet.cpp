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
#include <sofa/core/objectmodel/TagSet.h>

namespace sofa::core::objectmodel
{

TagSet::TagSet(const Tag& t)
{
    this->insert(t);
}

bool TagSet::includes(const Tag& t) const
{
    return this->count(t) > 0;
}

bool TagSet::includes(const TagSet& t) const
{
    if (t.empty())
        return true;
    if (empty())
    {
        // An empty TagSet satisfies the conditions only if either :
        // t is also empty (already handled)
        // t only includes negative tags
        if (*t.rbegin() <= Tag(0))
            return true;
        // t includes the "0" tag
        if (t.count(Tag(0)) > 0)
            return true;
        // otherwise the TagSet t does not "include" empty sets
        return false;
    }
    for (std::set<Tag>::const_iterator first2 = t.begin(), last2 = t.end();
        first2 != last2; ++first2)
    {
        Tag t2 = *first2;
        if (t2 == Tag(0)) continue; // tag "0" is used to indicate that we should include objects without any tag
        if (!t2.negative())
        {
            if (this->count(t2) == 0)
                return false; // tag not found in this
        }
        else
        {
            if (this->count(-t2) > 0)
                return false; // tag found in this
        }
    }
    return true;
}

} //namespace sofa::core::objectmodel
