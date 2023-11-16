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
#define SOFA_IMAGE_IMAGEEXPORTER_CPP


#include "ImageExporter.h"
#include <sofa/core/ObjectFactory.h>


namespace sofa
{

namespace component
{

namespace misc
{

using namespace defaulttype;

int ImageExporterClass = core::RegisterObject("Save an image")
        .add<ImageExporter<ImageUC> >(true)
        .add<ImageExporter<ImageD> >()
        .add<ImageExporter<ImageB> >()
#if PLUGIN_IMAGE_COMPILE_SET == PLUGIN_IMAGE_COMPILE_SET_FULL
        .add<ImageExporter<ImageC> >()
        .add<ImageExporter<ImageI> >()
        .add<ImageExporter<ImageUI> >()
        .add<ImageExporter<ImageS> >()
        .add<ImageExporter<ImageUS> >()
        .add<ImageExporter<ImageL> >()
        .add<ImageExporter<ImageUL> >()
        .add<ImageExporter<ImageF> >()
#endif
        ;

template class SOFA_IMAGE_API ImageExporter<ImageUC>;
template class SOFA_IMAGE_API ImageExporter<ImageD>;
template class SOFA_IMAGE_API ImageExporter<ImageB>;
#if PLUGIN_IMAGE_COMPILE_SET == PLUGIN_IMAGE_COMPILE_SET_FULL
template class SOFA_IMAGE_API ImageExporter<ImageC>;
template class SOFA_IMAGE_API ImageExporter<ImageI>;
template class SOFA_IMAGE_API ImageExporter<ImageUI>;
template class SOFA_IMAGE_API ImageExporter<ImageS>;
template class SOFA_IMAGE_API ImageExporter<ImageUS>;
template class SOFA_IMAGE_API ImageExporter<ImageL>;
template class SOFA_IMAGE_API ImageExporter<ImageUL>;
template class SOFA_IMAGE_API ImageExporter<ImageF>;
#endif

} // namespace misc

} // namespace component

} // namespace sofa
