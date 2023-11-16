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
#define SOFA_IMAGE_MARCHINGCUBESENGINE_CPP

#include "MarchingCubesEngine.h"
#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace engine
{

using namespace defaulttype;

int MarchingCubesEngineClass = core::RegisterObject("Compute an isosurface from an image using marching cubes algorithm")
        .add<MarchingCubesEngine<ImageUC> >(true)
        .add<MarchingCubesEngine<ImageD> >()
        .add<MarchingCubesEngine<ImageB> >()
#if PLUGIN_IMAGE_COMPILE_SET == PLUGIN_IMAGE_COMPILE_SET_FULL
        .add<MarchingCubesEngine<ImageC> >()
        .add<MarchingCubesEngine<ImageI> >()
        .add<MarchingCubesEngine<ImageUI> >()
        .add<MarchingCubesEngine<ImageS> >()
        .add<MarchingCubesEngine<ImageUS> >()
        .add<MarchingCubesEngine<ImageL> >()
        .add<MarchingCubesEngine<ImageUL> >()
        .add<MarchingCubesEngine<ImageF> >()
#endif
        ;

template class SOFA_IMAGE_API MarchingCubesEngine<ImageUC>;
template class SOFA_IMAGE_API MarchingCubesEngine<ImageD>;
template class SOFA_IMAGE_API MarchingCubesEngine<ImageB>;
#if PLUGIN_IMAGE_COMPILE_SET == PLUGIN_IMAGE_COMPILE_SET_FULL
template class SOFA_IMAGE_API MarchingCubesEngine<ImageC>;
template class SOFA_IMAGE_API MarchingCubesEngine<ImageI>;
template class SOFA_IMAGE_API MarchingCubesEngine<ImageUI>;
template class SOFA_IMAGE_API MarchingCubesEngine<ImageS>;
template class SOFA_IMAGE_API MarchingCubesEngine<ImageUS>;
template class SOFA_IMAGE_API MarchingCubesEngine<ImageL>;
template class SOFA_IMAGE_API MarchingCubesEngine<ImageUL>;
template class SOFA_IMAGE_API MarchingCubesEngine<ImageF>;
#endif

} //
} // namespace component

} // namespace sofa

