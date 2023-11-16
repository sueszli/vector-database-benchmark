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
#define SOFA_IMAGE_IMAGEACCUMULATOR_CPP

#include "ImageAccumulator.h"
#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace engine
{

using namespace defaulttype;

int ImageAccumulatorClass = core::RegisterObject ( "Wraps images from a video stream into a single image" )
        .add<ImageAccumulator<ImageUC> >(true)
        .add<ImageAccumulator<ImageD> >()
        .add<ImageAccumulator<ImageB> >()
#if PLUGIN_IMAGE_COMPILE_SET == PLUGIN_IMAGE_COMPILE_SET_FULL
        .add<ImageAccumulator<ImageC> >()
        .add<ImageAccumulator<ImageI> >()
        .add<ImageAccumulator<ImageUI> >()
        .add<ImageAccumulator<ImageS> >()
        .add<ImageAccumulator<ImageUS> >()
        .add<ImageAccumulator<ImageL> >()
        .add<ImageAccumulator<ImageUL> >()
        .add<ImageAccumulator<ImageF> >()
#endif
        ;

template class SOFA_IMAGE_API ImageAccumulator<ImageUC>;
template class SOFA_IMAGE_API ImageAccumulator<ImageD>;
template class SOFA_IMAGE_API ImageAccumulator<ImageB>;
#if PLUGIN_IMAGE_COMPILE_SET == PLUGIN_IMAGE_COMPILE_SET_FULL
template class SOFA_IMAGE_API ImageAccumulator<ImageC>;
template class SOFA_IMAGE_API ImageAccumulator<ImageI>;
template class SOFA_IMAGE_API ImageAccumulator<ImageUI>;
template class SOFA_IMAGE_API ImageAccumulator<ImageS>;
template class SOFA_IMAGE_API ImageAccumulator<ImageUS>;
template class SOFA_IMAGE_API ImageAccumulator<ImageL>;
template class SOFA_IMAGE_API ImageAccumulator<ImageUL>;
template class SOFA_IMAGE_API ImageAccumulator<ImageF>;
#endif



} //
} // namespace component

} // namespace sofa

