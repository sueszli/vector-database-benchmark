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
#include <sofa/component/topology/mapping/Edge2QuadTopologicalMapping.h>
#include <sofa/core/visual/VisualParams.h>

#include <sofa/core/ObjectFactory.h>

#include <sofa/component/topology/container/dynamic/QuadSetTopologyModifier.h>
#include <sofa/component/topology/container/dynamic/QuadSetTopologyContainer.h>

#include <sofa/core/topology/TopologyChange.h>

#include <sofa/type/Vec.h>
#include <map>
#include <sofa/defaulttype/VecTypes.h>

#include <sofa/core/behavior/MechanicalState.h>

#include <cmath>

#include <sofa/defaulttype/RigidTypes.h>


namespace sofa::component::topology::mapping
{

using namespace type;
using namespace sofa::defaulttype;
using namespace sofa::component::topology::mapping;
using namespace sofa::core::topology;

// Register in the Factory
int Edge2QuadTopologicalMappingClass = core::RegisterObject("Special case of mapping where EdgeSetTopology is converted to QuadSetTopology.")
        .add< Edge2QuadTopologicalMapping >()

        ;

// Implementation
Edge2QuadTopologicalMapping::Edge2QuadTopologicalMapping()
    : TopologicalMapping()
    , d_nbPointsOnEachCircle( initData(&d_nbPointsOnEachCircle, "nbPointsOnEachCircle", "Discretization of created circles"))
    , d_radius( initData(&d_radius, 1_sreal, "radius", "Radius of created circles in yz plan"))
    , d_radiusFocal( initData(&d_radiusFocal, 0_sreal, "radiusFocal", "If greater than 0., radius in focal axis of created ellipses"))
    , d_focalAxis( initData(&d_focalAxis, Vec3(0_sreal, 0_sreal, 1_sreal), "focalAxis", "In case of ellipses"))
    , d_edgeList(initData(&d_edgeList, "edgeList", "list of input edges for the topological mapping: by default, all considered"))
    , d_flipNormals(initData(&d_flipNormals, bool(false), "flipNormals", "Flip Normal ? (Inverse point order when creating quad)"))
{
    m_inputType = geometry::ElementType::EDGE;
    m_outputType = geometry::ElementType::QUAD;
}

void Edge2QuadTopologicalMapping::init()
{
    d_componentState.setValue(sofa::core::objectmodel::ComponentState::Invalid);

    if (!d_radius.isSet() && !d_radiusFocal.isSet())
    {
        msg_error() << "No radius (neither radius nor radiusFocal) defined";
        return;
    }

    if (d_radius.isSet() && d_radius.getValue() < std::numeric_limits<SReal>::min())
    {
        msg_error() << "Radius is zero or negative";
        return;
    }
    if (d_radiusFocal.isSet() && d_radiusFocal.getValue() < std::numeric_limits<SReal>::min())
    {
        msg_warning() << "Focal Radius is zero or negative";
    }

    const SReal rho = d_radius.getValue();

    bool ellipse = false;
    SReal rhoFocal{};
    if (d_radiusFocal.isSet() && d_radiusFocal.getValue() >= std::numeric_limits<SReal>::min())
    {
        ellipse = true;
        rhoFocal = d_radiusFocal.getValue();
    }

    const unsigned int N = d_nbPointsOnEachCircle.getValue();

    // Check input/output topology
    if (!this->checkTopologyInputTypes()) // method will display error message if false
    {
        this->d_componentState.setValue(sofa::core::objectmodel::ComponentState::Invalid);
        return;
    }

    
    // INITIALISATION of QUADULAR mesh from EDGE mesh :
    const core::behavior::MechanicalState<Rigid3Types>* from_mstate = dynamic_cast<core::behavior::MechanicalState<Rigid3Types>*>(fromModel->getContext()->getMechanicalState());
    core::behavior::MechanicalState<Vec3Types>* to_mstate = dynamic_cast<core::behavior::MechanicalState<Vec3Types>*>(toModel->getContext()->getMechanicalState());

    if (fromModel)
    {
        msg_info() << "Edge2QuadTopologicalMapping - from = edge";

        if (toModel)
        {
            msg_info() << "Edge2QuadTopologicalMapping - to = quad";

            container::dynamic::QuadSetTopologyModifier *to_modifier;
            toModel->getContext()->get(to_modifier);

            container::dynamic::QuadSetTopologyContainer *to_container;
            toModel->getContext()->get(to_container);

            const sofa::type::vector<Edge>& edgeArray = fromModel->getEdges();
            
            auto Loc2GlobVec = sofa::helper::getWriteOnlyAccessor(Loc2GlobDataVec);
            Loc2GlobVec.clear();
            In2OutMap.clear();

            // CREATION of the points (new DOFs for the output topology) along the circles around each point of the input topology

            const Vec3 X0(1.,0.,0.);
            Vec3 Y0;
            Vec3 Z0;

            if (ellipse){
                Z0 = d_focalAxis.getValue();
                Z0.normalize();
                Y0 = cross(Z0,X0);
            } else {
                Y0[0] = (SReal) (0.0); Y0[1] = (SReal) (1.0); Y0[2] = (SReal) (0.0);
                Z0[0] = (SReal) (0.0); Z0[1] = (SReal) (0.0); Z0[2] = (SReal) (1.0);
            }

            if (to_mstate)
            {
                to_mstate->resize(fromModel->getNbPoints() * N);
            }

            to_container->clear();

            toModel->setNbPoints(fromModel->getNbPoints() * N);

            if (to_mstate)
            {
                for (unsigned int i=0; i<(unsigned int) fromModel->getNbPoints(); ++i)
                {
                    const unsigned int p0=i;

                    Mat3x3 rotation;
                    (from_mstate->read(core::ConstVecCoordId::position())->getValue())[p0].writeRotationMatrix(rotation);

                    Vec3 t;
                    t=(from_mstate->read(core::ConstVecCoordId::position())->getValue())[p0].getCenter();

                    Vec3 Y;
                    Vec3 Z;

                    Y = rotation * Y0;
                    Z = rotation * Z0;

                    helper::WriteAccessor< Data< Vec3Types::VecCoord > > to_x = *to_mstate->write(core::VecCoordId::position());

                    for(unsigned int j=0; j<N; ++j)
                    {
                        Vec3 x;
                        if(ellipse){
                            x = t + Y*cos((SReal) (2.0*j*M_PI/N))*((SReal) rho) + Z*sin((SReal) (2.0*j*M_PI/N))*((SReal) rhoFocal);
                        } else {
                            x = t + (Y*cos((SReal) (2.0*j*M_PI/N)) + Z*sin((SReal) (2.0*j*M_PI/N)))*((SReal) rho);
                        }
                        to_x[p0*N+j] = x;
                    }
                }
            }


            // CREATION of the quads based on the circles
            sofa::type::vector< Quad > quads_to_create;
            sofa::type::vector< Index > quadsIndexList;
            if(d_edgeList.getValue().size()==0)
            {

                unsigned int nb_elems = (unsigned int)toModel->getNbQuads();

                for (unsigned int i=0; i<edgeArray.size(); ++i)
                {
                    const unsigned int p0 = edgeArray[i][0];
                    const unsigned int p1 = edgeArray[i][1];

                    sofa::type::vector<Index> out_info;

                    for(unsigned int j=0; j<N; ++j)
                    {
                        const unsigned int q0 = p0*N+j;
                        const unsigned int q1 = p1*N+j;
                        const unsigned int q2 = p1*N+((j+1)%N);
                        const unsigned int q3 = p0*N+((j+1)%N);

                        if (d_flipNormals.getValue())
                        {
                            Quad q = Quad((unsigned int) q3, (unsigned int) q2, (unsigned int) q1, (unsigned int) q0);
                            quads_to_create.push_back(q);
                            quadsIndexList.push_back(nb_elems);
                        }

                        else
                        {
                            Quad q = Quad((unsigned int) q0, (unsigned int) q1, (unsigned int) q2, (unsigned int) q3);
                            quads_to_create.push_back(q);
                            quadsIndexList.push_back(nb_elems);
                        }

                        Loc2GlobVec.push_back(i);
                        out_info.push_back((unsigned int)Loc2GlobVec.size()-1);
                    }


                    nb_elems++;

                    In2OutMap[i]=out_info;
                }
            }
            else
            {
                for (const auto i : d_edgeList.getValue())
                {
                    const Index p0 = edgeArray[i][0];
                    const Index p1 = edgeArray[i][1];

                    sofa::type::vector<Index> out_info;

                    for(unsigned int j=0; j<N; ++j)
                    {
                        const Index q0 = p0*N+j;
                        const Index q1 = p1*N+j;
                        const Index q2 = p1*N+((j+1)%N);
                        const Index q3 = p0*N+((j+1)%N);

                        if(d_flipNormals.getValue())
                            to_modifier->addQuadProcess(Quad(q0, q3, q2, q1));
                        else
                            to_modifier->addQuadProcess(Quad(q0, q1, q2, q3));
                        Loc2GlobVec.push_back(i);
                        out_info.push_back((unsigned int)Loc2GlobVec.size()-1);
                    }

                    In2OutMap[i]=out_info;
                }

            }

            to_modifier->addQuads(quads_to_create);

            // Need to fully init the target topology
            to_modifier->init();

            d_componentState.setValue(sofa::core::objectmodel::ComponentState::Valid);
        }

    }
    else
    {
        // Check type Rigid3 of input mechanical object (required)
        msg_error() << "Mechanical object associated with the input is not of type Rigid. Edge2QuadTopologicalMapping only supports Rigid3Types to Vec3Types";
        d_componentState.setValue(sofa::core::objectmodel::ComponentState::Invalid);
    }
}


Index Edge2QuadTopologicalMapping::getFromIndex(Index ind)
{
    return ind; // identity
}

void Edge2QuadTopologicalMapping::updateTopologicalMappingTopDown()
{
    if (d_componentState.getValue() != sofa::core::objectmodel::ComponentState::Valid)
        return;

    unsigned int N = d_nbPointsOnEachCircle.getValue();

    // INITIALISATION of QUADULAR mesh from EDGE mesh :

    if (fromModel)
    {

        container::dynamic::QuadSetTopologyModifier *to_modifier;
        toModel->getContext()->get(to_modifier);

        if (toModel)
        {

            std::list<const TopologyChange *>::const_iterator itBegin=fromModel->beginChange();
            std::list<const TopologyChange *>::const_iterator itEnd=fromModel->endChange();
            auto Loc2GlobVec = sofa::helper::getWriteAccessor(Loc2GlobDataVec);

            while( itBegin != itEnd )
            {
                TopologyChangeType changeType = (*itBegin)->getChangeType();

                switch( changeType )
                {

                case core::topology::ENDING_EVENT:
                {
                    to_modifier->notifyEndingEvent();
                    break;
                }

                case core::topology::EDGESADDED:
                {
                    if (fromModel)
                    {

                        const sofa::type::vector<Edge> &edgeArray=fromModel->getEdges();

                        const auto &tab = ( static_cast< const EdgesAdded *>( *itBegin ) )->edgeIndexArray;

                        sofa::type::vector< Quad > quads_to_create;
                        sofa::type::vector< Index > quadsIndexList;
                        std::size_t nb_elems = toModel->getNbQuads();

                        for (unsigned int i = 0; i < tab.size(); ++i)
                        {
                            unsigned int k = tab[i];

                            unsigned int p0 = edgeArray[k][0];
                            unsigned int p1 = edgeArray[k][1];

                            sofa::type::vector<Index> out_info;

                            for(unsigned int j=0; j<N; ++j)
                            {

                                unsigned int q0 = p0*N+j;
                                unsigned int q1 = p1*N+j;
                                unsigned int q2 = p1*N+((j+1)%N);
                                unsigned int q3 = p0*N+((j+1)%N);

                                Quad t = Quad((unsigned int) q0, (unsigned int) q1, (unsigned int) q2, (unsigned int) q3);

                                //quads_to_create.clear();
                                //quadsIndexList.clear();

                                quads_to_create.push_back(t);
                                quadsIndexList.push_back(nb_elems);
                                nb_elems+=1;

                                Loc2GlobVec.push_back(k);
                                out_info.push_back((unsigned int)Loc2GlobVec.size()-1);

                                //to_modifier->addQuadsProcess(quads_to_create) ;
                                //to_modifier->addQuadsWarning(quads_to_create.size(), quads_to_create, quadsIndexList) ;
                                //to_modifier->propagateTopologicalChanges();
                            }

                            In2OutMap[k]=out_info;
                        }

                        to_modifier->addQuads(quads_to_create);
                    }
                    break;
                }
                case core::topology::EDGESREMOVED:
                {
                    if (fromModel)
                    {
                        const auto &tab = ( static_cast< const EdgesRemoved *>( *itBegin ) )->getArray();

                        unsigned int last = (unsigned int)fromModel->getNbEdges() - 1;

                        Index ind_tmp;

                        sofa::type::vector<Index> ind_real_last;
                        Index ind_last = toModel->getNbQuads();

                        for (unsigned int i = 0; i < tab.size(); ++i)
                        {
                            unsigned int k = tab[i];
                            sofa::type::vector<Index> ind_k;

                            auto iter_1 = In2OutMap.find(k);
                            if(iter_1 != In2OutMap.end())
                            {

                                sofa::type::vector<unsigned int> ind_list;
                                for(unsigned int j=0; j<N; ++j)
                                {
                                    ind_list.push_back(In2OutMap[k][j]);
                                }

                                ind_last = ind_last - 1;

                                ind_k = In2OutMap[k];
                                ind_real_last = ind_k;

                                auto iter_2 = In2OutMap.find(last);
                                if(iter_2 != In2OutMap.end())
                                {

                                    ind_real_last = In2OutMap[last];

                                    if (k != last)
                                    {

                                        In2OutMap.erase(In2OutMap.find(k));
                                        In2OutMap[k] = ind_real_last;

                                        In2OutMap.erase(In2OutMap.find(last));
                                        In2OutMap[last] = ind_k;

                                        for(unsigned int j=0; j<N; ++j)
                                        {

                                            ind_tmp = Loc2GlobVec[ind_real_last[j]];
                                            Loc2GlobVec[ind_real_last[j]] = Loc2GlobVec[ind_k[j]];
                                            Loc2GlobVec[ind_k[j]] = ind_tmp;
                                        }
                                    }
                                }
                                else
                                {
                                    msg_info() << "INFO_print : Edge2QuadTopologicalMapping - In2OutMap should have the edge " << last;
                                }

                                if (ind_k[N-1] != ind_last)
                                {

                                    In2OutMap.erase(In2OutMap.find(Loc2GlobVec[ind_last]));
                                    In2OutMap[Loc2GlobVec[ind_last]] = ind_k;

                                    sofa::type::vector<Index> out_info;
                                    for(unsigned int j=0; j<N; ++j)
                                    {
                                        out_info.push_back(ind_last-j);
                                    }

                                    In2OutMap.erase(In2OutMap.find(Loc2GlobVec[ind_k[N-1]]));
                                    In2OutMap[Loc2GlobVec[ind_k[N-1]]] = out_info;

                                    ind_tmp = Loc2GlobVec[ind_k[N-1]];
                                    Loc2GlobVec[ind_k[N-1]] = Loc2GlobVec[ind_last];
                                    Loc2GlobVec[ind_last] = ind_tmp;

                                }

                                for(unsigned int j=1; j<N; ++j)
                                {

                                    ind_last = ind_last-1;

                                    if (ind_k[N-1-j] != ind_last)
                                    {

                                        ind_tmp = Loc2GlobVec[ind_k[N-1-j]];
                                        Loc2GlobVec[ind_k[N-1-j]] = Loc2GlobVec[ind_last];
                                        Loc2GlobVec[ind_last] = ind_tmp;
                                    }
                                }

                                In2OutMap.erase(In2OutMap.find(Loc2GlobVec[Loc2GlobVec.size() - 1]));

                                Loc2GlobVec.resize( Loc2GlobVec.size() - N );

                                sofa::type::vector< Index > quads_to_remove;
                                for(unsigned int j=0; j<N; ++j)
                                {
                                    quads_to_remove.push_back(ind_list[j]);
                                }

                                to_modifier->removeQuads(quads_to_remove, true, true);

                            }
                            else
                            {
                                msg_info() << "INFO_print : Edge2QuadTopologicalMapping - In2OutMap should have the edge " << k;
                            }

                            --last;
                        }
                    }

                    break;
                }

                case core::topology::POINTSRENUMBERING:
                {
                    const auto &tab = ( static_cast< const PointsRenumbering * >( *itBegin ) )->getIndexArray();
                    const auto &inv_tab = ( static_cast< const PointsRenumbering * >( *itBegin ) )->getinv_IndexArray();

                    sofa::type::vector<Index> indices;
                    sofa::type::vector<Index> inv_indices;

                    for(unsigned int i = 0; i < tab.size(); ++i)
                    {

                        for(unsigned int j=0; j<N; ++j)
                        {
                            indices.push_back(tab[i]*N + j);
                            inv_indices.push_back(inv_tab[i]*N + j);
                        }
                    }

                    sofa::type::vector<Index>& tab_indices = indices;
                    sofa::type::vector<Index>& inv_tab_indices = inv_indices;

                    to_modifier->renumberPoints(tab_indices, inv_tab_indices, true);
                    break;
                }

                case core::topology::POINTSADDED:
                {
                    const auto *ta=static_cast< const sofa::core::topology::PointsAdded * >( *itBegin );

                    unsigned int to_nVertices = (unsigned int)ta->getNbAddedVertices() * N;
                    sofa::type::vector< sofa::type::vector< Index > > to_ancestorsList;
                    sofa::type::vector< sofa::type::vector< SReal > > to_coefs;

                    for(unsigned int i =0; i < ta->getNbAddedVertices(); i++)
                    {
                        sofa::type::vector< Index > my_ancestors;
                        sofa::type::vector< SReal > my_coefs;

                        for(unsigned int j =0; j < N; j++)
                        {

                            for(unsigned int k = 0; k < ta->ancestorsList[i].size(); k++)
                            {
                                my_ancestors.push_back(ta->ancestorsList[i][k]*N + j);
                            }
                            for(unsigned int k = 0; k < ta->coefs[i].size(); k++)
                            {
                                my_coefs.push_back(ta->coefs[i][k]*N + j);
                            }

                            to_ancestorsList.push_back(my_ancestors);
                            to_coefs.push_back(my_coefs);
                        }
                    }

                    to_modifier->addPoints(to_nVertices, to_ancestorsList, to_coefs, true);
                    break;
                }

                default:
                    // Ignore events that are not Quad  related.
                    break;
                }

                ++itBegin;
            }
        }
    }
    return;
}

} //namespace sofa::component::topology::mapping
