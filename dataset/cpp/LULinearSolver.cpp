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
#include <SofaNewmat/LULinearSolver.h>
#include <sofa/core/visual/VisualParams.h>
#include <SofaNewmat/NewMatMatrix.h>
#include <sofa/linearalgebra/FullMatrix.h>
#include <sofa/core/ObjectFactory.h>

namespace sofa::component::linearsolver
{

using namespace sofa::defaulttype;
using namespace sofa::core::behavior;
using namespace sofa::simulation;
using namespace sofa::linearalgebra;

template<class Matrix, class Vector>
LULinearSolver<Matrix,Vector>::LULinearSolver()
    : f_verbose( initData(&f_verbose,false,"verbose","Dump system state at each iteration") )
    , solver(nullptr), computedMinv(false)
{
}

template<class Matrix, class Vector>
LULinearSolver<Matrix,Vector>::~LULinearSolver()
{
    if (solver != nullptr)
        delete solver;
}

template<class Matrix, class Vector>
void LULinearSolver<Matrix,Vector>::invert (Matrix& M)
{
    if (solver != nullptr)
        delete solver;
    solver = M.makeLUSolver();
    computedMinv = false;
}

template<class Matrix, class Vector>
void LULinearSolver<Matrix,Vector>::solve (Matrix& M, Vector& x, Vector& b)
{



    const bool verbose  = f_verbose.getValue();

    if( verbose )
    {
        msg_info() << "LULinearSolver, b = " << b;
        msg_info() << "LULinearSolver, M = " << M;
    }
    if (solver)
        M.solve(&x,&b, solver);
    else
        M.solve(&x,&b);

    // x is the solution of the system
    if( verbose )
    {
        msg_info() << "LULinearSolver::solve, solution = " << x;
    }
}

template<class Matrix, class Vector>
void LULinearSolver<Matrix,Vector>::computeMinv()
{
    if (!computedMinv)
    {
        if (solver)
            Minv = solver->i();
        else
            Minv = this->linearSystem.systemMatrix->i();
        computedMinv = true;
    }
}

template<class Matrix, class Vector>
double LULinearSolver<Matrix,Vector>::getMinvElement(int i, int j)
{
    return Minv.element(i,j);
}

template<class Matrix, class Vector>
template<class RMatrix, class JMatrix>
bool LULinearSolver<Matrix, Vector>::addJMInvJt(RMatrix& result, JMatrix& J, double fact)
{
    const unsigned int Jrows = J.rowSize();
    const unsigned int Jcols = J.colSize();
    if (Jcols != (unsigned int)this->linearSystem.systemMatrix->rowSize())
    {
        msg_error() << "AddJMInvJt ERROR: incompatible J matrix size.";
        return false;
    }

    if (!Jrows) return false;
    computeMinv();

    const typename JMatrix::LineConstIterator jitend = J.end();
    for (typename JMatrix::LineConstIterator jit1 = J.begin(); jit1 != jitend; ++jit1)
    {
        int row1 = jit1->first;
        for (typename JMatrix::LineConstIterator jit2 = jit1; jit2 != jitend; ++jit2)
        {
            int row2 = jit2->first;
            double acc = 0.0;
            for (typename JMatrix::LElementConstIterator i1 = jit1->second.begin(), i1end = jit1->second.end(); i1 != i1end; ++i1)
            {
                int col1 = i1->first;
                double val1 = i1->second;
                for (typename JMatrix::LElementConstIterator i2 = jit2->second.begin(), i2end = jit2->second.end(); i2 != i2end; ++i2)
                {
                    int col2 = i2->first;
                    double val2 = i2->second;
                    acc += val1 * getMinvElement(col1,col2) * val2;
                }
            }
            acc *= fact;

            result.add(row1,row2,acc);
            if (row1!=row2)
                result.add(row2,row1,acc);
        }
    }
    return true;
}

template<class Matrix, class Vector>
bool LULinearSolver<Matrix,Vector>::addJMInvJt(linearalgebra::BaseMatrix* result, linearalgebra::BaseMatrix* J, double fact)
{
    if (FullMatrix<double>* r = dynamic_cast<FullMatrix<double>*>(result))
    {
        if (SparseMatrix<double>* j = dynamic_cast<SparseMatrix<double>*>(J))
        {
            return addJMInvJt(*r,*j,fact);
        }
        else if (SparseMatrix<float>* j = dynamic_cast<SparseMatrix<float>*>(J))
        {
            return addJMInvJt(*r,*j,fact);
        }
    }
    else if (FullMatrix<double>* r = dynamic_cast<FullMatrix<double>*>(result))
    {
        if (SparseMatrix<double>* j = dynamic_cast<SparseMatrix<double>*>(J))
        {
            return addJMInvJt(*r,*j,fact);
        }
        else if (SparseMatrix<float>* j = dynamic_cast<SparseMatrix<float>*>(J))
        {
            return addJMInvJt(*r,*j,fact);
        }
    }
    else if (linearalgebra::BaseMatrix* r = result)
    {
        if (SparseMatrix<double>* j = dynamic_cast<SparseMatrix<double>*>(J))
        {
            return addJMInvJt(*r,*j,fact);
        }
        else if (SparseMatrix<float>* j = dynamic_cast<SparseMatrix<float>*>(J))
        {
            return addJMInvJt(*r,*j,fact);
        }
    }
    return false;
}

int LULinearSolverClass = core::RegisterObject("Direct linear solver based on LU factorization")
        .add< LULinearSolver<NewMatMatrix,NewMatVector> >(true)
        .add< LULinearSolver<NewMatSymmetricMatrix,NewMatVector> >()
        .add< LULinearSolver<NewMatBandMatrix,NewMatVector> >()
        .add< LULinearSolver<NewMatSymmetricBandMatrix,NewMatVector> >()
        ;

} // namespace sofa::component::linearsolver
