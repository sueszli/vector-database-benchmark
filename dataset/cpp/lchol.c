//------------------------------------------------------------------------------
// CHOLMOD/MATLAB/lchol: MATLAB interface to CHOLMOD factorization
//------------------------------------------------------------------------------

// CHOLMOD/MATLAB Module.  Copyright (C) 2005-2022, Timothy A. Davis.
// All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0+

//------------------------------------------------------------------------------

/* Numeric LL' factorization.  Note that LL' and LDL' are faster than R'R
 * and use less memory.  The LL' factorization methods use tril(A).
 *
 * L = lchol (A)		same as L = chol (A)', just faster
 * [L,p] = lchol (A)		save as [R,p] = chol(A) ; L=R', just faster
 * [L,p,q] = lchol (A)		factorizes A(q,q) into L*L'
 *
 * A must be sparse.  It can be complex or real.
 *
 * L is returned with no explicit zero entries.  This means it might not be
 * chordal, and L cannot be passed back to CHOLMOD for an update/downdate or
 * for a fast simplicial solve.  spones (L) will be equal to the L returned
 * by symbfact2 if no numerically zero entries are dropped, or a subset
 * otherwise.
 */

#include "cholmod_matlab.h"

void mexFunction
(
    int	nargout,
    mxArray *pargout [ ],
    int	nargin,
    const mxArray *pargin [ ]
)
{
    double dummy = 0, *px ;
    cholmod_sparse Amatrix, *A, *Lsparse ;
    cholmod_factor *L ;
    cholmod_common Common, *cm ;
    int64_t n, minor ;

    /* ---------------------------------------------------------------------- */
    /* start CHOLMOD and set parameters */ 
    /* ---------------------------------------------------------------------- */

    cm = &Common ;
    cholmod_l_start (cm) ;
    sputil_config (SPUMONI, cm) ;

    /* convert to packed LL' when done */
    cm->final_asis = FALSE ;
    cm->final_super = FALSE ;
    cm->final_ll = TRUE ;
    cm->final_pack = TRUE ;
    cm->final_monotonic = TRUE ;

    /* no need to prune entries due to relaxed supernodal amalgamation, since
     * zeros are dropped with sputil_drop_zeros instead */
    cm->final_resymbol = FALSE ;

    cm->quick_return_if_not_posdef = (nargout < 2) ;

    /* ---------------------------------------------------------------------- */
    /* get inputs */
    /* ---------------------------------------------------------------------- */

    if (nargin != 1 || nargout > 3)
    {
	mexErrMsgTxt ("usage: [L,p,q] = lchol (A)") ;
    }

    n = mxGetN (pargin [0]) ;

    if (!mxIsSparse (pargin [0]) || n != mxGetM (pargin [0]))
    {
    	mexErrMsgTxt ("A must be square and sparse") ;
    }

    /* get sparse matrix A, use tril(A)  */
    A = sputil_get_sparse (pargin [0], &Amatrix, &dummy, -1) ; 

    /* use natural ordering if no q output parameter */
    if (nargout < 3)
    {
	cm->nmethods = 1 ;
	cm->method [0].ordering = CHOLMOD_NATURAL ;
	cm->postorder = FALSE ;
    }

    /* ---------------------------------------------------------------------- */
    /* analyze and factorize */
    /* ---------------------------------------------------------------------- */

    L = cholmod_l_analyze (A, cm) ;
    cholmod_l_factorize (A, L, cm) ;

    if (nargout < 2 && cm->status != CHOLMOD_OK)
    {
	mexErrMsgTxt ("matrix is not positive definite") ;
    }

    /* ---------------------------------------------------------------------- */
    /* convert L to a sparse matrix */
    /* ---------------------------------------------------------------------- */

    /* the conversion sets L->minor back to n, so get a copy of it first */
    minor = L->minor ;
    Lsparse = cholmod_l_factor_to_sparse (L, cm) ;
    if (Lsparse->xtype == CHOLMOD_COMPLEX)
    {
	/* convert Lsparse from complex to zomplex */
	cholmod_l_sparse_xtype (CHOLMOD_ZOMPLEX, Lsparse, cm) ;
    }

    if (minor < n)
    {
	/* remove columns minor to n-1 from Lsparse */
	sputil_trim (Lsparse, minor, cm) ;
    }

    /* drop zeros from Lsparse */
    sputil_drop_zeros (Lsparse) ;

    /* ---------------------------------------------------------------------- */
    /* return results to MATLAB */
    /* ---------------------------------------------------------------------- */

    /* return L as a sparse matrix */
    pargout [0] = sputil_put_sparse (&Lsparse, cm) ;

    /* return minor (translate to MATLAB convention) */
    if (nargout > 1)
    {
	pargout [1] = mxCreateDoubleMatrix (1, 1, mxREAL) ;
	px = mxGetPr (pargout [1]) ;
	px [0] = ((minor == n) ? 0 : (minor+1)) ;
    }

    /* return permutation */
    if (nargout > 2)
    {
	pargout [2] = sputil_put_int (L->Perm, n, 1) ;
    }

    /* ---------------------------------------------------------------------- */
    /* free workspace and the CHOLMOD L, except for what is copied to MATLAB */
    /* ---------------------------------------------------------------------- */

    cholmod_l_free_factor (&L, cm) ;
    cholmod_l_finish (cm) ;
    cholmod_l_print_common (" ", cm) ;
    /* 
    if (cm->malloc_count != 3 + mxIsComplex (pargout[0])) mexErrMsgTxt ("!") ;
    */
}
