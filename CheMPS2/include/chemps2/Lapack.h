/*
   CheMPS2: a spin-adapted implementation of DMRG for ab initio quantum chemistry
   Copyright (C) 2013-2017 Sebastian Wouters

   This program is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation; either version 2 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License along
   with this program; if not, write to the Free Software Foundation, Inc.,
   51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
*/

#ifndef LAPACK_CHEMPS2_H
#define LAPACK_CHEMPS2_H

#include <complex>

typedef std::complex< double > dcomplex;

extern "C" {

// Routines for doubles
void dgeqrf_( int * m, int * n, double * A, int * LDA, double * tau, double * WORK, int * LWORK, int * INFO );
void dorgqr_( int * m, int * n, int * k, double * A, int * LDA, double * tau, double * WORK, int * LWORK, int * INFO );
void dgelqf_( int * m, int * n, double * A, int * LDA, double * tau, double * WORK, int * LWORK, int * INFO );
void dorglq_( int * m, int * n, int * k, double * A, int * LDA, double * tau, double * WORK, int * LWORK, int * INFO );
void dcopy_( int * n, double * x, int * incx, double * y, int * incy );
void daxpy_( int * n, double * alpha, double * x, int * incx, double * y, int * incy );
void dscal_( int * n, double * alpha, double * x, int * incx );
void dgemm_( char * transA, char * transB, int * m, int * n, int * k, double * alpha, double * A, int * lda, double * B, int * ldb, double * beta, double * C, int * ldc );
void dgemv_( char * trans, int * m, int * n, double * alpha, double * A, int * lda, double * X, int * incx, double * beta, double * Y, int * incy );
void dsyev_( char * jobz, char * uplo, int * n, double * A, int * lda, double * W, double * work, int * lwork, int * info );
void dgesdd_( char * JOBZ, int * M, int * N, double * A, int * LDA, double * S, double * U, int * LDU, double * VT, int * LDVT, double * WORK, int * LWORK, int * IWORK, int * INFO );
void dlasrt_( char * id, int * n, double * vec, int * info );

double ddot_( int * n, double * x, int * incx, double * y, int * incy );
double dlansy_( char * norm, char * uplo, int * dimR, double * mx, int * lda, double * work );
double dlange_( char * norm, int * m, int * n, double * mx, int * lda, double * work );

// Routines for complex
void zscal_( int * n, dcomplex * alpha, dcomplex * x, int * incx );
void zgeqrf_( int * m, int * n, dcomplex * A, int * LDA, dcomplex * tau, dcomplex * WORK, int * LWORK, int * INFO );
void zungqr_( int * m, int * n, int * k, dcomplex * A, int * LDA, dcomplex * tau, dcomplex * WORK, int * LWORK, int * INFO );
void zgelqf_( int * m, int * n, dcomplex * A, int * LDA, dcomplex * tau, dcomplex * WORK, int * LWORK, int * INFO );
void zunglq_( int * m, int * n, int * k, dcomplex * A, int * LDA, dcomplex * tau, dcomplex * WORK, int * LWORK, int * INFO );
void zgemm_( char * transA, char * transB, int * m, int * n, int * k, dcomplex * alpha, dcomplex * A, int * lda, dcomplex * B, int * ldb, dcomplex * beta, dcomplex * C, int * ldc );
void zcopy_( int * n, dcomplex * x, int * incx, dcomplex * y, int * incy );
void zaxpy_( int * n, dcomplex * alpha, dcomplex * x, int * incx, dcomplex * y, int * incy );
void zgesdd_( char * JOBZ, int * M, int * N, dcomplex * A, int * LDA, double * S, dcomplex * U, int * LDU, dcomplex * VT, int * LDVT, dcomplex * WORK, int * LWORK, double * RWORK, int * IWORK, int * INFO );
void zheev_( char * JOBZ, char * UPLO, int * N, dcomplex * A, int * LDA, double * W, dcomplex * WORK, int * LWORK, double * RWORK, int * INFO );
void zgeev_( char * JOBVL, char * JOBVR, int * N, dcomplex * A, int * LDA, dcomplex * W, dcomplex * VL, int *LDVL, dcomplex* VR, int* LDVR, dcomplex * WORK, int * LWORK, double * RWORK, int * INFO );
void zgpadm_( int * ideg, int * m, double * t, dcomplex * H, int * ldh, dcomplex * wsp, int * lwsp, int * ipiv, int * iexph, int * ns, int * iflag);
void zpotri_( char * uplo, int * n, dcomplex * a, int * lda, int * info );
void zgetri_( int * n, dcomplex * a, int * lda, int * ipiv, dcomplex * work, int * lwork, int * info );
void zgetrf_( int * m, int * n, dcomplex * a, int * lda, int * ipiv, int * info );
double zlansy_( char * norm, char * uplo, int * dimR, dcomplex * mx, int * lda, dcomplex * work );

#ifdef CHEMPS2_MKL
void zdotc_( dcomplex * r, int * n, dcomplex * x, int * incx, dcomplex * y, int * incy );
#else
dcomplex zdotc_( int * n, dcomplex * x, int * incx, dcomplex * y, int * incy );
#endif

}
#endif
