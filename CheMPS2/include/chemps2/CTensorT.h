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

#ifndef CTENSORT_CHEMPS2_H
#define CTENSORT_CHEMPS2_H

#include "CTensor.h"
#include "SyBookkeeper.h"

#include <vector>

namespace CheMPS2 {
   /** CTensorT class.
    \author Sebastian Wouters <sebastianwouters@gmail.com>
    \date February 18, 2013
  
    The CTensorT class is a storage and gauge transformation class for MPS tensors. */
   class CTensorT : public CTensor {

      public:
      //! Constructor
      /** \param site_index The site index
             \param denBK The symmetry bookkeeper of the MPS */
      CTensorT( const int site_index, const SyBookkeeper * denBK );

      CTensorT( CTensorT * cpy );

      //! Destructor
      virtual ~CTensorT();

      //! Get the number of symmetry blocks
      /** \return The number of symmetry blocks */
      int gNKappa() const;

      //! Get the pointer to the storage
      /** return pointer to the storage */
      dcomplex * gStorage();

      //! Get the index corresponding to a certain tensor block
      /** \param N1 The left particle number sector
             \param TwoS1 The left spin symmetry sector
             \param I1 The left irrep sector
             \param N2 The right particle number sector
             \param TwoS2 The right spin symmetry sector
             \param I2 The right irrep sector
             \return The kappa corresponding to the input parameters; -1 means no such block */
      int gKappa( const int N1, const int TwoS1, const int I1, const int N2, const int TwoS2, const int I2 ) const;

      //! Get the storage jump corresponding to a certain tensor block
      /** \param kappa The symmetry block
             \return kappa2index[ kappa ], the memory jumper to a certain block */
      int gKappa2index( const int kappa ) const;

      //! Get the pointer to the storage of a certain tensor block
      /** \param N1 The left particle number sector
             \param TwoS1 The left spin symmetry sector
             \param I1 The left irrep sector
             \param N2 The right particle number sector
             \param TwoS2 The right spin symmetry sector
             \param I2 The right irrep sector
             \return Pointer to the storage of the specified tensor block; NULL means no such block */
      dcomplex * gStorage( const int N1, const int TwoS1, const int I1, const int N2, const int TwoS2, const int I2 );

      //! Get the location index
      /** \return the index */
      int gIndex() const;

      int gNL( const int kappa ) const { return sectorNL[ kappa ]; }

      int gTwoSL( const int kappa ) const { return sectorTwoSL[ kappa ]; }

      int gIL( const int kappa ) const { return sectorIL[ kappa ]; }

      int gNR( const int kappa ) const { return sectorNR[ kappa ]; }

      int gTwoSR( const int kappa ) const { return sectorTwoSR[ kappa ]; }

      int gIR( const int kappa ) const { return sectorIR[ kappa ]; }

      //! Get the pointer to the symmetry bookkeeper
      /** \return the pointer to the symmetry bookkeeper */
      const SyBookkeeper * gBK() const;

      //! Set the pointer to the symmetry bookkeeper
      /** \param newBK The pointer to the symmetry bookkeeper */
      void sBK( const SyBookkeeper * newBK );

      //! Fill storage with random numbers 0 < val < 1.
      void random();

      //! Apply alpha * ( number operator ) + beta to the MPS tensor
      /** \param alpha Prefactor of the number operator
             \param beta  Constant to be multiplied with the MPS tensor */
      void number_operator( dcomplex alpha, dcomplex beta );

      void scale( dcomplex alpha );

      void add( CTensorT * toAdd );

      void addNoise( dcomplex NoiseLevel );

      //! Left-normalization
      /** \param Rstorage Where the R-part of the QR-decomposition can be stored (diagonal TensorOperator). */
      void QR( CTensor * Rstorage );

      //! Right-normalization
      /** \param Lstorage Where the L-part of the LQ-decomposition can be stored (diagonal TensorOperator). */
      void LQ( CTensor * Lstorage );

      //! Multiply at the left with a diagonal TensorOperator
      /** \param Mx The diagonal TensorOperator with which the current CTensorT should be multiplied at the left */
      void LeftMultiply( CTensor * Mx, char * trans );

      //! Multiply at the right with a diagonal TensorOperator
      /** \param Mx The diagonal TensorOperator with which the current CTensorT should be multiplied at the right */
      void RightMultiply( CTensor * Mx, char * trans );

      //! Join two TensorOperators with a CTensorT element
      void Join( CTensor * left, CTensorT * buddy, CTensor * right );

      //! Add CTensorT elements
      void zaxpy( dcomplex factor, CTensorT * y );

      // Copy all data to y
      void zcopy( CTensorT * y );

      //! Reset the CTensorT (if virtual dimensions are changed)
      void Reset();

      //! Set all elements of the CTensorT to zero
      void Clear();

      //! Check whether the CTensorT is left-normal
      /** \return Whether CTensorT is left-normal */
      bool CheckLeftNormal() const;

      //! Check whether the CTensorT is right-normal
      /** \return Whether CTensorT is right-normal */
      bool CheckRightNormal() const;

      friend std::ostream & operator<<( std::ostream & os, const CheMPS2::CTensorT & tns );

      private:
      //! The MPS bookkeeper
      const SyBookkeeper * denBK;

      //! The left particle number sector
      int * sectorNL;

      //! The right particle number sector
      int * sectorNR;

      //! The left spin sector
      int * sectorTwoSL;

      //! The right spin sector
      int * sectorTwoSR;

      //! The left irrep sector
      int * sectorIL;

      //! The right irrep sector
      int * sectorIR;

      //! Delete all arrays
      void DeleteAllArrays();

      //! Allocate all arrays
      void AllocateAllArrays();
   };

   dcomplex overlap( CTensorT ** mpsA, CTensorT ** mpsB );

   double norm( CTensorT ** mps );

   void scale( const dcomplex factor, const int L, CTensorT ** mps );

   void normalize( const int L, CTensorT ** mps );

   std::ostream & operator<<( std::ostream & os, const CheMPS2::CTensorT & tns );

   void printFCITensor( Problem * prob, CTensorT ** mps );

   dcomplex getFCICoefficient( Problem * prob, CTensorT ** mps, int * alpha, int * beta );

   void getFCITensor( Problem * prob, CTensorT ** mps, 
                      std::vector< std::vector< int > >& alphasOut,
                      std::vector< std::vector< int > >& betasOut,
                      std::vector< double >& coefsRealOut,
                      std::vector< double >& coefsImagOut);

   void left_normalize( CTensorT * left_mps, CTensorT * right_mps );

   void right_normalize( CTensorT * left_mps, CTensorT * right_mps );

   void decomposeMovingLeft( bool change, int virtualdimensionD, double cut_off,
                             CTensorT * oldLeft, SyBookkeeper * oldLeftBK,
                             CTensorT * oldRight, SyBookkeeper * oldRightBK,
                             CTensorT * newLeft, SyBookkeeper * newLeftBK,
                             CTensorT * newRight, SyBookkeeper * newRightBK );

   void decomposeMovingRight( bool change, int virtualdimensionD, double cut_off,
                              CTensorT * oldLeft, SyBookkeeper * oldLeftBK,
                              CTensorT * oldRight, SyBookkeeper * oldRightBK,
                              CTensorT * newLeft, SyBookkeeper * newLeftBK,
                              CTensorT * newRight, SyBookkeeper * newRightBK );
} // namespace CheMPS2

#endif
