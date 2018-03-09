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

#ifndef CONEDM_CHEMPS2_H
#define CONEDM_CHEMPS2_H

#include "CTensorF0.h"
#include "CTensorF1.h"
#include "CTensorL.h"
#include "CTensorOperator.h"
#include "CTensorT.h"

namespace CheMPS2 {

   class COneDM {

      public:
      //! Constructor
      /** \param denBKIn Symmetry sector bookkeeper
             \param ProbIn The problem to be solved */
      COneDM( CTensorT ** mps, const SyBookkeeper * denBKIn );

      ~COneDM();

      void gOEDMRe( double * array );

      void gOEDMIm( double * array );

      private:
      void updateMovingRightSafe( const int cnt );

      void updateMovingRight( const int index );

      void allocateTensors( const int index );

      void deleteAllBoundaryOperators();

      void deleteTensors( const int index );

      const int L;

      //Whether or not allocated
      int * isAllocated;

      const SyBookkeeper * denBK;

      CTensorT ** mps;

      dcomplex * matrix;

      //TensorL's
      CTensorL *** Ltensors;

      //TensorF0's
      CTensorF0 **** F0tensors;

      //TensorF1's
      CTensorF1 **** F1tensors;

      // TensorO's
      CTensorO ** Otensors;
   };
} // namespace CheMPS2

#endif
