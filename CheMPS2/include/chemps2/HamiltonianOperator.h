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

#ifndef HAMILTONIANOPERATOR_CHEMPS2_H
#define HAMILTONIANOPERATOR_CHEMPS2_H

#include "CTensorF0.h"
#include "CTensorF0T.h"
#include "CTensorF1.h"
#include "CTensorF1T.h"
#include "CTensorL.h"
#include "CTensorLT.h"
#include "CTensorQ.h"
#include "CTensorQT.h"
#include "CTensorS0.h"
#include "CTensorS0T.h"
#include "CTensorS1.h"
#include "CTensorS1T.h"
#include "CTensorT.h"
#include "CTensorX.h"
#include "Problem.h"

namespace CheMPS2 {
   /** HamiltonianOperator class.
    \author Lars-Hendrik Frahm
    \date February 22, 2018
    
 */
   class HamiltonianOperator {

      public:
      //! Constructor
      /** \param Norbitals The number of orbitals (L)
             \param nGroup The group number
             \param OrbIrreps Pointer to array containing the orbital irreps */
      HamiltonianOperator( Problem * probIn );

      ~HamiltonianOperator();

      dcomplex ExpectationValue( CTensorT ** mps, SyBookkeeper * bk );

      dcomplex Overlap( CTensorT ** mpsLeft, SyBookkeeper * bkLeft, CTensorT ** mpsRight, SyBookkeeper * bkRight );

      void ApplyAndAdd( CTensorT ** mpsA, SyBookkeeper * bkA,
                        int statesToAdd,
                        dcomplex * factors,
                        CTensorT *** states,
                        SyBookkeeper ** bookkeepers,
                        CTensorT ** mpsOut, SyBookkeeper * bkOut,
                        int numberOfSweeps = 2 );

      void Sum( int statesToAdd,
                dcomplex * factors, CTensorT *** states, SyBookkeeper ** bookkeepers,
                CTensorT ** mpsOut, SyBookkeeper * bkOut, 
                int numberOfSweeps = 2 );

      private:
      void updateMovingLeftSafe( const int cnt, CTensorT ** mpsUp, SyBookkeeper * bkUp, CTensorT ** mpsDown, SyBookkeeper * bkDown );

      void updateMovingRightSafe( const int cnt, CTensorT ** mpsUp, SyBookkeeper * bkUp, CTensorT ** mpsDown, SyBookkeeper * bkDown );

      void deleteAllBoundaryOperators();

      void updateMovingLeft( const int index, CTensorT ** mpsUp, SyBookkeeper * bkUp, CTensorT ** mpsDown, SyBookkeeper * bkDown );

      void updateMovingRight( const int index, CTensorT ** mpsUp, SyBookkeeper * bkUp, CTensorT ** mpsDown, SyBookkeeper * bkDown );

      void allocateTensors( const int index, const bool movingRight, SyBookkeeper * bkUp, SyBookkeeper * bkDown );

      void deleteTensors( const int index, const bool movingRight );

      Problem * prob;

      int L;

      // Whether or not allocated
      int * isAllocated;

      // TensorL's
      CTensorL *** Ltensors;
      CTensorLT *** LtensorsT;

      // TensorF0F0's
      CTensorF0 **** F0tensors;
      CTensorF0T **** F0tensorsT;

      // TensorF1F1's
      CTensorF1 **** F1tensors;
      CTensorF1T **** F1tensorsT;

      // TensorS0S0's
      CTensorS0 **** S0tensors;
      CTensorS0T **** S0tensorsT;

      // TensorS1S1's
      CTensorS1 **** S1tensors;
      CTensorS1T **** S1tensorsT;

      // ABCD-tensors
      CTensorOperator **** Atensors;
      CTensorOperator **** AtensorsT;
      CTensorOperator **** Btensors;
      CTensorOperator **** BtensorsT;
      CTensorOperator **** Ctensors;
      CTensorOperator **** CtensorsT;
      CTensorOperator **** Dtensors;
      CTensorOperator **** DtensorsT;

      // TensorQQ's
      CTensorQ *** Qtensors;
      CTensorQT *** QtensorsT;

      // TensorX's
      CTensorX ** Xtensors;

      // TensorO's
      CTensorO ** Otensors;
   };
} // namespace CheMPS2

#endif
