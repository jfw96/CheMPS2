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

#ifndef CTWODMBUILDER_CHEMPS2_H
#define CTWODMBUILDER_CHEMPS2_H

#include "CTensorF0.h"
#include "CTensorF1.h"
#include "CTensorL.h"
#include "CTensorQ.h"
#include "CTensorS0.h"
#include "CTensorS1.h"
#include "CTensorT.h"
#include "CTensorX.h"
#include "CTwoDM.h"
#include "Irreps.h"
#include "SyBookkeeper.h"

namespace CheMPS2 {

   class CTwoDMBuilder {

      public:
      CTwoDMBuilder( Problem * ProbIn, CTensorT ** MpsIn, SyBookkeeper * bk_in );

      //! Destructor
      virtual ~CTwoDMBuilder();

      void Build2RDM( CTwoDM * rdm );

      private:

      //Pointer to the Problem --> constructed and destructed outside of this class
      Problem * Prob;

      int L;

      //Symmetry information object
      SyBookkeeper * denBK;

      //The MPS
      CTensorT ** MPS;

      //Whether or not allocated
      int * isAllocated;

      //TensorL's
      CTensorL *** Ltensors;

      //TensorX's
      CTensorX ** Xtensors;

      //TensorF0's
      CTensorF0 **** F0tensors;

      //TensorF1's
      CTensorF1 **** F1tensors;

      //TensorS0's
      CTensorS0 **** S0tensors;

      //TensorS1's
      CTensorS1 **** S1tensors;

      //TwoDM
      CTwoDM * the2DM;

      void deleteAllBoundaryOperators();
      void deleteTensors( const int index, const bool movingRight );
      void allocateTensors( const int index, const bool movingRight );
      // void updateMovingLeftSafeFirstTime( const int cnt );
      void updateMovingRightSafeFirstTime( const int cnt );
      void updateMovingLeftSafe( const int cnt );
      // void updateMovingRightSafe( const int cnt );
      void updateMovingLeft( const int index );
      void updateMovingRight( const int index );
   };
} // namespace CheMPS2

#endif