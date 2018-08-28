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

#include <iostream>
#include <fstream>
#include <algorithm>
#include <math.h>
#include <string.h>
#include <getopt.h>
#include <sys/stat.h>
#include <assert.h>
#include <sstream>

#include "Initialize.h"
#include "CASSCF.h"
#include "Molden.h"
#include "MPIchemps2.h"
#include "EdmistonRuedenberg.h"
#include "TimeEvolution.h"
#include "CFCI.h"
#include "CTensorO.h"
#include "Lapack.h"
#include "Irreps.h"

using namespace std;

int main( int argc, char ** argv ){

   std::cout << "hi from the Playground" << std::endl;

   const int L = 10;

   CheMPS2::Initialize::Init();
   string fcidump = "h10_sto-6g-ao.fcidump";
   CheMPS2::Hamiltonian * ham = new CheMPS2::Hamiltonian( fcidump, 0 );
   // CheMPS2::ConvergenceScheme * opt_scheme = new CheMPS2::ConvergenceScheme( ni_d );
   // for ( int count = 0; count < ni_d; count++ ){
   //    opt_scheme->set_instruction( count, value_states[ count ],
   //                                        value_cutoff[ count ],
   //                                        value_maxit [ count ],
   //                                        value_noise [ count ]);
   // }
   // delete [] value_states;
   // delete [] value_cutoff;
   // delete [] value_maxit;
   // delete [] value_noise;

   CheMPS2::Problem * prob = new CheMPS2::Problem( ham, 0, 10, 0 );

   CheMPS2::SyBookkeeper * bkIn  = new CheMPS2::SyBookkeeper( prob, 500 );

   for( int bla = 0; bla < 1000; bla++ ){
      CheMPS2::CTensorT** mpsIn = new CheMPS2::CTensorT *[ prob->gL() ];

      for ( int index = 0; index < prob->gL(); index++ ) {
         mpsIn[ index ] = new CheMPS2::CTensorT( index, bkIn );
         mpsIn[ index ]->random();
      }
      
      CheMPS2::CTensorO * overlapOld;
      overlapOld = new CheMPS2::CTensorO( 1, true, bkIn, bkIn );
      overlapOld->create( mpsIn[ 0 ], mpsIn[ 0 ] );

      CheMPS2::CTensorO * overlapNext;

      for ( int i = 1; i < L; i++ ) {
         // std::cout << i << std::endl;
         overlapNext = new CheMPS2::CTensorO( i + 1, true, bkIn, bkIn );
         overlapNext->update_ownmem( mpsIn[ i ], mpsIn[ i ], overlapOld );
         delete overlapOld;
         overlapOld = overlapNext;

      }

      assert( overlapOld->gNKappa() == 1 );
      dcomplex result = overlapOld->trace();
      delete overlapOld;

      std::cout << bla << " " << result << std::endl;

   }

   // for( int bla = 0; bla < 1000; bla++ ){
   //    CheMPS2::TensorT** mpsIn = new CheMPS2::TensorT *[ prob->gL() ];

   //    for ( int index = 0; index < prob->gL(); index++ ) {
   //       mpsIn[ index ] = new CheMPS2::TensorT( index, bkIn );
   //       mpsIn[ index ]->random();
   //    }
      
   //    CheMPS2::TensorO * overlapOld;
   //    overlapOld = new CheMPS2::TensorO( 1, true, bkIn, bkIn );
   //    overlapOld->create( mpsIn[ 0 ], mpsIn[ 0 ] );

   //    CheMPS2::TensorO * overlapNext;

   //    for ( int i = 1; i < L; i++ ) {
   //       // std::cout << i << std::endl;
   //       overlapNext = new CheMPS2::TensorO( i + 1, true, bkIn, bkIn );
   //       overlapNext->update_ownmem( mpsIn[ i ], mpsIn[ i ], overlapOld );
   //       delete overlapOld;
   //       overlapOld = overlapNext;

   //    }

   //    assert( overlapOld->gNKappa() == 1 );
   //    dcomplex result = overlapOld->gStorage()[0];
   //    delete overlapOld;

   //    std::cout << bla << " " << result << std::endl;

   // }


   // int dim = 1000;

   // // dcomplex * dataA = new dcomplex[ dim*dim ];
   // // dcomplex * dataB = new dcomplex[ dim*dim ];
   // // dcomplex * dataC = new dcomplex[ dim*dim ];
   
   // // char trans = 'C';
   // // char notrans = 'N';
   // // dcomplex alpha = 1.0;

   // // for(size_t i = 0; i < 100; i++)
   // // {
   // //    zgemm_( &notrans, &notrans, &dim, &dim, &dim, &alpha, dataA, &dim, dataB, &dim, &alpha, dataC, &dim );     
   // // }
   

   // double * dataA = new double[ dim*dim ];
   // double * dataB = new double[ dim*dim ];
   // double * dataC = new double[ dim*dim ];

   // char trans = 'C';
   // char notrans = 'N';
   // double alpha = 1.0;
   
   // for(size_t i = 0; i < 100; i++)
   // {
   // dgemm_( &notrans, &notrans, &dim, &dim, &dim, &alpha, dataA, &dim, dataB, &dim, &alpha, dataC, &dim );
   // }


   return 0;

}