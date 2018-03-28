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

#include "CHeffNS.h"
#include "CHeffNS_1S.h"
#include "CTensorOperator.h"
#include "CTensorT.h"
#include "HamiltonianOperator.h"
#include "Special.h"

CheMPS2::HamiltonianOperator::HamiltonianOperator( Problem * probIn ) : prob( probIn ), L( probIn->gL() ) {

   Ltensors    = new CTensorL **[ L - 1 ];
   LtensorsT   = new CTensorLT **[ L - 1 ];
   F0tensors   = new CTensorF0 ***[ L - 1 ];
   F0tensorsT  = new CTensorF0T ***[ L - 1 ];
   F1tensors   = new CTensorF1 ***[ L - 1 ];
   F1tensorsT  = new CTensorF1T ***[ L - 1 ];
   S0tensors   = new CTensorS0 ***[ L - 1 ];
   S0tensorsT  = new CTensorS0T ***[ L - 1 ];
   S1tensors   = new CTensorS1 ***[ L - 1 ];
   S1tensorsT  = new CTensorS1T ***[ L - 1 ];
   Atensors    = new CTensorOperator ***[ L - 1 ];
   AtensorsT   = new CTensorOperator ***[ L - 1 ];
   Btensors    = new CTensorOperator ***[ L - 1 ];
   BtensorsT   = new CTensorOperator ***[ L - 1 ];
   Ctensors    = new CTensorOperator ***[ L - 1 ];
   CtensorsT   = new CTensorOperator ***[ L - 1 ];
   Dtensors    = new CTensorOperator ***[ L - 1 ];
   DtensorsT   = new CTensorOperator ***[ L - 1 ];
   Qtensors    = new CTensorQ **[ L - 1 ];
   QtensorsT   = new CTensorQT **[ L - 1 ];
   Xtensors    = new CTensorX *[ L - 1 ];
   Otensors    = new CTensorO *[ L - 1 ];
   isAllocated = new int[ L - 1 ];

   for ( int cnt = 0; cnt < L - 1; cnt++ ) {
      isAllocated[ cnt ] = 0;
   }
}

CheMPS2::HamiltonianOperator::~HamiltonianOperator() {

   deleteAllBoundaryOperators();

   delete[] Ltensors;
   delete[] LtensorsT;
   delete[] F0tensors;
   delete[] F0tensorsT;
   delete[] F1tensors;
   delete[] F1tensorsT;
   delete[] S0tensors;
   delete[] S0tensorsT;
   delete[] S1tensors;
   delete[] S1tensorsT;
   delete[] Atensors;
   delete[] AtensorsT;
   delete[] Btensors;
   delete[] BtensorsT;
   delete[] Ctensors;
   delete[] CtensorsT;
   delete[] Dtensors;
   delete[] DtensorsT;
   delete[] Qtensors;
   delete[] QtensorsT;
   delete[] Xtensors;
   delete[] Otensors;
   delete[] isAllocated;
}

dcomplex CheMPS2::HamiltonianOperator::ExpectationValue( CTensorT ** mps, SyBookkeeper * bk ) {
   return Overlap( mps, bk, mps, bk );
}

dcomplex CheMPS2::HamiltonianOperator::Overlap( CTensorT ** mpsLeft, SyBookkeeper * bkLeft, CTensorT ** mpsRight, SyBookkeeper * bkRight ) {
   deleteAllBoundaryOperators();

   for ( int cnt = 0; cnt < L - 1; cnt++ ) {
      updateMovingRightSafe( cnt, mpsLeft, bkLeft, mpsRight, bkRight );
   }

   CTensorO * lastOverlap = new CTensorO( L, true, bkLeft, bkRight );
   lastOverlap->update_ownmem( mpsLeft[ L - 1 ], mpsRight[ L - 1 ], Otensors[ L - 1 - 1 ] );

   CTensorX * last = new CTensorX( L, true, bkLeft, bkRight, prob );
   last->update( mpsLeft[ L - 1 ], mpsRight[ L - 1 ],
                 Otensors[ L - 1 - 1 ],
                 Ltensors[ L - 1 - 1 ], LtensorsT[ L - 1 - 1 ],
                 Xtensors[ L - 1 - 1 ],
                 Qtensors[ L - 1 - 1 ][ 0 ], QtensorsT[ L - 1 - 1 ][ 0 ],
                 Atensors[ L - 1 - 1 ][ 0 ][ 0 ], AtensorsT[ L - 1 - 1 ][ 0 ][ 0 ],
                 CtensorsT[ L - 1 - 1 ][ 0 ][ 0 ], DtensorsT[ L - 1 - 1 ][ 0 ][ 0 ] );
   dcomplex result = last->trace() + lastOverlap->trace() * prob->gEconst();

   delete lastOverlap;
   delete last;

   return result;
}

void CheMPS2::HamiltonianOperator::SSApplyAndAdd( CTensorT ** mpsA, SyBookkeeper * bkA,
                                                  int statesToAdd,
                                                  dcomplex * factors,
                                                  CTensorT *** states,
                                                  SyBookkeeper ** bookkeepers,
                                                  CTensorT ** mpsOut, SyBookkeeper * bkOut,
                                                  int numberOfSweeps ) {
   deleteAllBoundaryOperators();
   for ( int index = 0; index < L - 1; index++ ) {
      left_normalize( mpsOut[ index ], mpsOut[ index + 1 ] );
   }
   left_normalize( mpsOut[ L - 1 ], NULL );

   CTensorO *** overlaps = new CTensorO **[ statesToAdd ];
   for ( int idx = 0; idx < statesToAdd; idx++ ) {
      overlaps[ idx ] = new CTensorO *[ L - 1 ];
   }

   for ( int cnt = 0; cnt < L - 1; cnt++ ) {
      updateMovingRightSafe( cnt, mpsOut, bkOut, mpsA, bkA );

      for ( int st = 0; st < statesToAdd; st++ ) {
         overlaps[ st ][ cnt ] = new CTensorO( cnt + 1, true, bkOut, bookkeepers[ st ] );
         if ( cnt == 0 ) {
            overlaps[ st ][ cnt ]->create( mpsOut[ cnt ], states[ st ][ cnt ] );
         } else {
            overlaps[ st ][ cnt ]->update_ownmem( mpsOut[ cnt ], states[ st ][ cnt ], overlaps[ st ][ cnt - 1 ] );
         }
      }
   }

   for ( int i = 0; i < numberOfSweeps; ++i ) {
      for ( int site = L - 1; site > 0; site-- ) {

         CTensorT * fromAdded = new CTensorT( site, bkOut );
         fromAdded->Clear();
         for ( int st = 0; st < statesToAdd; st++ ) {
            CTensorT * add           = new CTensorT( site, bkOut );
            CTensorO * leftOverlapA  = ( site - 1 ) >= 0 ? overlaps[ st ][ site - 1 ] : NULL;
            CTensorO * rightOverlapA = ( site + 1 ) < L ? overlaps[ st ][ site ] : NULL;
            add->Join( leftOverlapA, states[ st ][ site ], rightOverlapA );
            add->zaxpy( factors[ st ], fromAdded );
            delete add;
         }

         CTensorT * applied = new CTensorT( mpsOut[ site ] );
         applied->Clear();
         CHeffNS_1S * heff = new CHeffNS_1S( bkOut, bkA, prob );
         heff->Apply( mpsA[ site ], applied,
                      Ltensors, LtensorsT,
                      Atensors, AtensorsT,
                      Btensors, BtensorsT,
                      Ctensors, CtensorsT,
                      Dtensors, DtensorsT,
                      S0tensors, S0tensorsT,
                      S1tensors, S1tensorsT,
                      F0tensors, F0tensorsT,
                      F1tensors, F1tensorsT,
                      Qtensors, QtensorsT,
                      Xtensors, Otensors, false );

         fromAdded->zaxpy( 1.0, applied );
         applied->zcopy( mpsOut[ site ] );
         delete fromAdded;
         delete applied;
         delete heff;

         right_normalize( mpsOut[ site - 1 ], mpsOut[ site ] );
         updateMovingLeftSafe( site - 1, mpsOut, bkOut, mpsA, bkA );

         // Otensors
         for ( int st = 0; st < statesToAdd; st++ ) {
            overlaps[ st ][ site - 1 ] = new CTensorO( site, false, bkOut, bookkeepers[ st ] );
            if ( site == L - 1 ) {
               overlaps[ st ][ site - 1 ]->create( mpsOut[ site ], states[ st ][ site ] );
            } else {
               overlaps[ st ][ site - 1 ]->update_ownmem( mpsOut[ site ], states[ st ][ site ], overlaps[ st ][ site ] );
            }
         }
         std::cout << i << " " << overlap( mpsOut, mpsOut ) << std::endl;
      }
      // abort();
      for ( int site = 0; site < L - 1; site++ ) {
         CTensorT * fromAdded = new CTensorT( site, bkOut );
         fromAdded->Clear();
         for ( int st = 0; st < statesToAdd; st++ ) {
            CTensorT * add           = new CTensorT( site, bkOut );
            CTensorO * leftOverlapA  = ( site - 1 ) >= 0 ? overlaps[ st ][ site - 1 ] : NULL;
            CTensorO * rightOverlapA = ( site + 1 ) < L ? overlaps[ st ][ site ] : NULL;
            add->Join( leftOverlapA, states[ st ][ site ], rightOverlapA );
            add->zaxpy( factors[ st ], fromAdded );
            delete add;
         }

         CTensorT * applied = new CTensorT( mpsOut[ site ] );
         applied->Clear();
         CHeffNS_1S * heff = new CHeffNS_1S( bkOut, bkA, prob );
         heff->Apply( mpsA[ site ], applied,
                      Ltensors, LtensorsT,
                      Atensors, AtensorsT,
                      Btensors, BtensorsT,
                      Ctensors, CtensorsT,
                      Dtensors, DtensorsT,
                      S0tensors, S0tensorsT,
                      S1tensors, S1tensorsT,
                      F0tensors, F0tensorsT,
                      F1tensors, F1tensorsT,
                      Qtensors, QtensorsT,
                      Xtensors, Otensors, true );
         delete heff;

         fromAdded->zaxpy( 1.0, applied );
         applied->zcopy( mpsOut[ site ] );
         delete fromAdded;
         delete applied;

         left_normalize( mpsOut[ site ], mpsOut[ site + 1 ] );
         updateMovingRightSafe( site, mpsOut, bkOut, mpsA, bkA );

         // Otensors
         for ( int st = 0; st < statesToAdd; st++ ) {
            overlaps[ st ][ site ] = new CTensorO( site + 1, true, bkOut, bookkeepers[ st ] );
            if ( site == 0 ) {
               overlaps[ st ][ site ]->create( mpsOut[ site ], states[ st ][ site ] );
            } else {
               overlaps[ st ][ site ]->update_ownmem( mpsOut[ site ], states[ st ][ site ], overlaps[ st ][ site - 1 ] );
            }
         }
         std::cout << i << " " << overlap( mpsOut, mpsOut ) << std::endl;
      }
   }
}

void CheMPS2::HamiltonianOperator::SSSum( int statesToAdd,
                                          dcomplex * factors, CTensorT *** states, SyBookkeeper ** bookkeepers,
                                          CTensorT ** mpsOut, SyBookkeeper * bkOut,
                                          int numberOfSweeps ) {
   deleteAllBoundaryOperators();
   for ( int index = 0; index < L - 1; index++ ) {
      left_normalize( mpsOut[ index ], mpsOut[ index + 1 ] );
   }
   left_normalize( mpsOut[ L - 1 ], NULL );

   CTensorO *** overlaps = new CTensorO **[ statesToAdd ];
   for ( int idx = 0; idx < statesToAdd; idx++ ) {
      overlaps[ idx ] = new CTensorO *[ L - 1 ];
   }

   for ( int cnt = 0; cnt < L - 1; cnt++ ) {
      for ( int st = 0; st < statesToAdd; st++ ) {
         overlaps[ st ][ cnt ] = new CTensorO( cnt + 1, true, bkOut, bookkeepers[ st ] );
         if ( cnt == 0 ) {
            overlaps[ st ][ cnt ]->create( mpsOut[ cnt ], states[ st ][ cnt ] );
         } else {
            overlaps[ st ][ cnt ]->update_ownmem( mpsOut[ cnt ], states[ st ][ cnt ], overlaps[ st ][ cnt - 1 ] );
         }
      }
   }

   for ( int i = 0; i < numberOfSweeps; ++i ) {
      for ( int site = L - 1; site > 0; site-- ) {

         CTensorT * added = new CTensorT( site, bkOut );
         added->Clear();
         for ( int st = 0; st < statesToAdd; st++ ) {
            CTensorT * add           = new CTensorT( site, bkOut );
            CTensorO * leftOverlapA  = ( site - 1 ) >= 0 ? overlaps[ st ][ site - 1 ] : NULL;
            CTensorO * rightOverlapA = ( site + 1 ) < L ? overlaps[ st ][ site ] : NULL;
            add->Join( leftOverlapA, states[ st ][ site ], rightOverlapA );
            add->zaxpy( factors[ st ], added );
            delete add;
         }

         added->zcopy( mpsOut[ site ] );
         delete added;

         right_normalize( mpsOut[ site - 1 ], mpsOut[ site ] );

         // Otensors
         for ( int st = 0; st < statesToAdd; st++ ) {
            overlaps[ st ][ site - 1 ] = new CTensorO( site, false, bkOut, bookkeepers[ st ] );
            if ( site == L - 1 ) {
               overlaps[ st ][ site - 1 ]->create( mpsOut[ site ], states[ st ][ site ] );
            } else {
               overlaps[ st ][ site - 1 ]->update_ownmem( mpsOut[ site ], states[ st ][ site ], overlaps[ st ][ site ] );
            }
         }
      }

      for ( int site = 0; site < L - 1; site++ ) {
         CTensorT * added = new CTensorT( site, bkOut );
         added->Clear();
         for ( int st = 0; st < statesToAdd; st++ ) {
            CTensorT * add           = new CTensorT( site, bkOut );
            CTensorO * leftOverlapA  = ( site - 1 ) >= 0 ? overlaps[ st ][ site - 1 ] : NULL;
            CTensorO * rightOverlapA = ( site + 1 ) < L ? overlaps[ st ][ site ] : NULL;
            add->Join( leftOverlapA, states[ st ][ site ], rightOverlapA );
            add->zaxpy( factors[ st ], added );
            delete add;
         }

         added->zcopy( mpsOut[ site ] );
         delete added;

         left_normalize( mpsOut[ site ], mpsOut[ site + 1 ] );

         // Otensors
         for ( int st = 0; st < statesToAdd; st++ ) {
            overlaps[ st ][ site ] = new CTensorO( site + 1, true, bkOut, bookkeepers[ st ] );
            if ( site == 0 ) {
               overlaps[ st ][ site ]->create( mpsOut[ site ], states[ st ][ site ] );
            } else {
               overlaps[ st ][ site ]->update_ownmem( mpsOut[ site ], states[ st ][ site ], overlaps[ st ][ site - 1 ] );
            }
         }
      }
   }

   for ( int st = 0; st < statesToAdd; st++ ) {
      for ( int cnt = 0; cnt < L - 1; cnt++ ) {
         delete overlaps[ st ][ cnt ];
      }
      delete[] overlaps[ st ];
   }
   delete overlaps;
   deleteAllBoundaryOperators();
}

void CheMPS2::HamiltonianOperator::DSApply( CTensorT ** mpsA, SyBookkeeper * bkA,
                                            CTensorT ** mpsOut, SyBookkeeper * bkOut,
                                            int numberOfSweeps, int maxM, double cutOff ) {
   DSApplyAndAdd( mpsA, bkA, 0, NULL, NULL, NULL, mpsOut, bkOut, numberOfSweeps );
}

void CheMPS2::HamiltonianOperator::DSApplyAndAdd( CTensorT ** mpsA, SyBookkeeper * bkA,
                                                  int statesToAdd,
                                                  dcomplex * factors,
                                                  CTensorT *** states,
                                                  SyBookkeeper ** bookkeepers,
                                                  CTensorT ** mpsOut, SyBookkeeper * bkOut,
                                                  int numberOfSweeps, int maxM, double cutOff  ) {
   deleteAllBoundaryOperators();
   for ( int index = 0; index < L - 2; index++ ) {
      left_normalize( mpsOut[ index ], mpsOut[ index + 1 ] );
   }

   CTensorO *** overlaps = new CTensorO **[ statesToAdd ];
   for ( int idx = 0; idx < statesToAdd; idx++ ) {
      overlaps[ idx ] = new CTensorO *[ L - 1 ];
   }

   for ( int cnt = 0; cnt < L - 1; cnt++ ) {
      updateMovingRightSafe( cnt, mpsOut, bkOut, mpsA, bkA );

      for ( int st = 0; st < statesToAdd; st++ ) {
         overlaps[ st ][ cnt ] = new CTensorO( cnt + 1, true, bkOut, bookkeepers[ st ] );
         if ( cnt == 0 ) {
            overlaps[ st ][ cnt ]->create( mpsOut[ cnt ], states[ st ][ cnt ] );
         } else {
            overlaps[ st ][ cnt ]->update_ownmem( mpsOut[ cnt ], states[ st ][ cnt ], overlaps[ st ][ cnt - 1 ] );
         }
      }
   }

   for ( int i = 0; i < numberOfSweeps; ++i ) {
      for ( int site = L - 2; site > 0; site-- ) {

         CSobject * fromAdded = new CSobject( site, bkOut );
         fromAdded->Clear();
         for ( int st = 0; st < statesToAdd; st++ ) {
            CSobject * add           = new CSobject( site, bkOut );
            CTensorO * leftOverlapA  = ( site - 1 ) >= 0 ? overlaps[ st ][ site - 1 ] : NULL;
            CTensorO * rightOverlapA = ( site + 2 ) < L ? overlaps[ st ][ site + 1 ] : NULL;
            add->Join( leftOverlapA, states[ st ][ site ], states[ st ][ site + 1 ], rightOverlapA );
            fromAdded->Add( factors[ st ], add );
            delete add;
         }

         CTensorO * leftOverlapA  = ( site - 1 ) >= 0 ? Otensors[ site - 1 ] : NULL;
         CTensorO * rightOverlapA = ( site + 2 ) < L ? Otensors[ site + 1 ] : NULL;

         CSobject * in = new CSobject( site, bkA );
         in->Clear();
         in->Join( mpsA[ site ], mpsA[ site + 1 ] );

         CSobject * applied = new CSobject( site, bkOut );

         CHeffNS * heff = new CHeffNS( bkOut, bkA, prob, 0.0 );
         heff->Apply( in, applied, Ltensors, LtensorsT, Atensors, AtensorsT,
                      Btensors, BtensorsT, Ctensors, CtensorsT, Dtensors, DtensorsT,
                      S0tensors, S0tensorsT, S1tensors, S1tensorsT, F0tensors,
                      F0tensorsT, F1tensors, F1tensorsT, Qtensors, QtensorsT,
                      Xtensors, leftOverlapA, rightOverlapA );

         applied->Add( 1.0, fromAdded );
         delete fromAdded;
         double disc = applied->Split( mpsOut[ site ], mpsOut[ site + 1 ], maxM, cutOff, false, true );
         // std::cout << disc << std::endl;
         delete heff;
         delete applied;
         delete in;

         updateMovingLeftSafe( site, mpsOut, bkOut, mpsA, bkA );

         // Otensors
         for ( int st = 0; st < statesToAdd; st++ ) {
            if ( isAllocated[ site ] > 0 ) { delete overlaps[ st ][ site ]; }
            overlaps[ st ][ site ] = new CTensorO( site + 1, false, bkOut, bookkeepers[ st ] );
            if ( site == L - 2 ) {
               overlaps[ st ][ site ]->create( mpsOut[ site + 1 ], states[ st ][ site + 1 ] );
            } else {
               overlaps[ st ][ site ]->update_ownmem( mpsOut[ site + 1 ], states[ st ][ site + 1 ], overlaps[ st ][ site + 1 ] );
            }
         }
      }

      for ( int site = 0; site < L - 2; site++ ) {

         CSobject * fromAdded = new CSobject( site, bkOut );
         fromAdded->Clear();
         for ( int st = 0; st < statesToAdd; st++ ) {
            CSobject * add           = new CSobject( site, bkOut );
            CTensorO * leftOverlapA  = ( site - 1 ) >= 0 ? overlaps[ st ][ site - 1 ] : NULL;
            CTensorO * rightOverlapA = ( site + 2 ) < L ? overlaps[ st ][ site + 1 ] : NULL;
            add->Join( leftOverlapA, states[ st ][ site ], states[ st ][ site + 1 ], rightOverlapA );
            fromAdded->Add( factors[ st ], add );
            delete add;
         }

         CTensorO * leftOverlapA  = ( site - 1 ) >= 0 ? Otensors[ site - 1 ] : NULL;
         CTensorO * rightOverlapA = ( site + 2 ) < L ? Otensors[ site + 1 ] : NULL;

         CSobject * in = new CSobject( site, bkA );
         in->Clear();
         in->Join( mpsA[ site ], mpsA[ site + 1 ] );

         CSobject * applied = new CSobject( site, bkOut );
         applied->Clear();

         CHeffNS * heff = new CHeffNS( bkOut, bkA, prob, 0.0 );
         heff->Apply( in, applied, Ltensors, LtensorsT, Atensors, AtensorsT,
                      Btensors, BtensorsT, Ctensors, CtensorsT, Dtensors, DtensorsT,
                      S0tensors, S0tensorsT, S1tensors, S1tensorsT, F0tensors,
                      F0tensorsT, F1tensors, F1tensorsT, Qtensors, QtensorsT,
                      Xtensors, leftOverlapA, rightOverlapA );

         delete heff;

         applied->Add( 1.0, fromAdded );
         double disc = applied->Split( mpsOut[ site ], mpsOut[ site + 1 ], maxM, cutOff, true, true );
         // std::cout << disc << std::endl;
         delete applied;
         delete in;
         delete fromAdded;

         updateMovingRightSafe( site, mpsOut, bkOut, mpsA, bkA );

         // Otensors
         for ( int st = 0; st < statesToAdd; st++ ) {
            if ( isAllocated[ site ] > 0 ) { delete overlaps[ st ][ site ]; }
            overlaps[ st ][ site ] = new CTensorO( site + 1, true, bkOut, bookkeepers[ st ] );
            if ( site == 0 ) {
               overlaps[ st ][ site ]->create( mpsOut[ site ], states[ st ][ site ] );
            } else {
               overlaps[ st ][ site ]->update_ownmem( mpsOut[ site ], states[ st ][ site ], overlaps[ st ][ site - 1 ] );
            }
         }
      }
   }

   for ( int st = 0; st < statesToAdd; st++ ) {
      for ( int cnt = 0; cnt < L - 1; cnt++ ) {
         delete overlaps[ st ][ cnt ];
      }
      delete[] overlaps[ st ];
   }
   delete[] overlaps;
}

void CheMPS2::HamiltonianOperator::DSSum( int statesToAdd,
                                          dcomplex * factors, CTensorT *** states, SyBookkeeper ** bookkeepers,
                                          CTensorT ** mpsOut, SyBookkeeper * bkOut,
                                          int numberOfSweeps, int maxM, double cutOff ) {
   deleteAllBoundaryOperators();

   for ( int index = 0; index < L - 1; index++ ) {
      left_normalize( mpsOut[ index ], mpsOut[ index + 1 ] );
   }
   left_normalize( mpsOut[ L - 1 ], NULL );

   CTensorO *** overlaps = new CTensorO **[ statesToAdd ];
   for ( int idx = 0; idx < statesToAdd; idx++ ) {
      overlaps[ idx ] = new CTensorO *[ L - 1 ];
   }

   for ( int cnt = 0; cnt < L - 1; cnt++ ) {
      for ( int st = 0; st < statesToAdd; st++ ) {
         overlaps[ st ][ cnt ] = new CTensorO( cnt + 1, true, bkOut, bookkeepers[ st ] );
         if ( cnt == 0 ) {
            overlaps[ st ][ cnt ]->create( mpsOut[ cnt ], states[ st ][ cnt ] );
         } else {
            overlaps[ st ][ cnt ]->update_ownmem( mpsOut[ cnt ], states[ st ][ cnt ], overlaps[ st ][ cnt - 1 ] );
         }
      }
   }

   for ( int i = 0; i < numberOfSweeps; ++i ) {
      for ( int site = L - 2; site > 0; site-- ) {

         CSobject * added = new CSobject( site, bkOut );
         added->Clear();
         for ( int st = 0; st < statesToAdd; st++ ) {
            CSobject * add           = new CSobject( site, bkOut );
            CTensorO * leftOverlapA  = ( site - 1 ) >= 0 ? overlaps[ st ][ site - 1 ] : NULL;
            CTensorO * rightOverlapA = ( site + 2 ) < L ? overlaps[ st ][ site + 1 ] : NULL;
            add->Join( leftOverlapA, states[ st ][ site ], states[ st ][ site + 1 ], rightOverlapA );
            added->Add( factors[ st ], add );
            delete add;
         }
         added->Split( mpsOut[ site ], mpsOut[ site + 1 ], maxM, cutOff, false, true );
         delete added;

         // Otensors
         for ( int st = 0; st < statesToAdd; st++ ) {
            delete overlaps[ st ][ site ];
            overlaps[ st ][ site ] = new CTensorO( site + 1, false, bkOut, bookkeepers[ st ] );
            if ( site == L - 2 ) {
               overlaps[ st ][ site ]->create( mpsOut[ site + 1 ], states[ st ][ site + 1 ] );
            } else {
               overlaps[ st ][ site ]->update_ownmem( mpsOut[ site + 1 ], states[ st ][ site + 1 ], overlaps[ st ][ site + 1 ] );
            }
         }
      }

      for ( int site = 0; site < L - 1; site++ ) {
         CSobject * added = new CSobject( site, bkOut );
         added->Clear();
         for ( int st = 0; st < statesToAdd; st++ ) {
            CSobject * add           = new CSobject( site, bkOut );
            CTensorO * leftOverlapA  = ( site - 1 ) >= 0 ? overlaps[ st ][ site - 1 ] : NULL;
            CTensorO * rightOverlapA = ( site + 2 ) < L ? overlaps[ st ][ site + 1 ] : NULL;
            add->Join( leftOverlapA, states[ st ][ site ], states[ st ][ site + 1 ], rightOverlapA );
            added->Add( factors[ st ], add );
            delete add;
         }

         added->Split( mpsOut[ site ], mpsOut[ site + 1 ], maxM, cutOff, true, true );
         delete added;

         // Otensors
         for ( int st = 0; st < statesToAdd; st++ ) {
            delete overlaps[ st ][ site ];
            overlaps[ st ][ site ] = new CTensorO( site + 1, true, bkOut, bookkeepers[ st ] );
            if ( site == 0 ) {
               overlaps[ st ][ site ]->create( mpsOut[ site ], states[ st ][ site ] );
            } else {
               overlaps[ st ][ site ]->update_ownmem( mpsOut[ site ], states[ st ][ site ], overlaps[ st ][ site - 1 ] );
            }
         }
      }
   }

   for ( int st = 0; st < statesToAdd; st++ ) {
      for ( int cnt = 0; cnt < L - 1; cnt++ ) {
         delete overlaps[ st ][ cnt ];
      }
      delete[] overlaps[ st ];
   }
   delete[] overlaps;
}

void CheMPS2::HamiltonianOperator::updateMovingLeftSafe( const int cnt, CTensorT ** mpsUp, SyBookkeeper * bkUp, CTensorT ** mpsDown, SyBookkeeper * bkDown ) {
   if ( isAllocated[ cnt ] == 1 ) {
      deleteTensors( cnt, true );
      isAllocated[ cnt ] = 0;
   }
   if ( isAllocated[ cnt ] == 0 ) {
      allocateTensors( cnt, false, bkUp, bkDown );
      isAllocated[ cnt ] = 2;
   }
   updateMovingLeft( cnt, mpsUp, bkUp, mpsDown, bkDown );
}

void CheMPS2::HamiltonianOperator::updateMovingRightSafe( const int cnt, CTensorT ** mpsUp, SyBookkeeper * bkUp, CTensorT ** mpsDown, SyBookkeeper * bkDown ) {
   if ( isAllocated[ cnt ] == 2 ) {
      deleteTensors( cnt, false );
      isAllocated[ cnt ] = 0;
   }
   if ( isAllocated[ cnt ] == 0 ) {
      allocateTensors( cnt, true, bkUp, bkDown );
      isAllocated[ cnt ] = 1;
   }
   updateMovingRight( cnt, mpsUp, bkUp, mpsDown, bkDown );
}

void CheMPS2::HamiltonianOperator::deleteAllBoundaryOperators() {
   for ( int cnt = 0; cnt < L - 1; cnt++ ) {
      if ( isAllocated[ cnt ] == 1 ) {
         deleteTensors( cnt, true );
      }
      if ( isAllocated[ cnt ] == 2 ) {
         deleteTensors( cnt, false );
      }
      isAllocated[ cnt ] = 0;
   }
}

void CheMPS2::HamiltonianOperator::updateMovingLeft( const int index, CTensorT ** mpsUp, SyBookkeeper * bkUp, CTensorT ** mpsDown, SyBookkeeper * bkDown ) {

   const int dimL = std::max( bkUp->gMaxDimAtBound( index + 1 ), bkDown->gMaxDimAtBound( index + 1 ) );
   const int dimR = std::max( bkUp->gMaxDimAtBound( index + 2 ), bkDown->gMaxDimAtBound( index + 2 ) );

#pragma omp parallel
   {
      dcomplex * workmem = new dcomplex[ dimL * dimR ];
// Ltensors_MPSDT_MPS : all processes own all Ltensors_MPSDT_MPS
#pragma omp for schedule( static ) nowait
      for ( int cnt2 = 0; cnt2 < L - 1 - index; cnt2++ ) {
         if ( cnt2 == 0 ) {
            if ( index == L - 2 ) {
               Ltensors[ index ][ cnt2 ]->create( mpsUp[ index + 1 ], mpsDown[ index + 1 ], NULL, NULL );
               LtensorsT[ index ][ cnt2 ]->create( mpsUp[ index + 1 ], mpsDown[ index + 1 ], NULL, NULL );
            } else {
               Ltensors[ index ][ cnt2 ]->create( mpsUp[ index + 1 ], mpsDown[ index + 1 ], Otensors[ index + 1 ], workmem );
               LtensorsT[ index ][ cnt2 ]->create( mpsUp[ index + 1 ], mpsDown[ index + 1 ], Otensors[ index + 1 ], workmem );
            }
         } else {
            Ltensors[ index ][ cnt2 ]->update( Ltensors[ index + 1 ][ cnt2 - 1 ], mpsUp[ index + 1 ], mpsDown[ index + 1 ], workmem );
            LtensorsT[ index ][ cnt2 ]->update( LtensorsT[ index + 1 ][ cnt2 - 1 ], mpsUp[ index + 1 ], mpsDown[ index + 1 ], workmem );
         }
      }

      // Two-operator tensors : certain processes own certain two-operator tensors
      const int k1          = L - 1 - index;
      const int upperbound1 = ( k1 * ( k1 + 1 ) ) / 2;
      int result[ 2 ];
// After this parallel region, WAIT because F0,F1,S0,S1[ index ][ cnt2 ][ cnt3
// == 0 ] is required for the complementary operators
#pragma omp for schedule( static )
      for ( int global = 0; global < upperbound1; global++ ) {
         Special::invert_triangle_two( global, result );
         const int cnt2 = k1 - 1 - result[ 1 ];
         const int cnt3 = result[ 0 ];
         if ( cnt3 == 0 ) {
            if ( cnt2 == 0 ) {
               if ( index == L - 2 ) {
                  F0tensors[ index ][ cnt2 ][ cnt3 ]->makenew( mpsUp[ index + 1 ], mpsDown[ index + 1 ], NULL, NULL );
                  F0tensorsT[ index ][ cnt2 ][ cnt3 ]->makenew( mpsUp[ index + 1 ], mpsDown[ index + 1 ], NULL, NULL );
                  F1tensors[ index ][ cnt2 ][ cnt3 ]->makenew( mpsUp[ index + 1 ], mpsDown[ index + 1 ], NULL, NULL );
                  F1tensorsT[ index ][ cnt2 ][ cnt3 ]->makenew( mpsUp[ index + 1 ], mpsDown[ index + 1 ], NULL, NULL );
                  S0tensors[ index ][ cnt2 ][ cnt3 ]->makenew( mpsUp[ index + 1 ], mpsDown[ index + 1 ], NULL, NULL );
                  S0tensorsT[ index ][ cnt2 ][ cnt3 ]->makenew( mpsUp[ index + 1 ], mpsDown[ index + 1 ], NULL, NULL );
               } else {
                  F0tensors[ index ][ cnt2 ][ cnt3 ]->makenew( mpsUp[ index + 1 ], mpsDown[ index + 1 ], Otensors[ index + 1 ], workmem );
                  F0tensorsT[ index ][ cnt2 ][ cnt3 ]->makenew( mpsUp[ index + 1 ], mpsDown[ index + 1 ], Otensors[ index + 1 ], workmem );
                  F1tensors[ index ][ cnt2 ][ cnt3 ]->makenew( mpsUp[ index + 1 ], mpsDown[ index + 1 ], Otensors[ index + 1 ], workmem );
                  F1tensorsT[ index ][ cnt2 ][ cnt3 ]->makenew( mpsUp[ index + 1 ], mpsDown[ index + 1 ], Otensors[ index + 1 ], workmem );
                  S0tensors[ index ][ cnt2 ][ cnt3 ]->makenew( mpsUp[ index + 1 ], mpsDown[ index + 1 ], Otensors[ index + 1 ], workmem );
                  S0tensorsT[ index ][ cnt2 ][ cnt3 ]->makenew( mpsUp[ index + 1 ], mpsDown[ index + 1 ], Otensors[ index + 1 ], workmem );
               }
            } else {
               F0tensors[ index ][ cnt2 ][ cnt3 ]->makenew( Ltensors[ index + 1 ][ cnt2 - 1 ], mpsUp[ index + 1 ], mpsDown[ index + 1 ], workmem );
               F0tensorsT[ index ][ cnt2 ][ cnt3 ]->makenew( LtensorsT[ index + 1 ][ cnt2 - 1 ], mpsUp[ index + 1 ], mpsDown[ index + 1 ], workmem );
               F1tensors[ index ][ cnt2 ][ cnt3 ]->makenew( Ltensors[ index + 1 ][ cnt2 - 1 ], mpsUp[ index + 1 ], mpsDown[ index + 1 ], workmem );
               F1tensorsT[ index ][ cnt2 ][ cnt3 ]->makenew( LtensorsT[ index + 1 ][ cnt2 - 1 ], mpsUp[ index + 1 ], mpsDown[ index + 1 ], workmem );
               S0tensors[ index ][ cnt2 ][ cnt3 ]->makenew( Ltensors[ index + 1 ][ cnt2 - 1 ], mpsUp[ index + 1 ], mpsDown[ index + 1 ], workmem );
               S0tensorsT[ index ][ cnt2 ][ cnt3 ]->makenew( LtensorsT[ index + 1 ][ cnt2 - 1 ], mpsUp[ index + 1 ], mpsDown[ index + 1 ], workmem );
               S1tensors[ index ][ cnt2 ][ cnt3 ]->makenew( Ltensors[ index + 1 ][ cnt2 - 1 ], mpsUp[ index + 1 ], mpsDown[ index + 1 ], workmem );
               S1tensorsT[ index ][ cnt2 ][ cnt3 ]->makenew( LtensorsT[ index + 1 ][ cnt2 - 1 ], mpsUp[ index + 1 ], mpsDown[ index + 1 ], workmem );
            }
         } else {
            F0tensors[ index ][ cnt2 ][ cnt3 ]->update( F0tensors[ index + 1 ][ cnt2 ][ cnt3 - 1 ], mpsUp[ index + 1 ], mpsDown[ index + 1 ], workmem );
            F0tensorsT[ index ][ cnt2 ][ cnt3 ]->update( F0tensorsT[ index + 1 ][ cnt2 ][ cnt3 - 1 ], mpsUp[ index + 1 ], mpsDown[ index + 1 ], workmem );
            F1tensors[ index ][ cnt2 ][ cnt3 ]->update( F1tensors[ index + 1 ][ cnt2 ][ cnt3 - 1 ], mpsUp[ index + 1 ], mpsDown[ index + 1 ], workmem );
            F1tensorsT[ index ][ cnt2 ][ cnt3 ]->update( F1tensorsT[ index + 1 ][ cnt2 ][ cnt3 - 1 ], mpsUp[ index + 1 ], mpsDown[ index + 1 ], workmem );
            S0tensors[ index ][ cnt2 ][ cnt3 ]->update( S0tensors[ index + 1 ][ cnt2 ][ cnt3 - 1 ], mpsUp[ index + 1 ], mpsDown[ index + 1 ], workmem );
            S0tensorsT[ index ][ cnt2 ][ cnt3 ]->update( S0tensorsT[ index + 1 ][ cnt2 ][ cnt3 - 1 ], mpsUp[ index + 1 ], mpsDown[ index + 1 ], workmem );
            if ( cnt2 > 0 ) {
               S1tensors[ index ][ cnt2 ][ cnt3 ]->update( S1tensors[ index + 1 ][ cnt2 ][ cnt3 - 1 ], mpsUp[ index + 1 ], mpsDown[ index + 1 ], workmem );
               S1tensorsT[ index ][ cnt2 ][ cnt3 ]->update( S1tensorsT[ index + 1 ][ cnt2 ][ cnt3 - 1 ], mpsUp[ index + 1 ], mpsDown[ index + 1 ], workmem );
            }
         }
      }

      // Complementary two-operator tensors : certain processes own certain
      // complementary two-operator tensors
      const int k2          = index + 1;
      const int upperbound2 = ( k2 * ( k2 + 1 ) ) / 2;
#pragma omp for schedule( static ) nowait
      for ( int global = 0; global < upperbound2; global++ ) {
         Special::invert_triangle_two( global, result );
         const int cnt2       = k2 - 1 - result[ 1 ];
         const int cnt3       = result[ 0 ];
         const int siteindex1 = index - cnt3 - cnt2;
         const int siteindex2 = index - cnt3;
         const int irrep_prod = Irreps::directProd( bkUp->gIrrep( siteindex1 ), bkUp->gIrrep( siteindex2 ) );
         if ( index == L - 2 ) {
            Atensors[ index ][ cnt2 ][ cnt3 ]->clear();
            AtensorsT[ index ][ cnt2 ][ cnt3 ]->clear();
            if ( cnt2 > 0 ) {
               Btensors[ index ][ cnt2 ][ cnt3 ]->clear();
               BtensorsT[ index ][ cnt2 ][ cnt3 ]->clear();
            }
            Ctensors[ index ][ cnt2 ][ cnt3 ]->clear();
            CtensorsT[ index ][ cnt2 ][ cnt3 ]->clear();
            Dtensors[ index ][ cnt2 ][ cnt3 ]->clear();
            DtensorsT[ index ][ cnt2 ][ cnt3 ]->clear();
         } else {
            Atensors[ index ][ cnt2 ][ cnt3 ]->update( Atensors[ index + 1 ][ cnt2 ][ cnt3 + 1 ], mpsUp[ index + 1 ], mpsDown[ index + 1 ], workmem );
            AtensorsT[ index ][ cnt2 ][ cnt3 ]->update( AtensorsT[ index + 1 ][ cnt2 ][ cnt3 + 1 ], mpsUp[ index + 1 ], mpsDown[ index + 1 ], workmem );
            if ( cnt2 > 0 ) {
               Btensors[ index ][ cnt2 ][ cnt3 ]->update( Btensors[ index + 1 ][ cnt2 ][ cnt3 + 1 ], mpsUp[ index + 1 ], mpsDown[ index + 1 ], workmem );
               BtensorsT[ index ][ cnt2 ][ cnt3 ]->update( BtensorsT[ index + 1 ][ cnt2 ][ cnt3 + 1 ], mpsUp[ index + 1 ], mpsDown[ index + 1 ], workmem );
            }
            Ctensors[ index ][ cnt2 ][ cnt3 ]->update( Ctensors[ index + 1 ][ cnt2 ][ cnt3 + 1 ], mpsUp[ index + 1 ], mpsDown[ index + 1 ], workmem );
            CtensorsT[ index ][ cnt2 ][ cnt3 ]->update( CtensorsT[ index + 1 ][ cnt2 ][ cnt3 + 1 ], mpsUp[ index + 1 ], mpsDown[ index + 1 ], workmem );
            Dtensors[ index ][ cnt2 ][ cnt3 ]->update( Dtensors[ index + 1 ][ cnt2 ][ cnt3 + 1 ], mpsUp[ index + 1 ], mpsDown[ index + 1 ], workmem );
            DtensorsT[ index ][ cnt2 ][ cnt3 ]->update( DtensorsT[ index + 1 ][ cnt2 ][ cnt3 + 1 ], mpsUp[ index + 1 ], mpsDown[ index + 1 ], workmem );
         }
         for ( int num = 0; num < L - index - 1; num++ ) {
            if ( irrep_prod ==
                 S0tensorsT[ index ][ num ][ 0 ]->get_irrep() ) { // Then the matrix elements are not 0 due to symm.
               double alpha = prob->gMxElement( siteindex1, siteindex2, index + 1, index + 1 + num );
               if ( ( cnt2 == 0 ) && ( num == 0 ) )
                  alpha *= 0.5;
               if ( ( cnt2 > 0 ) && ( num > 0 ) )
                  alpha += prob->gMxElement( siteindex1, siteindex2, index + 1 + num, index + 1 );
               Atensors[ index ][ cnt2 ][ cnt3 ]->zaxpy( alpha, S0tensors[ index ][ num ][ 0 ] );
               AtensorsT[ index ][ cnt2 ][ cnt3 ]->zaxpy( alpha, S0tensorsT[ index ][ num ][ 0 ] );

               if ( ( num > 0 ) && ( cnt2 > 0 ) ) {
                  alpha = prob->gMxElement( siteindex1, siteindex2, index + 1, index + 1 + num ) -
                          prob->gMxElement( siteindex1, siteindex2, index + 1 + num, index + 1 );
                  Btensors[ index ][ cnt2 ][ cnt3 ]->zaxpy( alpha, S1tensors[ index ][ num ][ 0 ] );
                  BtensorsT[ index ][ cnt2 ][ cnt3 ]->zaxpy( alpha, S1tensorsT[ index ][ num ][ 0 ] );
               }
               alpha = 2 * prob->gMxElement( siteindex1, index + 1, siteindex2, index + 1 + num ) -
                       prob->gMxElement( siteindex1, index + 1, index + 1 + num, siteindex2 );
               Ctensors[ index ][ cnt2 ][ cnt3 ]->zaxpy( alpha, F0tensors[ index ][ num ][ 0 ] );
               CtensorsT[ index ][ cnt2 ][ cnt3 ]->zaxpy( alpha, F0tensorsT[ index ][ num ][ 0 ] );

               alpha = -prob->gMxElement( siteindex1, index + 1, index + 1 + num, siteindex2 ); // Second line for CtensorsT
               Dtensors[ index ][ cnt2 ][ cnt3 ]->zaxpy( alpha, F1tensors[ index ][ num ][ 0 ] );
               DtensorsT[ index ][ cnt2 ][ cnt3 ]->zaxpy( alpha, F1tensorsT[ index ][ num ][ 0 ] );

               if ( num > 0 ) {
                  alpha = 2 * prob->gMxElement( siteindex1, index + 1 + num, siteindex2, index + 1 ) -
                          prob->gMxElement( siteindex1, index + 1 + num, index + 1, siteindex2 );
                  Ctensors[ index ][ cnt2 ][ cnt3 ]->zaxpy_tensorCD( alpha, F0tensorsT[ index ][ num ][ 0 ] );
                  CtensorsT[ index ][ cnt2 ][ cnt3 ]->zaxpy_tensorCTDT( alpha, F0tensors[ index ][ num ][ 0 ] );

                  alpha = -prob->gMxElement( siteindex1, index + 1 + num, index + 1, siteindex2 ); // Second line for Ctensors_MPS_mpsUp
                  Dtensors[ index ][ cnt2 ][ cnt3 ]->zaxpy_tensorCD( alpha, F1tensorsT[ index ][ num ][ 0 ] );
                  DtensorsT[ index ][ cnt2 ][ cnt3 ]->zaxpy_tensorCTDT( alpha, F1tensors[ index ][ num ][ 0 ] );
               }
            }
         }
      }
// QQtensors  : certain processes own certain QQtensors  --- You don't want to
// locally parallellize when sending and receiving buffers!
#pragma omp for schedule( static ) nowait
      for ( int cnt2 = 0; cnt2 < index + 1; cnt2++ ) {
         if ( index == L - 2 ) {
            Qtensors[ index ][ cnt2 ]->clear();
            QtensorsT[ index ][ cnt2 ]->clear();
            Qtensors[ index ][ cnt2 ]->AddTermSimple( mpsUp[ index + 1 ], mpsDown[ index + 1 ], NULL, NULL );
            QtensorsT[ index ][ cnt2 ]->AddTermSimple( mpsUp[ index + 1 ], mpsDown[ index + 1 ], NULL, NULL );
         } else {
            dcomplex * workmemBIS = new dcomplex[ dimR * dimR ];
            Qtensors[ index ][ cnt2 ]->update( Qtensors[ index + 1 ][ cnt2 + 1 ], mpsUp[ index + 1 ], mpsDown[ index + 1 ], workmem );
            QtensorsT[ index ][ cnt2 ]->update( QtensorsT[ index + 1 ][ cnt2 + 1 ], mpsUp[ index + 1 ], mpsDown[ index + 1 ], workmem );
            Qtensors[ index ][ cnt2 ]->AddTermSimple( mpsUp[ index + 1 ], mpsDown[ index + 1 ], Otensors[ index + 1 ], workmem );
            QtensorsT[ index ][ cnt2 ]->AddTermSimple( mpsUp[ index + 1 ], mpsDown[ index + 1 ], Otensors[ index + 1 ], workmem );
            Qtensors[ index ][ cnt2 ]->AddTermsL( Ltensors[ index + 1 ], LtensorsT[ index + 1 ], mpsUp[ index + 1 ], mpsDown[ index + 1 ], workmemBIS, workmem );
            QtensorsT[ index ][ cnt2 ]->AddTermsL( Ltensors[ index + 1 ], LtensorsT[ index + 1 ], mpsUp[ index + 1 ], mpsDown[ index + 1 ], workmemBIS, workmem );
            Qtensors[ index ][ cnt2 ]->AddTermsAB( Atensors[ index + 1 ][ cnt2 + 1 ][ 0 ], Btensors[ index + 1 ][ cnt2 + 1 ][ 0 ], mpsUp[ index + 1 ], mpsDown[ index + 1 ], workmemBIS, workmem );
            QtensorsT[ index ][ cnt2 ]->AddTermsAB( AtensorsT[ index + 1 ][ cnt2 + 1 ][ 0 ], BtensorsT[ index + 1 ][ cnt2 + 1 ][ 0 ], mpsUp[ index + 1 ], mpsDown[ index + 1 ], workmemBIS, workmem );
            Qtensors[ index ][ cnt2 ]->AddTermsCD( Ctensors[ index + 1 ][ cnt2 + 1 ][ 0 ], Dtensors[ index + 1 ][ cnt2 + 1 ][ 0 ], mpsUp[ index + 1 ], mpsDown[ index + 1 ], workmemBIS, workmem );
            QtensorsT[ index ][ cnt2 ]->AddTermsCD( CtensorsT[ index + 1 ][ cnt2 + 1 ][ 0 ], DtensorsT[ index + 1 ][ cnt2 + 1 ][ 0 ], mpsUp[ index + 1 ], mpsDown[ index + 1 ], workmemBIS, workmem );
            delete[] workmemBIS;
         }
      }

      delete[] workmem;
   }
   // Xtensors
   if ( index == L - 2 ) {
      Xtensors[ index ]->update( mpsUp[ index + 1 ], mpsDown[ index + 1 ] );
   } else {
      Xtensors[ index ]->update( mpsUp[ index + 1 ], mpsDown[ index + 1 ], Otensors[ index + 1 ],
                                 Ltensors[ index + 1 ], LtensorsT[ index + 1 ], Xtensors[ index + 1 ],
                                 Qtensors[ index + 1 ][ 0 ], QtensorsT[ index + 1 ][ 0 ],
                                 Atensors[ index + 1 ][ 0 ][ 0 ], AtensorsT[ index + 1 ][ 0 ][ 0 ],
                                 CtensorsT[ index + 1 ][ 0 ][ 0 ], DtensorsT[ index + 1 ][ 0 ][ 0 ] );
   }

   // Otensors
   if ( index == L - 2 ) {
      Otensors[ index ]->create( mpsUp[ index + 1 ], mpsDown[ index + 1 ] );
   } else {
      Otensors[ index ]->update_ownmem( mpsUp[ index + 1 ], mpsDown[ index + 1 ], Otensors[ index + 1 ] );
   }
}

void CheMPS2::HamiltonianOperator::updateMovingRight( const int index, CTensorT ** mpsUp, SyBookkeeper * bkUp, CTensorT ** mpsDown, SyBookkeeper * bkDown ) {

   const int dimL = std::max( bkUp->gMaxDimAtBound( index ), bkDown->gMaxDimAtBound( index ) );
   const int dimR = std::max( bkUp->gMaxDimAtBound( index + 1 ), bkDown->gMaxDimAtBound( index + 1 ) );

#pragma omp parallel
   {
      dcomplex * workmem = new dcomplex[ dimL * dimR ];

// Ltensors
#pragma omp for schedule( static ) nowait
      for ( int cnt2 = 0; cnt2 < index + 1; cnt2++ ) {
         if ( cnt2 == 0 ) {
            if ( index == 0 ) {
               Ltensors[ index ][ cnt2 ]->create( mpsUp[ index ], mpsDown[ index ], NULL, NULL );
               LtensorsT[ index ][ cnt2 ]->create( mpsUp[ index ], mpsDown[ index ], NULL, NULL );
            } else {
               Ltensors[ index ][ cnt2 ]->create( mpsUp[ index ], mpsDown[ index ], Otensors[ index - 1 ], workmem );
               LtensorsT[ index ][ cnt2 ]->create( mpsUp[ index ], mpsDown[ index ], Otensors[ index - 1 ], workmem );
            }
         } else {
            Ltensors[ index ][ cnt2 ]->update( Ltensors[ index - 1 ][ cnt2 - 1 ], mpsUp[ index ], mpsDown[ index ], workmem );
            LtensorsT[ index ][ cnt2 ]->update( LtensorsT[ index - 1 ][ cnt2 - 1 ], mpsUp[ index ], mpsDown[ index ], workmem );
         }
      }

      // Two-operator tensors : certain processes own certain two-operator tensors
      const int k1          = index + 1;
      const int upperbound1 = ( k1 * ( k1 + 1 ) ) / 2;
      int result[ 2 ];
// After this parallel region, WAIT because F0,F1,S0,S1[ index ][ cnt2 ][ cnt3
// == 0 ] is required for the complementary operators
#pragma omp for schedule( static )
      for ( int global = 0; global < upperbound1; global++ ) {
         Special::invert_triangle_two( global, result );
         const int cnt2 = index - result[ 1 ];
         const int cnt3 = result[ 0 ];
         if ( cnt3 == 0 ) { // Every MPI process owns the Operator[ index ][ cnt2 ][
            // cnt3 == 0 ]
            if ( cnt2 == 0 ) {
               if ( index == 0 ) {
                  F0tensors[ index ][ cnt2 ][ cnt3 ]->makenew( mpsUp[ index ], mpsDown[ index ], NULL, NULL );
                  F1tensors[ index ][ cnt2 ][ cnt3 ]->makenew( mpsUp[ index ], mpsDown[ index ], NULL, NULL );
                  S0tensors[ index ][ cnt2 ][ cnt3 ]->makenew( mpsUp[ index ], mpsDown[ index ], NULL, NULL );
                  F0tensorsT[ index ][ cnt2 ][ cnt3 ]->makenew( mpsUp[ index ], mpsDown[ index ], NULL, NULL );
                  F1tensorsT[ index ][ cnt2 ][ cnt3 ]->makenew( mpsUp[ index ], mpsDown[ index ], NULL, NULL );
                  S0tensorsT[ index ][ cnt2 ][ cnt3 ]->makenew( mpsUp[ index ], mpsDown[ index ], NULL, NULL );
               } else {
                  F0tensors[ index ][ cnt2 ][ cnt3 ]->makenew( mpsUp[ index ], mpsDown[ index ], Otensors[ index - 1 ], workmem );
                  F1tensors[ index ][ cnt2 ][ cnt3 ]->makenew( mpsUp[ index ], mpsDown[ index ], Otensors[ index - 1 ], workmem );
                  S0tensors[ index ][ cnt2 ][ cnt3 ]->makenew( mpsUp[ index ], mpsDown[ index ], Otensors[ index - 1 ], workmem );
                  F0tensorsT[ index ][ cnt2 ][ cnt3 ]->makenew( mpsUp[ index ], mpsDown[ index ], Otensors[ index - 1 ], workmem );
                  F1tensorsT[ index ][ cnt2 ][ cnt3 ]->makenew( mpsUp[ index ], mpsDown[ index ], Otensors[ index - 1 ], workmem );
                  S0tensorsT[ index ][ cnt2 ][ cnt3 ]->makenew( mpsUp[ index ], mpsDown[ index ], Otensors[ index - 1 ], workmem );
               }
               // // S1[ index ][ 0 ][ cnt3 ] doesn't exist
            } else {
               F0tensors[ index ][ cnt2 ][ cnt3 ]->makenew( Ltensors[ index - 1 ][ cnt2 - 1 ], mpsUp[ index ], mpsDown[ index ], workmem );
               F1tensors[ index ][ cnt2 ][ cnt3 ]->makenew( Ltensors[ index - 1 ][ cnt2 - 1 ], mpsUp[ index ], mpsDown[ index ], workmem );
               S0tensors[ index ][ cnt2 ][ cnt3 ]->makenew( Ltensors[ index - 1 ][ cnt2 - 1 ], mpsUp[ index ], mpsDown[ index ], workmem );
               S1tensors[ index ][ cnt2 ][ cnt3 ]->makenew( Ltensors[ index - 1 ][ cnt2 - 1 ], mpsUp[ index ], mpsDown[ index ], workmem );
               F0tensorsT[ index ][ cnt2 ][ cnt3 ]->makenew( LtensorsT[ index - 1 ][ cnt2 - 1 ], mpsUp[ index ], mpsDown[ index ], workmem );
               F1tensorsT[ index ][ cnt2 ][ cnt3 ]->makenew( LtensorsT[ index - 1 ][ cnt2 - 1 ], mpsUp[ index ], mpsDown[ index ], workmem );
               S0tensorsT[ index ][ cnt2 ][ cnt3 ]->makenew( LtensorsT[ index - 1 ][ cnt2 - 1 ], mpsUp[ index ], mpsDown[ index ], workmem );
               S1tensorsT[ index ][ cnt2 ][ cnt3 ]->makenew( LtensorsT[ index - 1 ][ cnt2 - 1 ], mpsUp[ index ], mpsDown[ index ], workmem );
            }
         } else {
            F0tensors[ index ][ cnt2 ][ cnt3 ]->update( F0tensors[ index - 1 ][ cnt2 ][ cnt3 - 1 ], mpsUp[ index ], mpsDown[ index ], workmem );
            F1tensors[ index ][ cnt2 ][ cnt3 ]->update( F1tensors[ index - 1 ][ cnt2 ][ cnt3 - 1 ], mpsUp[ index ], mpsDown[ index ], workmem );
            S0tensors[ index ][ cnt2 ][ cnt3 ]->update( S0tensors[ index - 1 ][ cnt2 ][ cnt3 - 1 ], mpsUp[ index ], mpsDown[ index ], workmem );
            F0tensorsT[ index ][ cnt2 ][ cnt3 ]->update( F0tensorsT[ index - 1 ][ cnt2 ][ cnt3 - 1 ], mpsUp[ index ], mpsDown[ index ], workmem );
            F1tensorsT[ index ][ cnt2 ][ cnt3 ]->update( F1tensorsT[ index - 1 ][ cnt2 ][ cnt3 - 1 ], mpsUp[ index ], mpsDown[ index ], workmem );
            S0tensorsT[ index ][ cnt2 ][ cnt3 ]->update( S0tensorsT[ index - 1 ][ cnt2 ][ cnt3 - 1 ], mpsUp[ index ], mpsDown[ index ], workmem );
            if ( cnt2 > 0 ) {
               S1tensors[ index ][ cnt2 ][ cnt3 ]->update( S1tensors[ index - 1 ][ cnt2 ][ cnt3 - 1 ], mpsUp[ index ], mpsDown[ index ], workmem );
               S1tensorsT[ index ][ cnt2 ][ cnt3 ]->update( S1tensorsT[ index - 1 ][ cnt2 ][ cnt3 - 1 ], mpsUp[ index ], mpsDown[ index ], workmem );
            }
         }
      }

      // Complementary two-operator tensors : certain processes own certain
      // complementary two-operator tensors
      const int k2          = L - 1 - index;
      const int upperbound2 = ( k2 * ( k2 + 1 ) ) / 2;
#pragma omp for schedule( static ) nowait
      for ( int global = 0; global < upperbound2; global++ ) {
         Special::invert_triangle_two( global, result );
         const int cnt2       = k2 - 1 - result[ 1 ];
         const int cnt3       = result[ 0 ];
         const int siteindex1 = index + 1 + cnt3;
         const int siteindex2 = index + 1 + cnt2 + cnt3;
         const int irrep_prod = CheMPS2::Irreps::directProd( bkUp->gIrrep( siteindex1 ), bkUp->gIrrep( siteindex2 ) );
         if ( index == 0 ) {
            Atensors[ index ][ cnt2 ][ cnt3 ]->clear();
            AtensorsT[ index ][ cnt2 ][ cnt3 ]->clear();
            if ( cnt2 > 0 ) {
               Btensors[ index ][ cnt2 ][ cnt3 ]->clear();
               BtensorsT[ index ][ cnt2 ][ cnt3 ]->clear();
            }
            Ctensors[ index ][ cnt2 ][ cnt3 ]->clear();
            CtensorsT[ index ][ cnt2 ][ cnt3 ]->clear();
            Dtensors[ index ][ cnt2 ][ cnt3 ]->clear();
            DtensorsT[ index ][ cnt2 ][ cnt3 ]->clear();
         } else {
            Atensors[ index ][ cnt2 ][ cnt3 ]->update( Atensors[ index - 1 ][ cnt2 ][ cnt3 + 1 ], mpsUp[ index ], mpsDown[ index ], workmem );
            AtensorsT[ index ][ cnt2 ][ cnt3 ]->update( AtensorsT[ index - 1 ][ cnt2 ][ cnt3 + 1 ], mpsUp[ index ], mpsDown[ index ], workmem );
            if ( cnt2 > 0 ) {
               Btensors[ index ][ cnt2 ][ cnt3 ]->update( Btensors[ index - 1 ][ cnt2 ][ cnt3 + 1 ], mpsUp[ index ], mpsDown[ index ], workmem );
               BtensorsT[ index ][ cnt2 ][ cnt3 ]->update( BtensorsT[ index - 1 ][ cnt2 ][ cnt3 + 1 ], mpsUp[ index ], mpsDown[ index ], workmem );
            }
            Ctensors[ index ][ cnt2 ][ cnt3 ]->update( Ctensors[ index - 1 ][ cnt2 ][ cnt3 + 1 ], mpsUp[ index ], mpsDown[ index ], workmem );
            CtensorsT[ index ][ cnt2 ][ cnt3 ]->update( CtensorsT[ index - 1 ][ cnt2 ][ cnt3 + 1 ], mpsUp[ index ], mpsDown[ index ], workmem );
            Dtensors[ index ][ cnt2 ][ cnt3 ]->update( Dtensors[ index - 1 ][ cnt2 ][ cnt3 + 1 ], mpsUp[ index ], mpsDown[ index ], workmem );
            DtensorsT[ index ][ cnt2 ][ cnt3 ]->update( DtensorsT[ index - 1 ][ cnt2 ][ cnt3 + 1 ], mpsUp[ index ], mpsDown[ index ], workmem );
         }

         for ( int num = 0; num < index + 1; num++ ) {
            if ( irrep_prod == S0tensorsT[ index ][ num ][ 0 ]->get_irrep() ) { // Then the matrix elements are not 0 due to symm.
               double alpha = prob->gMxElement( index - num, index, siteindex1, siteindex2 );
               if ( ( cnt2 == 0 ) && ( num == 0 ) ) {
                  alpha *= 0.5;
               }
               if ( ( cnt2 > 0 ) && ( num > 0 ) ) {
                  alpha += prob->gMxElement( index - num, index, siteindex2, siteindex1 );
               }
               Atensors[ index ][ cnt2 ][ cnt3 ]->zaxpy( alpha, S0tensors[ index ][ num ][ 0 ] );
               AtensorsT[ index ][ cnt2 ][ cnt3 ]->zaxpy( alpha, S0tensorsT[ index ][ num ][ 0 ] );

               if ( ( num > 0 ) && ( cnt2 > 0 ) ) {
                  alpha =
                      prob->gMxElement( index - num, index, siteindex1, siteindex2 ) -
                      prob->gMxElement( index - num, index, siteindex2, siteindex1 );
                  Btensors[ index ][ cnt2 ][ cnt3 ]->zaxpy( alpha, S1tensors[ index ][ num ][ 0 ] );
                  BtensorsT[ index ][ cnt2 ][ cnt3 ]->zaxpy( alpha, S1tensorsT[ index ][ num ][ 0 ] );
               }

               alpha = 2 * prob->gMxElement( index - num, siteindex1, index, siteindex2 ) -
                       prob->gMxElement( index - num, siteindex1, siteindex2, index );
               Ctensors[ index ][ cnt2 ][ cnt3 ]->zaxpy( alpha, F0tensors[ index ][ num ][ 0 ] );
               CtensorsT[ index ][ cnt2 ][ cnt3 ]->zaxpy( alpha, F0tensorsT[ index ][ num ][ 0 ] );

               alpha = -prob->gMxElement( index - num, siteindex1, siteindex2, index ); // Second line for CtensorsT
               Dtensors[ index ][ cnt2 ][ cnt3 ]->zaxpy( alpha, F1tensors[ index ][ num ][ 0 ] );
               DtensorsT[ index ][ cnt2 ][ cnt3 ]->zaxpy( alpha, F1tensorsT[ index ][ num ][ 0 ] );
               if ( num > 0 ) {
                  alpha = 2 * prob->gMxElement( index - num, siteindex2, index, siteindex1 ) -
                          prob->gMxElement( index - num, siteindex2, siteindex1, index );

                  Ctensors[ index ][ cnt2 ][ cnt3 ]->zaxpy_tensorCD( alpha, F0tensorsT[ index ][ num ][ 0 ] );
                  CtensorsT[ index ][ cnt2 ][ cnt3 ]->zaxpy_tensorCTDT( alpha, F0tensors[ index ][ num ][ 0 ] );

                  alpha = -prob->gMxElement( index - num, siteindex2, siteindex1, index ); // Second line for CtensorsT
                  Dtensors[ index ][ cnt2 ][ cnt3 ]->zaxpy_tensorCD( alpha, F1tensorsT[ index ][ num ][ 0 ] );
                  DtensorsT[ index ][ cnt2 ][ cnt3 ]->zaxpy_tensorCTDT( alpha, F1tensors[ index ][ num ][ 0 ] );
               }
            }
         }
      }

// QQtensors_mpsUp_MPS : certain processes own certain QQtensors_mpsUp_MPS ---
// You don't want to locally parallellize when sending and receiving buffers!
#pragma omp for schedule( static ) nowait
      for ( int cnt2 = 0; cnt2 < L - 1 - index; cnt2++ ) {
         if ( index == 0 ) {
            Qtensors[ index ][ cnt2 ]->clear();
            QtensorsT[ index ][ cnt2 ]->clear();
            Qtensors[ index ][ cnt2 ]->AddTermSimple( mpsUp[ index ], mpsDown[ index ], NULL, NULL );
            QtensorsT[ index ][ cnt2 ]->AddTermSimple( mpsUp[ index ], mpsDown[ index ], NULL, NULL );
         } else {
            dcomplex * workmemBIS = new dcomplex[ dimL * dimL ];
            Qtensors[ index ][ cnt2 ]->update( Qtensors[ index - 1 ][ cnt2 + 1 ], mpsUp[ index ], mpsDown[ index ], workmem );
            QtensorsT[ index ][ cnt2 ]->update( QtensorsT[ index - 1 ][ cnt2 + 1 ], mpsUp[ index ], mpsDown[ index ], workmem );
            Qtensors[ index ][ cnt2 ]->AddTermSimple( mpsUp[ index ], mpsDown[ index ], Otensors[ index - 1 ], workmem );
            QtensorsT[ index ][ cnt2 ]->AddTermSimple( mpsUp[ index ], mpsDown[ index ], Otensors[ index - 1 ], workmem );
            Qtensors[ index ][ cnt2 ]->AddTermsL( Ltensors[ index - 1 ], LtensorsT[ index - 1 ], mpsUp[ index ], mpsDown[ index ], workmemBIS, workmem );
            QtensorsT[ index ][ cnt2 ]->AddTermsL( Ltensors[ index - 1 ], LtensorsT[ index - 1 ], mpsUp[ index ], mpsDown[ index ], workmemBIS, workmem );
            Qtensors[ index ][ cnt2 ]->AddTermsAB( Atensors[ index - 1 ][ cnt2 + 1 ][ 0 ], Btensors[ index - 1 ][ cnt2 + 1 ][ 0 ], mpsUp[ index ], mpsDown[ index ], workmemBIS, workmem );
            QtensorsT[ index ][ cnt2 ]->AddTermsAB( AtensorsT[ index - 1 ][ cnt2 + 1 ][ 0 ], BtensorsT[ index - 1 ][ cnt2 + 1 ][ 0 ], mpsUp[ index ], mpsDown[ index ], workmemBIS, workmem );
            Qtensors[ index ][ cnt2 ]->AddTermsCD( Ctensors[ index - 1 ][ cnt2 + 1 ][ 0 ], Dtensors[ index - 1 ][ cnt2 + 1 ][ 0 ], mpsUp[ index ], mpsDown[ index ], workmemBIS, workmem );
            QtensorsT[ index ][ cnt2 ]->AddTermsCD( CtensorsT[ index - 1 ][ cnt2 + 1 ][ 0 ], DtensorsT[ index - 1 ][ cnt2 + 1 ][ 0 ], mpsUp[ index ], mpsDown[ index ], workmemBIS, workmem );
            delete[] workmemBIS;
         }
      }

      delete[] workmem;
   }

   // Xtensors
   if ( index == 0 ) {
      Xtensors[ index ]->update( mpsUp[ index ], mpsDown[ index ] );
   } else {
      Xtensors[ index ]->update( mpsUp[ index ], mpsDown[ index ], Otensors[ index - 1 ], Ltensors[ index - 1 ],
                                 LtensorsT[ index - 1 ], Xtensors[ index - 1 ], Qtensors[ index - 1 ][ 0 ],
                                 QtensorsT[ index - 1 ][ 0 ], Atensors[ index - 1 ][ 0 ][ 0 ],
                                 AtensorsT[ index - 1 ][ 0 ][ 0 ], CtensorsT[ index - 1 ][ 0 ][ 0 ],
                                 DtensorsT[ index - 1 ][ 0 ][ 0 ] );
   }

   // Otensors
   if ( index == 0 ) {
      Otensors[ index ]->create( mpsUp[ index ], mpsDown[ index ] );
   } else {
      Otensors[ index ]->update_ownmem( mpsUp[ index ], mpsDown[ index ], Otensors[ index - 1 ] );
   }
}

void CheMPS2::HamiltonianOperator::allocateTensors( const int index, const bool movingRight, SyBookkeeper * bkUp, SyBookkeeper * bkDown ) {

   if ( movingRight ) {
      // Ltensors
      Ltensors[ index ]  = new CTensorL *[ index + 1 ];
      LtensorsT[ index ] = new CTensorLT *[ index + 1 ];
      for ( int cnt2 = 0; cnt2 < index + 1; cnt2++ ) {
         Ltensors[ index ][ cnt2 ]  = new CTensorL( index + 1, bkUp->gIrrep( index - cnt2 ), movingRight, bkUp, bkDown );
         LtensorsT[ index ][ cnt2 ] = new CTensorLT( index + 1, bkUp->gIrrep( index - cnt2 ), movingRight, bkUp, bkDown );
      }

      // Two-operator tensors : certain processes own certain two-operator tensors
      // To right: F0tens[ cnt][ cnt2 ][ cnt3 ] = operators on sites cnt-cnt3-cnt2
      // and cnt-cnt3; at boundary cnt+1
      F0tensors[ index ]  = new CTensorF0 **[ index + 1 ];
      F0tensorsT[ index ] = new CTensorF0T **[ index + 1 ];
      F1tensors[ index ]  = new CTensorF1 **[ index + 1 ];
      F1tensorsT[ index ] = new CTensorF1T **[ index + 1 ];
      S0tensors[ index ]  = new CTensorS0 **[ index + 1 ];
      S0tensorsT[ index ] = new CTensorS0T **[ index + 1 ];
      S1tensors[ index ]  = new CTensorS1 **[ index + 1 ];
      S1tensorsT[ index ] = new CTensorS1T **[ index + 1 ];
      for ( int cnt2 = 0; cnt2 < index + 1; cnt2++ ) {
         F0tensors[ index ][ cnt2 ]  = new CTensorF0 *[ index - cnt2 + 1 ];
         F0tensorsT[ index ][ cnt2 ] = new CTensorF0T *[ index - cnt2 + 1 ];
         F1tensors[ index ][ cnt2 ]  = new CTensorF1 *[ index - cnt2 + 1 ];
         F1tensorsT[ index ][ cnt2 ] = new CTensorF1T *[ index - cnt2 + 1 ];
         S0tensors[ index ][ cnt2 ]  = new CTensorS0 *[ index - cnt2 + 1 ];
         S0tensorsT[ index ][ cnt2 ] = new CTensorS0T *[ index - cnt2 + 1 ];
         if ( cnt2 > 0 ) {
            S1tensors[ index ][ cnt2 ] = new CTensorS1 *[ index - cnt2 + 1 ];
         }
         if ( cnt2 > 0 ) {
            S1tensorsT[ index ][ cnt2 ] = new CTensorS1T *[ index - cnt2 + 1 ];
         }
         for ( int cnt3 = 0; cnt3 < index - cnt2 + 1; cnt3++ ) {
            const int Iprod = CheMPS2::Irreps::directProd( bkUp->gIrrep( index - cnt2 - cnt3 ), bkUp->gIrrep( index - cnt3 ) );

            F0tensors[ index ][ cnt2 ][ cnt3 ]  = new CTensorF0( index + 1, Iprod, movingRight, bkUp, bkDown );
            F0tensorsT[ index ][ cnt2 ][ cnt3 ] = new CTensorF0T( index + 1, Iprod, movingRight, bkUp, bkDown );
            F1tensors[ index ][ cnt2 ][ cnt3 ]  = new CTensorF1( index + 1, Iprod, movingRight, bkUp, bkDown );
            F1tensorsT[ index ][ cnt2 ][ cnt3 ] = new CTensorF1T( index + 1, Iprod, movingRight, bkUp, bkDown );
            S0tensors[ index ][ cnt2 ][ cnt3 ]  = new CTensorS0( index + 1, Iprod, movingRight, bkUp, bkDown );
            S0tensorsT[ index ][ cnt2 ][ cnt3 ] = new CTensorS0T( index + 1, Iprod, movingRight, bkUp, bkDown );
            if ( cnt2 > 0 ) {
               S1tensors[ index ][ cnt2 ][ cnt3 ]  = new CTensorS1( index + 1, Iprod, movingRight, bkUp, bkDown );
               S1tensorsT[ index ][ cnt2 ][ cnt3 ] = new CTensorS1T( index + 1, Iprod, movingRight, bkUp, bkDown );
            }
         }
      }

      // Complementary two-operator tensors : certain processes own certain
      // complementary two-operator tensors
      // To right: Atens[ cnt][ cnt2 ][ cnt3 ] = operators on sites cnt+1+cnt3 and
      // cnt+1+cnt2+cnt3; at boundary cnt+1
      Atensors[ index ]  = new CTensorOperator **[ L - 1 - index ];
      AtensorsT[ index ] = new CTensorOperator **[ L - 1 - index ];
      Btensors[ index ]  = new CTensorOperator **[ L - 1 - index ];
      BtensorsT[ index ] = new CTensorOperator **[ L - 1 - index ];
      Ctensors[ index ]  = new CTensorOperator **[ L - 1 - index ];
      CtensorsT[ index ] = new CTensorOperator **[ L - 1 - index ];
      Dtensors[ index ]  = new CTensorOperator **[ L - 1 - index ];
      DtensorsT[ index ] = new CTensorOperator **[ L - 1 - index ];
      for ( int cnt2 = 0; cnt2 < L - 1 - index; cnt2++ ) {
         Atensors[ index ][ cnt2 ]  = new CTensorOperator *[ L - 1 - index - cnt2 ];
         AtensorsT[ index ][ cnt2 ] = new CTensorOperator *[ L - 1 - index - cnt2 ];
         if ( cnt2 > 0 ) {
            Btensors[ index ][ cnt2 ]  = new CTensorOperator *[ L - 1 - index - cnt2 ];
            BtensorsT[ index ][ cnt2 ] = new CTensorOperator *[ L - 1 - index - cnt2 ];
         }
         Ctensors[ index ][ cnt2 ]  = new CTensorOperator *[ L - 1 - index - cnt2 ];
         CtensorsT[ index ][ cnt2 ] = new CTensorOperator *[ L - 1 - index - cnt2 ];
         Dtensors[ index ][ cnt2 ]  = new CTensorOperator *[ L - 1 - index - cnt2 ];
         DtensorsT[ index ][ cnt2 ] = new CTensorOperator *[ L - 1 - index - cnt2 ];
         for ( int cnt3 = 0; cnt3 < L - 1 - index - cnt2; cnt3++ ) {
            const int Idiff = CheMPS2::Irreps::directProd( bkUp->gIrrep( index + 1 + cnt2 + cnt3 ), bkUp->gIrrep( index + 1 + cnt3 ) );

            Atensors[ index ][ cnt2 ][ cnt3 ]  = new CTensorOperator( index + 1, 0, 2, Idiff, movingRight, true, false, bkUp, bkDown );
            AtensorsT[ index ][ cnt2 ][ cnt3 ] = new CTensorOperator( index + 1, 0, -2, Idiff, movingRight, false, false, bkUp, bkDown );
            if ( cnt2 > 0 ) {
               Btensors[ index ][ cnt2 ][ cnt3 ]  = new CTensorOperator( index + 1, 2, 2, Idiff, movingRight, true, false, bkUp, bkDown );
               BtensorsT[ index ][ cnt2 ][ cnt3 ] = new CTensorOperator( index + 1, 2, -2, Idiff, movingRight, false, false, bkUp, bkDown );
            }
            Ctensors[ index ][ cnt2 ][ cnt3 ]  = new CTensorOperator( index + 1, 0, 0, Idiff, movingRight, true, false, bkUp, bkDown );
            CtensorsT[ index ][ cnt2 ][ cnt3 ] = new CTensorOperator( index + 1, 0, 0, Idiff, movingRight, false, false, bkUp, bkDown );
            Dtensors[ index ][ cnt2 ][ cnt3 ]  = new CTensorOperator( index + 1, 2, 0, Idiff, movingRight, movingRight, false, bkUp, bkDown );
            DtensorsT[ index ][ cnt2 ][ cnt3 ] = new CTensorOperator( index + 1, 2, 0, Idiff, movingRight, !movingRight, false, bkUp, bkDown );
         }
      }

      // QQtensors_MPSDT_MPS
      // To right: Qtens[ cnt][ cnt2 ] = operator on site cnt+1+cnt2; at boundary cnt+1
      Qtensors[ index ]  = new CTensorQ *[ L - 1 - index ];
      QtensorsT[ index ] = new CTensorQT *[ L - 1 - index ];
      for ( int cnt2 = 0; cnt2 < L - 1 - index; cnt2++ ) {
         Qtensors[ index ][ cnt2 ]  = new CTensorQ( index + 1, bkUp->gIrrep( index + 1 + cnt2 ), movingRight, bkUp, bkDown, prob, index + 1 + cnt2 );
         QtensorsT[ index ][ cnt2 ] = new CTensorQT( index + 1, bkUp->gIrrep( index + 1 + cnt2 ), movingRight, bkUp, bkDown, prob, index + 1 + cnt2 );
      }

      // Xtensors : a certain process owns the Xtensors
      Xtensors[ index ] = new CTensorX( index + 1, movingRight, bkUp, bkDown, prob );

      // Otensors :
      Otensors[ index ] = new CTensorO( index + 1, movingRight, bkUp, bkDown );
   } else {
      Ltensors[ index ]  = new CTensorL *[ L - 1 - index ];
      LtensorsT[ index ] = new CTensorLT *[ L - 1 - index ];
      for ( int cnt2 = 0; cnt2 < L - 1 - index; cnt2++ ) {
         Ltensors[ index ][ cnt2 ]  = new CTensorL( index + 1, bkUp->gIrrep( index + 1 + cnt2 ), movingRight, bkUp, bkDown );
         LtensorsT[ index ][ cnt2 ] = new CTensorLT( index + 1, bkUp->gIrrep( index + 1 + cnt2 ), movingRight, bkUp, bkDown );
      }

      // Two-operator tensors : certain processes own certain two-operator tensors
      // To left: F0tens[ cnt][ cnt2 ][ cnt3 ] = operators on sites cnt+1+cnt3 and
      // cnt+1+cnt3+cnt2; at boundary cnt+1
      F0tensors[ index ]  = new CTensorF0 **[ L - 1 - index ];
      F0tensorsT[ index ] = new CTensorF0T **[ L - 1 - index ];
      F1tensors[ index ]  = new CTensorF1 **[ L - 1 - index ];
      F1tensorsT[ index ] = new CTensorF1T **[ L - 1 - index ];
      S0tensors[ index ]  = new CTensorS0 **[ L - 1 - index ];
      S0tensorsT[ index ] = new CTensorS0T **[ L - 1 - index ];
      S1tensors[ index ]  = new CTensorS1 **[ L - 1 - index ];
      S1tensorsT[ index ] = new CTensorS1T **[ L - 1 - index ];
      for ( int cnt2 = 0; cnt2 < L - 1 - index; cnt2++ ) {
         F0tensors[ index ][ cnt2 ]  = new CTensorF0 *[ L - 1 - index - cnt2 ];
         F0tensorsT[ index ][ cnt2 ] = new CTensorF0T *[ L - 1 - index - cnt2 ];
         F1tensors[ index ][ cnt2 ]  = new CTensorF1 *[ L - 1 - index - cnt2 ];
         F1tensorsT[ index ][ cnt2 ] = new CTensorF1T *[ L - 1 - index - cnt2 ];
         S0tensors[ index ][ cnt2 ]  = new CTensorS0 *[ L - 1 - index - cnt2 ];
         S0tensorsT[ index ][ cnt2 ] = new CTensorS0T *[ L - 1 - index - cnt2 ];
         if ( cnt2 > 0 ) {
            S1tensors[ index ][ cnt2 ]  = new CTensorS1 *[ L - 1 - index - cnt2 ];
            S1tensorsT[ index ][ cnt2 ] = new CTensorS1T *[ L - 1 - index - cnt2 ];
         }
         for ( int cnt3 = 0; cnt3 < L - 1 - index - cnt2; cnt3++ ) {
            const int Iprod = Irreps::directProd( bkUp->gIrrep( index + 1 + cnt3 ), bkUp->gIrrep( index + 1 + cnt2 + cnt3 ) );

            F0tensors[ index ][ cnt2 ][ cnt3 ]  = new CTensorF0( index + 1, Iprod, movingRight, bkUp, bkDown );
            F0tensorsT[ index ][ cnt2 ][ cnt3 ] = new CTensorF0T( index + 1, Iprod, movingRight, bkUp, bkDown );
            F1tensors[ index ][ cnt2 ][ cnt3 ]  = new CTensorF1( index + 1, Iprod, movingRight, bkUp, bkDown );
            F1tensorsT[ index ][ cnt2 ][ cnt3 ] = new CTensorF1T( index + 1, Iprod, movingRight, bkUp, bkDown );
            S0tensors[ index ][ cnt2 ][ cnt3 ]  = new CTensorS0( index + 1, Iprod, movingRight, bkUp, bkDown );
            S0tensorsT[ index ][ cnt2 ][ cnt3 ] = new CTensorS0T( index + 1, Iprod, movingRight, bkUp, bkDown );
            if ( cnt2 > 0 ) {
               S1tensors[ index ][ cnt2 ][ cnt3 ]  = new CTensorS1( index + 1, Iprod, movingRight, bkUp, bkDown );
               S1tensorsT[ index ][ cnt2 ][ cnt3 ] = new CTensorS1T( index + 1, Iprod, movingRight, bkUp, bkDown );
            }
         }
      }

      // Complementary two-operator tensors : certain processes own certain
      // complementary two-operator tensors
      // To left: Atens[ cnt][ cnt2 ][ cnt3 ] = operators on sites cnt-cnt2-cnt3
      // and cnt-cnt3; at boundary cnt+1
      Atensors[ index ]  = new CTensorOperator **[ index + 1 ];
      AtensorsT[ index ] = new CTensorOperator **[ index + 1 ];
      Btensors[ index ]  = new CTensorOperator **[ index + 1 ];
      BtensorsT[ index ] = new CTensorOperator **[ index + 1 ];
      Ctensors[ index ]  = new CTensorOperator **[ index + 1 ];
      CtensorsT[ index ] = new CTensorOperator **[ index + 1 ];
      Dtensors[ index ]  = new CTensorOperator **[ index + 1 ];
      DtensorsT[ index ] = new CTensorOperator **[ index + 1 ];
      for ( int cnt2 = 0; cnt2 < index + 1; cnt2++ ) {
         Atensors[ index ][ cnt2 ]  = new CTensorOperator *[ index + 1 - cnt2 ];
         AtensorsT[ index ][ cnt2 ] = new CTensorOperator *[ index + 1 - cnt2 ];
         if ( cnt2 > 0 ) {
            Btensors[ index ][ cnt2 ]  = new CTensorOperator *[ index + 1 - cnt2 ];
            BtensorsT[ index ][ cnt2 ] = new CTensorOperator *[ index + 1 - cnt2 ];
         }
         Ctensors[ index ][ cnt2 ]  = new CTensorOperator *[ index + 1 - cnt2 ];
         CtensorsT[ index ][ cnt2 ] = new CTensorOperator *[ index + 1 - cnt2 ];
         Dtensors[ index ][ cnt2 ]  = new CTensorOperator *[ index + 1 - cnt2 ];
         DtensorsT[ index ][ cnt2 ] = new CTensorOperator *[ index + 1 - cnt2 ];
         for ( int cnt3 = 0; cnt3 < index + 1 - cnt2; cnt3++ ) {
            const int Idiff = CheMPS2::Irreps::directProd( bkUp->gIrrep( index - cnt2 - cnt3 ), bkUp->gIrrep( index - cnt3 ) );

            Atensors[ index ][ cnt2 ][ cnt3 ]  = new CTensorOperator( index + 1, 0, 2, Idiff, movingRight, true, false, bkUp, bkDown );
            AtensorsT[ index ][ cnt2 ][ cnt3 ] = new CTensorOperator( index + 1, 0, -2, Idiff, movingRight, false, false, bkUp, bkDown );
            if ( cnt2 > 0 ) {
               Btensors[ index ][ cnt2 ][ cnt3 ]  = new CTensorOperator( index + 1, 2, 2, Idiff, movingRight, true, false, bkUp, bkDown );
               BtensorsT[ index ][ cnt2 ][ cnt3 ] = new CTensorOperator( index + 1, 2, -2, Idiff, movingRight, false, false, bkUp, bkDown );
            }
            Ctensors[ index ][ cnt2 ][ cnt3 ]  = new CTensorOperator( index + 1, 0, 0, Idiff, movingRight, true, false, bkUp, bkDown );
            CtensorsT[ index ][ cnt2 ][ cnt3 ] = new CTensorOperator( index + 1, 0, 0, Idiff, movingRight, false, false, bkUp, bkDown );
            Dtensors[ index ][ cnt2 ][ cnt3 ]  = new CTensorOperator( index + 1, 2, 0, Idiff, movingRight, movingRight, false, bkUp, bkDown );
            DtensorsT[ index ][ cnt2 ][ cnt3 ] = new CTensorOperator( index + 1, 2, 0, Idiff, movingRight, !movingRight, false, bkUp, bkDown );
         }
      }

      // QQtensors  : certain processes own certain QQtensors
      // To left: Qtens[ cnt][ cnt2 ] = operator on site cnt-cnt2; at boundary
      // cnt+1
      Qtensors[ index ]  = new CTensorQ *[ index + 1 ];
      QtensorsT[ index ] = new CTensorQT *[ index + 1 ];
      for ( int cnt2 = 0; cnt2 < index + 1; cnt2++ ) {
         Qtensors[ index ][ cnt2 ]  = new CTensorQ( index + 1, bkUp->gIrrep( index - cnt2 ), movingRight, bkUp, bkDown, prob, index - cnt2 );
         QtensorsT[ index ][ cnt2 ] = new CTensorQT( index + 1, bkUp->gIrrep( index - cnt2 ), movingRight, bkUp, bkDown, prob, index - cnt2 );
      }

      // Xtensors : a certain process owns the Xtensors
      Xtensors[ index ] = new CTensorX( index + 1, movingRight, bkUp, bkDown, prob );

      // Otensors :
      Otensors[ index ] = new CTensorO( index + 1, movingRight, bkUp, bkDown );
   }
}

void CheMPS2::HamiltonianOperator::deleteTensors( const int index, const bool movingRight ) {
   const int Nbound = movingRight ? index + 1 : L - 1 - index;
   const int Cbound = movingRight ? L - 1 - index : index + 1;

   // Ltensors  : all processes own all Ltensors_MPSDT_MPS
   for ( int cnt2 = 0; cnt2 < Nbound; cnt2++ ) {
      delete Ltensors[ index ][ cnt2 ];
      delete LtensorsT[ index ][ cnt2 ];
   }
   delete[] Ltensors[ index ];
   delete[] LtensorsT[ index ];

   // Two-operator tensors : certain processes own certain two-operator tensors
   for ( int cnt2 = 0; cnt2 < Nbound; cnt2++ ) {
      for ( int cnt3 = 0; cnt3 < Nbound - cnt2; cnt3++ ) {
         delete F0tensors[ index ][ cnt2 ][ cnt3 ];
         delete F0tensorsT[ index ][ cnt2 ][ cnt3 ];
         delete F1tensors[ index ][ cnt2 ][ cnt3 ];
         delete F1tensorsT[ index ][ cnt2 ][ cnt3 ];
         delete S0tensors[ index ][ cnt2 ][ cnt3 ];
         delete S0tensorsT[ index ][ cnt2 ][ cnt3 ];
         if ( cnt2 > 0 ) {
            delete S1tensors[ index ][ cnt2 ][ cnt3 ];
            delete S1tensorsT[ index ][ cnt2 ][ cnt3 ];
         }
      }
      delete[] F0tensors[ index ][ cnt2 ];
      delete[] F0tensorsT[ index ][ cnt2 ];
      delete[] F1tensors[ index ][ cnt2 ];
      delete[] F1tensorsT[ index ][ cnt2 ];
      delete[] S0tensors[ index ][ cnt2 ];
      delete[] S0tensorsT[ index ][ cnt2 ];
      if ( cnt2 > 0 ) {
         delete[] S1tensors[ index ][ cnt2 ];
         delete[] S1tensorsT[ index ][ cnt2 ];
      }
   }
   delete[] F0tensors[ index ];
   delete[] F0tensorsT[ index ];
   delete[] F1tensors[ index ];
   delete[] F1tensorsT[ index ];
   delete[] S0tensors[ index ];
   delete[] S0tensorsT[ index ];
   delete[] S1tensors[ index ];
   delete[] S1tensorsT[ index ];

   // Complementary two-operator tensors : certain processes own certain complementary two-operator tensors
   for ( int cnt2 = 0; cnt2 < Cbound; cnt2++ ) {
      for ( int cnt3 = 0; cnt3 < Cbound - cnt2; cnt3++ ) {
         delete Atensors[ index ][ cnt2 ][ cnt3 ];
         delete AtensorsT[ index ][ cnt2 ][ cnt3 ];
         if ( cnt2 > 0 ) {
            delete Btensors[ index ][ cnt2 ][ cnt3 ];
            delete BtensorsT[ index ][ cnt2 ][ cnt3 ];
         }
         delete Ctensors[ index ][ cnt2 ][ cnt3 ];
         delete CtensorsT[ index ][ cnt2 ][ cnt3 ];
         delete Dtensors[ index ][ cnt2 ][ cnt3 ];
         delete DtensorsT[ index ][ cnt2 ][ cnt3 ];
      }
      delete[] Atensors[ index ][ cnt2 ];
      delete[] AtensorsT[ index ][ cnt2 ];
      if ( cnt2 > 0 ) {
         delete[] Btensors[ index ][ cnt2 ];
         delete[] BtensorsT[ index ][ cnt2 ];
      }
      delete[] Ctensors[ index ][ cnt2 ];
      delete[] CtensorsT[ index ][ cnt2 ];
      delete[] Dtensors[ index ][ cnt2 ];
      delete[] DtensorsT[ index ][ cnt2 ];
   }
   delete[] Atensors[ index ];
   delete[] AtensorsT[ index ];
   delete[] Btensors[ index ];
   delete[] BtensorsT[ index ];
   delete[] Ctensors[ index ];
   delete[] CtensorsT[ index ];
   delete[] Dtensors[ index ];
   delete[] DtensorsT[ index ];

   // QQtensors_MPSDT_MPS : certain processes own certain QQtensors_MPSDT_MPS
   for ( int cnt2 = 0; cnt2 < Cbound; cnt2++ ) {
      delete Qtensors[ index ][ cnt2 ];
      delete QtensorsT[ index ][ cnt2 ];
   }
   delete[] Qtensors[ index ];
   delete[] QtensorsT[ index ];

   // Xtensors
   delete Xtensors[ index ];

   // Otensors
   delete Otensors[ index ];
}