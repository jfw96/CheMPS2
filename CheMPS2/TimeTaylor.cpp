
#include "TimeTaylor.h"
#include <assert.h>
#include <iostream>

#include "CHeffNS.h"
#include "CSobject.h"
#include "Special.h"

CheMPS2::TimeTaylor::TimeTaylor( Problem * probIn, ConvergenceScheme * schemeIn, Logger * loggerIn )
    : prob( probIn ), scheme( schemeIn ), logger( loggerIn ), L( probIn->gL() ) {
   assert( probIn->checkConsistency() );

   prob->construct_mxelem();

   logger->TextWithDate( "Starting to run a time evolution calculation", time( NULL ) );
   ( *logger ) << hashline;

   denBK = new SyBookkeeper( prob, scheme->get_D( 0 ) );

   ( *logger ) << "### Full CI dimensions are:\n";
   ( *logger ) << "### ";
   for ( int i = 0; i < L + 1; i++ ) {
      ( *logger ) << denBK->gFCIDimAtBound( i ) << " ";
   }
   ( *logger ) << "\n";

   MPS         = new CTensorT *[ L ];
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

   for ( int index = 0; index < L; index++ ) {
      MPS[ index ] = new CTensorT( index, denBK );
   }

   switch ( prob->gInitial() ) {
      case Problem::INIT_RANDOM: {
         for ( int idx = 0; idx < L; ++idx ) {
            MPS[ idx ]->random();
         }
         double mps_norm = norm( MPS );
         MPS[ 0 ]->number_operator( 0.0, 1.0 / mps_norm );
         break;
      }
      case Problem::HARTREE_FOCK_GS: {
         for ( int idx = 0; idx < L; ++idx ) {
            MPS[ idx ]->Clear();
         }

         const int nElec = prob->gN();
         for ( int idx = 0; idx < L; ++idx ) {
            for ( int ikappa = 0; ikappa < MPS[ idx ]->gNKappa(); ++ikappa ) {
               if ( idx < ( nElec / 2.0 ) && MPS[ idx ]->gNL( ikappa ) == ( idx * 2 ) && MPS[ idx ]->gNR( ikappa ) == ( idx + 1 ) * 2 ) {
                  dcomplex * block = MPS[ idx ]->gStorage() + MPS[ idx ]->gKappa2index( ikappa );
                  block[ 0 ]       = 1.0;
               } else if ( idx >= ( nElec / 2.0 ) && MPS[ idx ]->gNL( ikappa ) == nElec && MPS[ idx ]->gNR( ikappa ) == nElec ) {
                  dcomplex * block = MPS[ idx ]->gStorage() + MPS[ idx ]->gKappa2index( ikappa );
                  block[ 0 ]       = 1.0;
               }
            }
         }
         break;
      }
   }
}

CheMPS2::TimeTaylor::~TimeTaylor() {

   deleteAllBoundaryOperators();

   for ( int site = 0; site < L; site++ ) {
      delete MPS[ site ];
   }
   delete[] MPS;

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

   delete denBK;

   logger->TextWithDate( "Finished to run a time evolution calculation", time( NULL ) );
   ( *logger ) << hashline;
}

void CheMPS2::TimeTaylor::updateMovingLeftSafe( const int cnt ) {
   if ( isAllocated[ cnt ] == 1 ) {
      deleteTensors( cnt, true );
      isAllocated[ cnt ] = 0;
   }
   if ( isAllocated[ cnt ] == 0 ) {
      allocateTensors( cnt, false );
      isAllocated[ cnt ] = 2;
   }
   updateMovingLeft( cnt );
}

void CheMPS2::TimeTaylor::updateMovingRightSafe( const int cnt ) {
   if ( isAllocated[ cnt ] == 2 ) {
      deleteTensors( cnt, false );
      isAllocated[ cnt ] = 0;
   }
   if ( isAllocated[ cnt ] == 0 ) {
      allocateTensors( cnt, true );
      isAllocated[ cnt ] = 1;
   }
   updateMovingRight( cnt );
}

void CheMPS2::TimeTaylor::deleteAllBoundaryOperators() {
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

void CheMPS2::TimeTaylor::updateMovingLeft( const int index ) {

   const int dimL = std::max( denBKDT->gMaxDimAtBound( index + 1 ), denBK->gMaxDimAtBound( index + 1 ) );
   const int dimR = std::max( denBKDT->gMaxDimAtBound( index + 2 ), denBK->gMaxDimAtBound( index + 2 ) );

#pragma omp parallel
   {
      dcomplex * workmem = new dcomplex[ dimL * dimR ];
// Ltensors_MPSDT_MPS : all processes own all Ltensors_MPSDT_MPS
#pragma omp for schedule( static ) nowait
      for ( int cnt2 = 0; cnt2 < L - 1 - index; cnt2++ ) {
         if ( cnt2 == 0 ) {
            if ( index == L - 2 ) {
               Ltensors[ index ][ cnt2 ]->create( MPSDT[ index + 1 ], MPS[ index + 1 ], NULL, NULL );
               LtensorsT[ index ][ cnt2 ]->create( MPSDT[ index + 1 ], MPS[ index + 1 ], NULL, NULL );
            } else {
               Ltensors[ index ][ cnt2 ]->create( MPSDT[ index + 1 ], MPS[ index + 1 ], Otensors[ index + 1 ], workmem );
               LtensorsT[ index ][ cnt2 ]->create( MPSDT[ index + 1 ], MPS[ index + 1 ], Otensors[ index + 1 ], workmem );
            }
         } else {
            Ltensors[ index ][ cnt2 ]->update( Ltensors[ index + 1 ][ cnt2 - 1 ], MPSDT[ index + 1 ], MPS[ index + 1 ], workmem );
            LtensorsT[ index ][ cnt2 ]->update( LtensorsT[ index + 1 ][ cnt2 - 1 ], MPSDT[ index + 1 ], MPS[ index + 1 ], workmem );
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
                  F0tensors[ index ][ cnt2 ][ cnt3 ]->makenew( MPSDT[ index + 1 ], MPS[ index + 1 ], NULL, NULL );
                  F0tensorsT[ index ][ cnt2 ][ cnt3 ]->makenew( MPSDT[ index + 1 ], MPS[ index + 1 ], NULL, NULL );
                  F1tensors[ index ][ cnt2 ][ cnt3 ]->makenew( MPSDT[ index + 1 ], MPS[ index + 1 ], NULL, NULL );
                  F1tensorsT[ index ][ cnt2 ][ cnt3 ]->makenew( MPSDT[ index + 1 ], MPS[ index + 1 ], NULL, NULL );
                  S0tensors[ index ][ cnt2 ][ cnt3 ]->makenew( MPSDT[ index + 1 ], MPS[ index + 1 ], NULL, NULL );
                  S0tensorsT[ index ][ cnt2 ][ cnt3 ]->makenew( MPSDT[ index + 1 ], MPS[ index + 1 ], NULL, NULL );
               } else {
                  F0tensors[ index ][ cnt2 ][ cnt3 ]->makenew( MPSDT[ index + 1 ], MPS[ index + 1 ], Otensors[ index + 1 ], workmem );
                  F0tensorsT[ index ][ cnt2 ][ cnt3 ]->makenew( MPSDT[ index + 1 ], MPS[ index + 1 ], Otensors[ index + 1 ], workmem );
                  F1tensors[ index ][ cnt2 ][ cnt3 ]->makenew( MPSDT[ index + 1 ], MPS[ index + 1 ], Otensors[ index + 1 ], workmem );
                  F1tensorsT[ index ][ cnt2 ][ cnt3 ]->makenew( MPSDT[ index + 1 ], MPS[ index + 1 ], Otensors[ index + 1 ], workmem );
                  S0tensors[ index ][ cnt2 ][ cnt3 ]->makenew( MPSDT[ index + 1 ], MPS[ index + 1 ], Otensors[ index + 1 ], workmem );
                  S0tensorsT[ index ][ cnt2 ][ cnt3 ]->makenew( MPSDT[ index + 1 ], MPS[ index + 1 ], Otensors[ index + 1 ], workmem );
               }
            } else {
               F0tensors[ index ][ cnt2 ][ cnt3 ]->makenew( Ltensors[ index + 1 ][ cnt2 - 1 ], MPSDT[ index + 1 ],
                                                            MPS[ index + 1 ], workmem );
               F0tensorsT[ index ][ cnt2 ][ cnt3 ]->makenew( LtensorsT[ index + 1 ][ cnt2 - 1 ], MPSDT[ index + 1 ],
                                                             MPS[ index + 1 ], workmem );
               F1tensors[ index ][ cnt2 ][ cnt3 ]->makenew( Ltensors[ index + 1 ][ cnt2 - 1 ], MPSDT[ index + 1 ],
                                                            MPS[ index + 1 ], workmem );
               F1tensorsT[ index ][ cnt2 ][ cnt3 ]->makenew( LtensorsT[ index + 1 ][ cnt2 - 1 ], MPSDT[ index + 1 ],
                                                             MPS[ index + 1 ], workmem );
               S0tensors[ index ][ cnt2 ][ cnt3 ]->makenew( Ltensors[ index + 1 ][ cnt2 - 1 ], MPSDT[ index + 1 ],
                                                            MPS[ index + 1 ], workmem );
               S0tensorsT[ index ][ cnt2 ][ cnt3 ]->makenew( LtensorsT[ index + 1 ][ cnt2 - 1 ], MPSDT[ index + 1 ],
                                                             MPS[ index + 1 ], workmem );
               S1tensors[ index ][ cnt2 ][ cnt3 ]->makenew( Ltensors[ index + 1 ][ cnt2 - 1 ], MPSDT[ index + 1 ],
                                                            MPS[ index + 1 ], workmem );
               S1tensorsT[ index ][ cnt2 ][ cnt3 ]->makenew( LtensorsT[ index + 1 ][ cnt2 - 1 ], MPSDT[ index + 1 ],
                                                             MPS[ index + 1 ], workmem );
            }
         } else {
            F0tensors[ index ][ cnt2 ][ cnt3 ]->update( F0tensors[ index + 1 ][ cnt2 ][ cnt3 - 1 ], MPSDT[ index + 1 ],
                                                        MPS[ index + 1 ], workmem );
            F0tensorsT[ index ][ cnt2 ][ cnt3 ]->update( F0tensorsT[ index + 1 ][ cnt2 ][ cnt3 - 1 ], MPSDT[ index + 1 ],
                                                         MPS[ index + 1 ], workmem );
            F1tensors[ index ][ cnt2 ][ cnt3 ]->update( F1tensors[ index + 1 ][ cnt2 ][ cnt3 - 1 ], MPSDT[ index + 1 ],
                                                        MPS[ index + 1 ], workmem );
            F1tensorsT[ index ][ cnt2 ][ cnt3 ]->update( F1tensorsT[ index + 1 ][ cnt2 ][ cnt3 - 1 ], MPSDT[ index + 1 ],
                                                         MPS[ index + 1 ], workmem );
            S0tensors[ index ][ cnt2 ][ cnt3 ]->update( S0tensors[ index + 1 ][ cnt2 ][ cnt3 - 1 ], MPSDT[ index + 1 ],
                                                        MPS[ index + 1 ], workmem );
            S0tensorsT[ index ][ cnt2 ][ cnt3 ]->update( S0tensorsT[ index + 1 ][ cnt2 ][ cnt3 - 1 ], MPSDT[ index + 1 ],
                                                         MPS[ index + 1 ], workmem );
            if ( cnt2 > 0 ) {
               S1tensors[ index ][ cnt2 ][ cnt3 ]->update( S1tensors[ index + 1 ][ cnt2 ][ cnt3 - 1 ], MPSDT[ index + 1 ],
                                                           MPS[ index + 1 ], workmem );
               S1tensorsT[ index ][ cnt2 ][ cnt3 ]->update( S1tensorsT[ index + 1 ][ cnt2 ][ cnt3 - 1 ], MPSDT[ index + 1 ],
                                                            MPS[ index + 1 ], workmem );
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
         const int irrep_prod = Irreps::directProd( denBK->gIrrep( siteindex1 ), denBK->gIrrep( siteindex2 ) );
         if ( index == L - 2 ) {
            Atensors[ index ][ cnt2 ][ cnt3 ]->clear();
            AtensorsT[ index ][ cnt2 ][ cnt3 ]->clear();
            if ( cnt2 > 0 ) {
               Btensors[ index ][ cnt2 ][ cnt3 ]->clear();
            }
            if ( cnt2 > 0 ) {
               BtensorsT[ index ][ cnt2 ][ cnt3 ]->clear();
            }
            Ctensors[ index ][ cnt2 ][ cnt3 ]->clear();
            CtensorsT[ index ][ cnt2 ][ cnt3 ]->clear();
            Dtensors[ index ][ cnt2 ][ cnt3 ]->clear();
            DtensorsT[ index ][ cnt2 ][ cnt3 ]->clear();
         } else {
            Atensors[ index ][ cnt2 ][ cnt3 ]->update( Atensors[ index + 1 ][ cnt2 ][ cnt3 + 1 ], MPSDT[ index + 1 ],
                                                       MPS[ index + 1 ], workmem );
            AtensorsT[ index ][ cnt2 ][ cnt3 ]->update( AtensorsT[ index + 1 ][ cnt2 ][ cnt3 + 1 ], MPSDT[ index + 1 ],
                                                        MPS[ index + 1 ], workmem );
            if ( cnt2 > 0 ) {
               Btensors[ index ][ cnt2 ][ cnt3 ]->update( Btensors[ index + 1 ][ cnt2 ][ cnt3 + 1 ], MPSDT[ index + 1 ],
                                                          MPS[ index + 1 ], workmem );
               BtensorsT[ index ][ cnt2 ][ cnt3 ]->update( BtensorsT[ index + 1 ][ cnt2 ][ cnt3 + 1 ], MPSDT[ index + 1 ],
                                                           MPS[ index + 1 ], workmem );
            }
            Ctensors[ index ][ cnt2 ][ cnt3 ]->update( Ctensors[ index + 1 ][ cnt2 ][ cnt3 + 1 ], MPSDT[ index + 1 ],
                                                       MPS[ index + 1 ], workmem );
            CtensorsT[ index ][ cnt2 ][ cnt3 ]->update( CtensorsT[ index + 1 ][ cnt2 ][ cnt3 + 1 ], MPSDT[ index + 1 ],
                                                        MPS[ index + 1 ], workmem );
            Dtensors[ index ][ cnt2 ][ cnt3 ]->update( Dtensors[ index + 1 ][ cnt2 ][ cnt3 + 1 ], MPSDT[ index + 1 ],
                                                       MPS[ index + 1 ], workmem );
            DtensorsT[ index ][ cnt2 ][ cnt3 ]->update( DtensorsT[ index + 1 ][ cnt2 ][ cnt3 + 1 ], MPSDT[ index + 1 ],
                                                        MPS[ index + 1 ], workmem );
         }
         for ( int num = 0; num < L - index - 1; num++ ) {
            if ( irrep_prod ==
                 S0tensorsT[ index ][ num ][ 0 ]->get_irrep() ) { // Then the matrix elements are not 0 due to symm.
               double alpha = prob->gMxElement( siteindex1, siteindex2, index + 1, index + 1 + num );
               if ( ( cnt2 == 0 ) && ( num == 0 ) )
                  alpha *= 0.5;
               if ( ( cnt2 > 0 ) && ( num > 0 ) )
                  alpha += prob->gMxElement( siteindex1, siteindex2, index + 1 + num,
                                             index + 1 );
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

                  alpha = -prob->gMxElement( siteindex1, index + 1 + num, index + 1, siteindex2 ); // Second line for Ctensors_MPS_MPSDT
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
            Qtensors[ index ][ cnt2 ]->AddTermSimple( MPSDT[ index + 1 ], MPS[ index + 1 ], NULL, NULL );
            QtensorsT[ index ][ cnt2 ]->AddTermSimple( MPSDT[ index + 1 ], MPS[ index + 1 ], NULL, NULL );
         } else {
            dcomplex * workmemBIS = new dcomplex[ dimR * dimR ];
            Qtensors[ index ][ cnt2 ]->update( Qtensors[ index + 1 ][ cnt2 + 1 ], MPSDT[ index + 1 ],
                                               MPS[ index + 1 ], workmem );
            QtensorsT[ index ][ cnt2 ]->update( QtensorsT[ index + 1 ][ cnt2 + 1 ], MPSDT[ index + 1 ],
                                                MPS[ index + 1 ], workmem );
            Qtensors[ index ][ cnt2 ]->AddTermSimple( MPSDT[ index + 1 ], MPS[ index + 1 ], Otensors[ index + 1 ], workmem );
            QtensorsT[ index ][ cnt2 ]->AddTermSimple( MPSDT[ index + 1 ], MPS[ index + 1 ], Otensors[ index + 1 ], workmem );
            Qtensors[ index ][ cnt2 ]->AddTermsL( Ltensors[ index + 1 ], LtensorsT[ index + 1 ], MPSDT[ index + 1 ],
                                                  MPS[ index + 1 ], workmemBIS, workmem );
            QtensorsT[ index ][ cnt2 ]->AddTermsL( Ltensors[ index + 1 ], LtensorsT[ index + 1 ], MPSDT[ index + 1 ],
                                                   MPS[ index + 1 ], workmemBIS, workmem );
            Qtensors[ index ][ cnt2 ]->AddTermsAB( Atensors[ index + 1 ][ cnt2 + 1 ][ 0 ], Btensors[ index + 1 ][ cnt2 + 1 ][ 0 ],
                                                   MPSDT[ index + 1 ], MPS[ index + 1 ], workmemBIS, workmem );
            QtensorsT[ index ][ cnt2 ]->AddTermsAB( AtensorsT[ index + 1 ][ cnt2 + 1 ][ 0 ], BtensorsT[ index + 1 ][ cnt2 + 1 ][ 0 ],
                                                    MPSDT[ index + 1 ], MPS[ index + 1 ], workmemBIS, workmem );
            Qtensors[ index ][ cnt2 ]->AddTermsCD( Ctensors[ index + 1 ][ cnt2 + 1 ][ 0 ], Dtensors[ index + 1 ][ cnt2 + 1 ][ 0 ],
                                                   MPSDT[ index + 1 ], MPS[ index + 1 ], workmemBIS, workmem );
            QtensorsT[ index ][ cnt2 ]->AddTermsCD( CtensorsT[ index + 1 ][ cnt2 + 1 ][ 0 ], DtensorsT[ index + 1 ][ cnt2 + 1 ][ 0 ],
                                                    MPSDT[ index + 1 ], MPS[ index + 1 ], workmemBIS, workmem );
            delete[] workmemBIS;
         }
      }

      delete[] workmem;
   }
   // Xtensors
   if ( index == L - 2 ) {
      Xtensors[ index ]->update( MPSDT[ index + 1 ], MPS[ index + 1 ] );
   } else {
      Xtensors[ index ]->update( MPSDT[ index + 1 ], MPS[ index + 1 ], Otensors[ index + 1 ],
                                 Ltensors[ index + 1 ], LtensorsT[ index + 1 ], Xtensors[ index + 1 ],
                                 Qtensors[ index + 1 ][ 0 ], QtensorsT[ index + 1 ][ 0 ],
                                 Atensors[ index + 1 ][ 0 ][ 0 ], AtensorsT[ index + 1 ][ 0 ][ 0 ],
                                 CtensorsT[ index + 1 ][ 0 ][ 0 ], DtensorsT[ index + 1 ][ 0 ][ 0 ] );
   }

   // Otensors
   if ( index == L - 2 ) {
      Otensors[ index ]->create( MPSDT[ index + 1 ], MPS[ index + 1 ] );
   } else {
      Otensors[ index ]->update_ownmem( MPSDT[ index + 1 ], MPS[ index + 1 ], Otensors[ index + 1 ] );
   }
}

void CheMPS2::TimeTaylor::updateMovingRight( const int index ) {

   const int dimL = std::max( denBKDT->gMaxDimAtBound( index ), denBK->gMaxDimAtBound( index ) );
   const int dimR = std::max( denBKDT->gMaxDimAtBound( index + 1 ), denBK->gMaxDimAtBound( index + 1 ) );

#pragma omp parallel
   {
      dcomplex * workmem = new dcomplex[ dimL * dimR ];

// Ltensors
#pragma omp for schedule( static ) nowait
      for ( int cnt2 = 0; cnt2 < index + 1; cnt2++ ) {
         if ( cnt2 == 0 ) {
            if ( index == 0 ) {
               Ltensors[ index ][ cnt2 ]->create( MPSDT[ index ], MPS[ index ], NULL, NULL );
               LtensorsT[ index ][ cnt2 ]->create( MPSDT[ index ], MPS[ index ], NULL, NULL );
            } else {
               Ltensors[ index ][ cnt2 ]->create( MPSDT[ index ], MPS[ index ], Otensors[ index - 1 ], workmem );
               LtensorsT[ index ][ cnt2 ]->create( MPSDT[ index ], MPS[ index ], Otensors[ index - 1 ], workmem );
            }
         } else {
            Ltensors[ index ][ cnt2 ]->update( Ltensors[ index - 1 ][ cnt2 - 1 ], MPSDT[ index ], MPS[ index ], workmem );
            LtensorsT[ index ][ cnt2 ]->update( LtensorsT[ index - 1 ][ cnt2 - 1 ], MPSDT[ index ], MPS[ index ], workmem );
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
                  F0tensors[ index ][ cnt2 ][ cnt3 ]->makenew( MPSDT[ index ], MPS[ index ], NULL, NULL );
                  F1tensors[ index ][ cnt2 ][ cnt3 ]->makenew( MPSDT[ index ], MPS[ index ], NULL, NULL );
                  S0tensors[ index ][ cnt2 ][ cnt3 ]->makenew( MPSDT[ index ], MPS[ index ], NULL, NULL );
                  F0tensorsT[ index ][ cnt2 ][ cnt3 ]->makenew( MPSDT[ index ], MPS[ index ], NULL, NULL );
                  F1tensorsT[ index ][ cnt2 ][ cnt3 ]->makenew( MPSDT[ index ], MPS[ index ], NULL, NULL );
                  S0tensorsT[ index ][ cnt2 ][ cnt3 ]->makenew( MPSDT[ index ], MPS[ index ], NULL, NULL );
               } else {
                  F0tensors[ index ][ cnt2 ][ cnt3 ]->makenew( MPSDT[ index ], MPS[ index ], Otensors[ index - 1 ], workmem );
                  F1tensors[ index ][ cnt2 ][ cnt3 ]->makenew( MPSDT[ index ], MPS[ index ], Otensors[ index - 1 ], workmem );
                  S0tensors[ index ][ cnt2 ][ cnt3 ]->makenew( MPSDT[ index ], MPS[ index ], Otensors[ index - 1 ], workmem );
                  F0tensorsT[ index ][ cnt2 ][ cnt3 ]->makenew( MPSDT[ index ], MPS[ index ], Otensors[ index - 1 ], workmem );
                  F1tensorsT[ index ][ cnt2 ][ cnt3 ]->makenew( MPSDT[ index ], MPS[ index ], Otensors[ index - 1 ], workmem );
                  S0tensorsT[ index ][ cnt2 ][ cnt3 ]->makenew( MPSDT[ index ], MPS[ index ], Otensors[ index - 1 ], workmem );
               }
               // // S1[ index ][ 0 ][ cnt3 ] doesn't exist
            } else {
               F0tensors[ index ][ cnt2 ][ cnt3 ]->makenew( Ltensors[ index - 1 ][ cnt2 - 1 ], MPSDT[ index ], MPS[ index ], workmem );
               F1tensors[ index ][ cnt2 ][ cnt3 ]->makenew( Ltensors[ index - 1 ][ cnt2 - 1 ], MPSDT[ index ], MPS[ index ], workmem );
               S0tensors[ index ][ cnt2 ][ cnt3 ]->makenew( Ltensors[ index - 1 ][ cnt2 - 1 ], MPSDT[ index ], MPS[ index ], workmem );
               S1tensors[ index ][ cnt2 ][ cnt3 ]->makenew( Ltensors[ index - 1 ][ cnt2 - 1 ], MPSDT[ index ], MPS[ index ], workmem );
               F0tensorsT[ index ][ cnt2 ][ cnt3 ]->makenew( LtensorsT[ index - 1 ][ cnt2 - 1 ], MPSDT[ index ], MPS[ index ], workmem );
               F1tensorsT[ index ][ cnt2 ][ cnt3 ]->makenew( LtensorsT[ index - 1 ][ cnt2 - 1 ], MPSDT[ index ], MPS[ index ], workmem );
               S0tensorsT[ index ][ cnt2 ][ cnt3 ]->makenew( LtensorsT[ index - 1 ][ cnt2 - 1 ], MPSDT[ index ], MPS[ index ], workmem );
               S1tensorsT[ index ][ cnt2 ][ cnt3 ]->makenew( LtensorsT[ index - 1 ][ cnt2 - 1 ], MPSDT[ index ], MPS[ index ], workmem );
            }
         } else {
            F0tensors[ index ][ cnt2 ][ cnt3 ]->update( F0tensors[ index - 1 ][ cnt2 ][ cnt3 - 1 ], MPSDT[ index ],
                                                        MPS[ index ], workmem );
            F1tensors[ index ][ cnt2 ][ cnt3 ]->update( F1tensors[ index - 1 ][ cnt2 ][ cnt3 - 1 ], MPSDT[ index ],
                                                        MPS[ index ], workmem );
            S0tensors[ index ][ cnt2 ][ cnt3 ]->update( S0tensors[ index - 1 ][ cnt2 ][ cnt3 - 1 ], MPSDT[ index ],
                                                        MPS[ index ], workmem );
            F0tensorsT[ index ][ cnt2 ][ cnt3 ]->update( F0tensorsT[ index - 1 ][ cnt2 ][ cnt3 - 1 ], MPSDT[ index ],
                                                         MPS[ index ], workmem );
            F1tensorsT[ index ][ cnt2 ][ cnt3 ]->update( F1tensorsT[ index - 1 ][ cnt2 ][ cnt3 - 1 ], MPSDT[ index ],
                                                         MPS[ index ], workmem );
            S0tensorsT[ index ][ cnt2 ][ cnt3 ]->update( S0tensorsT[ index - 1 ][ cnt2 ][ cnt3 - 1 ], MPSDT[ index ],
                                                         MPS[ index ], workmem );
            if ( cnt2 > 0 ) {
               S1tensors[ index ][ cnt2 ][ cnt3 ]->update( S1tensors[ index - 1 ][ cnt2 ][ cnt3 - 1 ], MPSDT[ index ],
                                                           MPS[ index ], workmem );
               S1tensorsT[ index ][ cnt2 ][ cnt3 ]->update( S1tensorsT[ index - 1 ][ cnt2 ][ cnt3 - 1 ], MPSDT[ index ],
                                                            MPS[ index ], workmem );
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
         const int irrep_prod = CheMPS2::Irreps::directProd( denBK->gIrrep( siteindex1 ), denBK->gIrrep( siteindex2 ) );
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
            Atensors[ index ][ cnt2 ][ cnt3 ]->update( Atensors[ index - 1 ][ cnt2 ][ cnt3 + 1 ], MPSDT[ index ], MPS[ index ], workmem );
            AtensorsT[ index ][ cnt2 ][ cnt3 ]->update( AtensorsT[ index - 1 ][ cnt2 ][ cnt3 + 1 ], MPSDT[ index ],
                                                        MPS[ index ], workmem );
            if ( cnt2 > 0 ) {
               Btensors[ index ][ cnt2 ][ cnt3 ]->update( Btensors[ index - 1 ][ cnt2 ][ cnt3 + 1 ], MPSDT[ index ],
                                                          MPS[ index ], workmem );
            }
            if ( cnt2 > 0 ) {
               BtensorsT[ index ][ cnt2 ][ cnt3 ]->update( BtensorsT[ index - 1 ][ cnt2 ][ cnt3 + 1 ], MPSDT[ index ],
                                                           MPS[ index ], workmem );
            }
            Ctensors[ index ][ cnt2 ][ cnt3 ]->update( Ctensors[ index - 1 ][ cnt2 ][ cnt3 + 1 ], MPSDT[ index ],
                                                       MPS[ index ], workmem );
            CtensorsT[ index ][ cnt2 ][ cnt3 ]->update( CtensorsT[ index - 1 ][ cnt2 ][ cnt3 + 1 ], MPSDT[ index ],
                                                        MPS[ index ], workmem );
            Dtensors[ index ][ cnt2 ][ cnt3 ]->update( Dtensors[ index - 1 ][ cnt2 ][ cnt3 + 1 ], MPSDT[ index ],
                                                       MPS[ index ], workmem );
            DtensorsT[ index ][ cnt2 ][ cnt3 ]->update( DtensorsT[ index - 1 ][ cnt2 ][ cnt3 + 1 ], MPSDT[ index ],
                                                        MPS[ index ], workmem );
         }

         for ( int num = 0; num < index + 1; num++ ) {
            if ( irrep_prod ==
                 S0tensorsT[ index ][ num ][ 0 ]->get_irrep() ) { // Then the matrix elements are not 0 due to symm.
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

// QQtensors_MPSDT_MPS : certain processes own certain QQtensors_MPSDT_MPS ---
// You don't want to locally parallellize when sending and receiving buffers!
#pragma omp for schedule( static ) nowait
      for ( int cnt2 = 0; cnt2 < L - 1 - index; cnt2++ ) {
         if ( index == 0 ) {
            Qtensors[ index ][ cnt2 ]->clear();
            QtensorsT[ index ][ cnt2 ]->clear();
            Qtensors[ index ][ cnt2 ]->AddTermSimple( MPSDT[ index ], MPS[ index ], NULL, NULL );
            QtensorsT[ index ][ cnt2 ]->AddTermSimple( MPSDT[ index ], MPS[ index ], NULL, NULL );
         } else {
            dcomplex * workmemBIS = new dcomplex[ dimL * dimL ];
            Qtensors[ index ][ cnt2 ]->update( Qtensors[ index - 1 ][ cnt2 + 1 ], MPSDT[ index ], MPS[ index ], workmem );
            QtensorsT[ index ][ cnt2 ]->update( QtensorsT[ index - 1 ][ cnt2 + 1 ], MPSDT[ index ], MPS[ index ], workmem );
            Qtensors[ index ][ cnt2 ]->AddTermSimple( MPSDT[ index ], MPS[ index ], Otensors[ index - 1 ], workmem );
            QtensorsT[ index ][ cnt2 ]->AddTermSimple( MPSDT[ index ], MPS[ index ], Otensors[ index - 1 ], workmem );
            Qtensors[ index ][ cnt2 ]->AddTermsL( Ltensors[ index - 1 ], LtensorsT[ index - 1 ], MPSDT[ index ],
                                                  MPS[ index ], workmemBIS, workmem );
            QtensorsT[ index ][ cnt2 ]->AddTermsL( Ltensors[ index - 1 ], LtensorsT[ index - 1 ], MPSDT[ index ],
                                                   MPS[ index ], workmemBIS, workmem );
            Qtensors[ index ][ cnt2 ]->AddTermsAB( Atensors[ index - 1 ][ cnt2 + 1 ][ 0 ], Btensors[ index - 1 ][ cnt2 + 1 ][ 0 ],
                                                   MPSDT[ index ], MPS[ index ], workmemBIS, workmem );
            QtensorsT[ index ][ cnt2 ]->AddTermsAB( AtensorsT[ index - 1 ][ cnt2 + 1 ][ 0 ], BtensorsT[ index - 1 ][ cnt2 + 1 ][ 0 ],
                                                    MPSDT[ index ], MPS[ index ], workmemBIS, workmem );
            Qtensors[ index ][ cnt2 ]->AddTermsCD( Ctensors[ index - 1 ][ cnt2 + 1 ][ 0 ], Dtensors[ index - 1 ][ cnt2 + 1 ][ 0 ],
                                                   MPSDT[ index ], MPS[ index ], workmemBIS, workmem );
            QtensorsT[ index ][ cnt2 ]->AddTermsCD( CtensorsT[ index - 1 ][ cnt2 + 1 ][ 0 ], DtensorsT[ index - 1 ][ cnt2 + 1 ][ 0 ],
                                                    MPSDT[ index ], MPS[ index ], workmemBIS, workmem );
            delete[] workmemBIS;
         }
      }

      delete[] workmem;
   }

   // Xtensors
   if ( index == 0 ) {
      Xtensors[ index ]->update( MPSDT[ index ], MPS[ index ] );
   } else {
      Xtensors[ index ]->update( MPSDT[ index ], MPS[ index ], Otensors[ index - 1 ], Ltensors[ index - 1 ],
                                 LtensorsT[ index - 1 ], Xtensors[ index - 1 ], Qtensors[ index - 1 ][ 0 ],
                                 QtensorsT[ index - 1 ][ 0 ], Atensors[ index - 1 ][ 0 ][ 0 ],
                                 AtensorsT[ index - 1 ][ 0 ][ 0 ], CtensorsT[ index - 1 ][ 0 ][ 0 ],
                                 DtensorsT[ index - 1 ][ 0 ][ 0 ] );
   }

   // Otensors
   if ( index == 0 ) {
      Otensors[ index ]->create( MPSDT[ index ], MPS[ index ] );
   } else {
      Otensors[ index ]->update_ownmem( MPSDT[ index ], MPS[ index ], Otensors[ index - 1 ] );
   }
}

void CheMPS2::TimeTaylor::allocateTensors( const int index, const bool movingRight ) {

   if ( movingRight ) {
      // Ltensors
      Ltensors[ index ]  = new CTensorL *[ index + 1 ];
      LtensorsT[ index ] = new CTensorLT *[ index + 1 ];
      for ( int cnt2 = 0; cnt2 < index + 1; cnt2++ ) {
         Ltensors[ index ][ cnt2 ]  = new CTensorL( index + 1, denBK->gIrrep( index - cnt2 ), movingRight, denBKDT, denBK );
         LtensorsT[ index ][ cnt2 ] = new CTensorLT( index + 1, denBK->gIrrep( index - cnt2 ), movingRight, denBKDT, denBK );
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
            const int Iprod =
                CheMPS2::Irreps::directProd( denBKDT->gIrrep( index - cnt2 - cnt3 ), denBKDT->gIrrep( index - cnt3 ) );
            F0tensors[ index ][ cnt2 ][ cnt3 ]  = new CTensorF0( index + 1, Iprod, movingRight, denBKDT, denBK );
            F0tensorsT[ index ][ cnt2 ][ cnt3 ] = new CTensorF0T( index + 1, Iprod, movingRight, denBKDT, denBK );
            F1tensors[ index ][ cnt2 ][ cnt3 ]  = new CTensorF1( index + 1, Iprod, movingRight, denBKDT, denBK );
            F1tensorsT[ index ][ cnt2 ][ cnt3 ] = new CTensorF1T( index + 1, Iprod, movingRight, denBKDT, denBK );
            S0tensors[ index ][ cnt2 ][ cnt3 ]  = new CTensorS0( index + 1, Iprod, movingRight, denBKDT, denBK );
            S0tensorsT[ index ][ cnt2 ][ cnt3 ] = new CTensorS0T( index + 1, Iprod, movingRight, denBKDT, denBK );
            if ( cnt2 > 0 ) {
               S1tensors[ index ][ cnt2 ][ cnt3 ] = new CTensorS1( index + 1, Iprod, movingRight, denBKDT, denBK );
            }
            if ( cnt2 > 0 ) {
               S1tensorsT[ index ][ cnt2 ][ cnt3 ] = new CTensorS1T( index + 1, Iprod, movingRight, denBKDT, denBK );
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
            Btensors[ index ][ cnt2 ] = new CTensorOperator *[ L - 1 - index - cnt2 ];
         }
         if ( cnt2 > 0 ) {
            BtensorsT[ index ][ cnt2 ] = new CTensorOperator *[ L - 1 - index - cnt2 ];
         }
         Ctensors[ index ][ cnt2 ]  = new CTensorOperator *[ L - 1 - index - cnt2 ];
         CtensorsT[ index ][ cnt2 ] = new CTensorOperator *[ L - 1 - index - cnt2 ];
         Dtensors[ index ][ cnt2 ]  = new CTensorOperator *[ L - 1 - index - cnt2 ];
         DtensorsT[ index ][ cnt2 ] = new CTensorOperator *[ L - 1 - index - cnt2 ];
         for ( int cnt3 = 0; cnt3 < L - 1 - index - cnt2; cnt3++ ) {
            const int Idiff                    = CheMPS2::Irreps::directProd( denBKDT->gIrrep( index + 1 + cnt2 + cnt3 ),
                                                           denBKDT->gIrrep( index + 1 + cnt3 ) );
            Atensors[ index ][ cnt2 ][ cnt3 ]  = new CTensorOperator( index + 1, 0, 2, Idiff, movingRight, true, false, denBKDT, denBK );
            AtensorsT[ index ][ cnt2 ][ cnt3 ] = new CTensorOperator( index + 1, 0, -2, Idiff, movingRight, false, false, denBKDT, denBK );
            if ( cnt2 > 0 ) {
               Btensors[ index ][ cnt2 ][ cnt3 ] = new CTensorOperator( index + 1, 2, 2, Idiff, movingRight, true, false, denBKDT, denBK );
            }
            if ( cnt2 > 0 ) {
               BtensorsT[ index ][ cnt2 ][ cnt3 ] = new CTensorOperator( index + 1, 2, -2, Idiff, movingRight,
                                                                         false, false, denBKDT, denBK );
            }
            Ctensors[ index ][ cnt2 ][ cnt3 ]  = new CTensorOperator( index + 1, 0, 0, Idiff, movingRight, true, false, denBKDT, denBK );
            CtensorsT[ index ][ cnt2 ][ cnt3 ] = new CTensorOperator( index + 1, 0, 0, Idiff, movingRight, false, false, denBKDT, denBK );
            Dtensors[ index ][ cnt2 ][ cnt3 ]  = new CTensorOperator( index + 1, 2, 0, Idiff, movingRight,
                                                                     movingRight, false, denBKDT, denBK );
            DtensorsT[ index ][ cnt2 ][ cnt3 ] = new CTensorOperator( index + 1, 2, 0, Idiff, movingRight,
                                                                      !movingRight, false, denBKDT, denBK );
         }
      }

      // QQtensors_MPSDT_MPS
      // To right: Qtens[ cnt][ cnt2 ] = operator on site cnt+1+cnt2; at boundary cnt+1
      Qtensors[ index ]  = new CTensorQ *[ L - 1 - index ];
      QtensorsT[ index ] = new CTensorQT *[ L - 1 - index ];
      for ( int cnt2 = 0; cnt2 < L - 1 - index; cnt2++ ) {
         Qtensors[ index ][ cnt2 ]  = new CTensorQ( index + 1, denBK->gIrrep( index + 1 + cnt2 ), movingRight,
                                                   denBKDT, denBK, prob, index + 1 + cnt2 );
         QtensorsT[ index ][ cnt2 ] = new CTensorQT( index + 1, denBK->gIrrep( index + 1 + cnt2 ), movingRight,
                                                     denBKDT, denBK, prob, index + 1 + cnt2 );
      }

      // Xtensors : a certain process owns the Xtensors
      Xtensors[ index ] = new CTensorX( index + 1, movingRight, denBKDT, denBK, prob );

      // Otensors :
      Otensors[ index ] = new CTensorO( index + 1, movingRight, denBKDT, denBK );
   } else {
      Ltensors[ index ]  = new CTensorL *[ L - 1 - index ];
      LtensorsT[ index ] = new CTensorLT *[ L - 1 - index ];
      for ( int cnt2 = 0; cnt2 < L - 1 - index; cnt2++ ) {
         Ltensors[ index ][ cnt2 ]  = new CTensorL( index + 1, denBK->gIrrep( index + 1 + cnt2 ), movingRight, denBKDT, denBK );
         LtensorsT[ index ][ cnt2 ] = new CTensorLT( index + 1, denBK->gIrrep( index + 1 + cnt2 ), movingRight, denBKDT, denBK );
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
            S1tensors[ index ][ cnt2 ] = new CTensorS1 *[ L - 1 - index - cnt2 ];
         }
         if ( cnt2 > 0 ) {
            S1tensorsT[ index ][ cnt2 ] = new CTensorS1T *[ L - 1 - index - cnt2 ];
         }
         for ( int cnt3 = 0; cnt3 < L - 1 - index - cnt2; cnt3++ ) {
            const int Iprod                     = Irreps::directProd( denBKDT->gIrrep( index + 1 + cnt3 ),
                                                  denBKDT->gIrrep( index + 1 + cnt2 + cnt3 ) );
            F0tensors[ index ][ cnt2 ][ cnt3 ]  = new CTensorF0( index + 1, Iprod, movingRight, denBKDT, denBK );
            F0tensorsT[ index ][ cnt2 ][ cnt3 ] = new CTensorF0T( index + 1, Iprod, movingRight, denBKDT, denBK );
            F1tensors[ index ][ cnt2 ][ cnt3 ]  = new CTensorF1( index + 1, Iprod, movingRight, denBKDT, denBK );
            F1tensorsT[ index ][ cnt2 ][ cnt3 ] = new CTensorF1T( index + 1, Iprod, movingRight, denBKDT, denBK );
            S0tensors[ index ][ cnt2 ][ cnt3 ]  = new CTensorS0( index + 1, Iprod, movingRight, denBKDT, denBK );
            S0tensorsT[ index ][ cnt2 ][ cnt3 ] = new CTensorS0T( index + 1, Iprod, movingRight, denBKDT, denBK );
            if ( cnt2 > 0 ) {
               S1tensors[ index ][ cnt2 ][ cnt3 ] = new CTensorS1( index + 1, Iprod, movingRight, denBKDT, denBK );
            }
            if ( cnt2 > 0 ) {
               S1tensorsT[ index ][ cnt2 ][ cnt3 ] = new CTensorS1T( index + 1, Iprod, movingRight, denBKDT, denBK );
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
            Btensors[ index ][ cnt2 ] = new CTensorOperator *[ index + 1 - cnt2 ];
         }
         if ( cnt2 > 0 ) {
            BtensorsT[ index ][ cnt2 ] = new CTensorOperator *[ index + 1 - cnt2 ];
         }
         Ctensors[ index ][ cnt2 ]  = new CTensorOperator *[ index + 1 - cnt2 ];
         CtensorsT[ index ][ cnt2 ] = new CTensorOperator *[ index + 1 - cnt2 ];
         Dtensors[ index ][ cnt2 ]  = new CTensorOperator *[ index + 1 - cnt2 ];
         DtensorsT[ index ][ cnt2 ] = new CTensorOperator *[ index + 1 - cnt2 ];
         for ( int cnt3 = 0; cnt3 < index + 1 - cnt2; cnt3++ ) {
            const int Idiff = CheMPS2::Irreps::directProd(
                denBK->gIrrep( index - cnt2 - cnt3 ), denBK->gIrrep( index - cnt3 ) );
            Atensors[ index ][ cnt2 ][ cnt3 ]  = new CTensorOperator( index + 1, 0, 2, Idiff, movingRight, true, false, denBKDT, denBK );
            AtensorsT[ index ][ cnt2 ][ cnt3 ] = new CTensorOperator( index + 1, 0, -2, Idiff, movingRight, false, false, denBKDT, denBK );
            if ( cnt2 > 0 ) {
               Btensors[ index ][ cnt2 ][ cnt3 ] = new CTensorOperator( index + 1, 2, 2, Idiff, movingRight, true, false, denBKDT, denBK );
            }
            if ( cnt2 > 0 ) {
               BtensorsT[ index ][ cnt2 ][ cnt3 ] = new CTensorOperator( index + 1, 2, -2, Idiff, movingRight, false,
                                                                         false, denBKDT, denBK );
            }
            Ctensors[ index ][ cnt2 ][ cnt3 ]  = new CTensorOperator( index + 1, 0, 0, Idiff, movingRight, true, false, denBKDT, denBK );
            CtensorsT[ index ][ cnt2 ][ cnt3 ] = new CTensorOperator( index + 1, 0, 0, Idiff, movingRight, false, false, denBKDT, denBK );
            Dtensors[ index ][ cnt2 ][ cnt3 ]  = new CTensorOperator( index + 1, 2, 0, Idiff, movingRight, movingRight,
                                                                     false, denBKDT, denBK );
            DtensorsT[ index ][ cnt2 ][ cnt3 ] = new CTensorOperator( index + 1, 2, 0, Idiff, movingRight,
                                                                      !movingRight, false, denBKDT, denBK );
         }
      }

      // QQtensors  : certain processes own certain QQtensors
      // To left: Qtens[ cnt][ cnt2 ] = operator on site cnt-cnt2; at boundary
      // cnt+1
      Qtensors[ index ]  = new CTensorQ *[ index + 1 ];
      QtensorsT[ index ] = new CTensorQT *[ index + 1 ];
      for ( int cnt2 = 0; cnt2 < index + 1; cnt2++ ) {
         Qtensors[ index ][ cnt2 ]  = new CTensorQ( index + 1, denBK->gIrrep( index - cnt2 ), movingRight,
                                                   denBKDT, denBK, prob, index - cnt2 );
         QtensorsT[ index ][ cnt2 ] = new CTensorQT( index + 1, denBK->gIrrep( index - cnt2 ), movingRight,
                                                     denBKDT, denBK, prob, index - cnt2 );
      }

      // Xtensors : a certain process owns the Xtensors
      Xtensors[ index ] = new CTensorX( index + 1, movingRight, denBKDT, denBK, prob );

      // Otensors :
      Otensors[ index ] = new CTensorO( index + 1, movingRight, denBKDT, denBK );
   }
}

void CheMPS2::TimeTaylor::deleteTensors( const int index, const bool movingRight ) {
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

void CheMPS2::TimeTaylor::doStep( const int currentInstruction ) {

   for ( int i = 0; i < scheme->get_max_sweeps( currentInstruction ); ++i ) {
      for ( int site = L - 2; site > 0; site-- ) {
         CSobject * denSB = new CSobject( site, denBK );
         denSB->Join( MPS[ site ], MPS[ site + 1 ] );

         CSobject * denPA         = new CSobject( site, denBKDT );
         CTensorO * leftOverlapA  = ( site - 1 ) >= 0 ? Otensors[ site - 1 ] : NULL;
         CTensorO * rightOverlapA = ( site + 2 ) < L ? Otensors[ site + 1 ] : NULL;
         denPA->Join( leftOverlapA, denSB, rightOverlapA );

         CSobject * denPB = new CSobject( site, denBKDT );

         CHeffNS * heff = new CHeffNS( denBKDT, denBK, prob );
         heff->Apply( denSB, denPB, Ltensors, LtensorsT, Atensors, AtensorsT,
                      Btensors, BtensorsT, Ctensors, CtensorsT, Dtensors, DtensorsT,
                      S0tensors, S0tensorsT, S1tensors, S1tensorsT, F0tensors,
                      F0tensorsT, F1tensors, F1tensorsT, Qtensors, QtensorsT,
                      Xtensors, leftOverlapA, rightOverlapA );

         denPA->Add( -scheme->get_time_step( currentInstruction ), denPB );

         double disc = denPA->Split( MPSDT[ site ], MPSDT[ site + 1 ], scheme->get_D( currentInstruction ), scheme->get_cut_off( currentInstruction ), false, true );

         delete heff;
         delete denPA;
         delete denPB;
         delete denSB;

         updateMovingLeftSafe( site );
      }

      for ( int site = 0; site < L - 2; site++ ) {
         CSobject * denSB = new CSobject( site, denBK );
         denSB->Join( MPS[ site ], MPS[ site + 1 ] );

         CSobject * denPA         = new CSobject( site, denBKDT );
         CTensorO * leftOverlapA  = ( site - 1 ) >= 0 ? Otensors[ site - 1 ] : NULL;
         CTensorO * rightOverlapA = ( site + 2 ) < L ? Otensors[ site + 1 ] : NULL;
         denPA->Join( leftOverlapA, denSB, rightOverlapA );

         CSobject * denPB = new CSobject( site, denBKDT );

         CHeffNS * heff = new CHeffNS( denBKDT, denBK, prob );
         heff->Apply( denSB, denPB, Ltensors, LtensorsT, Atensors, AtensorsT,
                      Btensors, BtensorsT, Ctensors, CtensorsT, Dtensors, DtensorsT,
                      S0tensors, S0tensorsT, S1tensors, S1tensorsT, F0tensors,
                      F0tensorsT, F1tensors, F1tensorsT, Qtensors, QtensorsT,
                      Xtensors, leftOverlapA, rightOverlapA );

         denPA->Add( -scheme->get_time_step( currentInstruction ), denPB );

         double disc = denPA->Split( MPSDT[ site ], MPSDT[ site + 1 ], scheme->get_D( currentInstruction ), scheme->get_cut_off( currentInstruction ), true, true );

         delete heff;
         delete denPA;
         delete denPB;
         delete denSB;

         updateMovingRightSafe( site );
      }
   }
}

double CheMPS2::TimeTaylor::Energy() {

   deleteAllBoundaryOperators();

   denBKDT = denBK;
   MPSDT   = MPS;

   for ( int cnt = 0; cnt < L - 1; cnt++ ) {
      updateMovingRightSafe( cnt );
   }
   CTensorX * myX = new CTensorX( L, true, denBKDT, denBK, prob );

   myX->update( MPSDT[ L - 1 ], MPS[ L - 1 ], Otensors[ L - 2 ], Ltensors[ L - 2 ],
                LtensorsT[ L - 2 ], Xtensors[ L - 2 ], Qtensors[ L - 2 ][ 0 ],
                QtensorsT[ L - 2 ][ 0 ], Atensors[ L - 2 ][ 0 ][ 0 ],
                AtensorsT[ L - 2 ][ 0 ][ 0 ], CtensorsT[ L - 2 ][ 0 ][ 0 ],
                DtensorsT[ L - 2 ][ 0 ][ 0 ] );

   dcomplex result = myX->gStorage()[ 0 ];
   delete myX;
   return std::real( result );
   return 0.0;
}

void CheMPS2::TimeTaylor::Propagate() {

   double t = 0.0;
   for ( int inst = 0; inst < scheme->get_number(); inst++ ) {

      for ( ; t < scheme->get_max_time( inst ); t += scheme->get_time_step( inst ) ) {
         ( *logger ) << "t = " << t << " " << Energy();

         for ( int i = 0; i < L + 1; i++ ) {
            ( *logger ) << " " << denBK->gTotDimAtBound( i );
         }
         ( *logger ) << "\n";

         deleteAllBoundaryOperators();

         denBKDT = new SyBookkeeper( prob, scheme->get_D( inst ) );
         MPSDT   = new CTensorT *[ L ];
         for ( int index = 0; index < L; index++ ) {
            MPSDT[ index ] = new CTensorT( index, denBKDT );
            MPSDT[ index ]->random();
         }

         for ( int cnt = 0; cnt < L - 1; cnt++ ) {
            updateMovingRightSafe( cnt );
         }

         doStep( inst );

         for ( int site = 0; site < L; site++ ) {
            delete MPS[ site ];
         }
         delete[] MPS;
         delete denBK;

         double normDT = norm( MPSDT );
         MPSDT[ 0 ]->number_operator( 0.0, 1.0 / normDT );

         MPS   = MPSDT;
         denBK = denBKDT;
      }
   }
}
