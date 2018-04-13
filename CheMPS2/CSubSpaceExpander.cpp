
#include <algorithm>
#include <assert.h>
#include <iostream>
#include <math.h>
#include <stdlib.h>

#include "CSubSpaceExpander.h"
#include "Davidson.h"
#include "Lapack.h"
#include "MPIchemps2.h"
#include "Wigner.h"

CheMPS2::CSubSpaceExpander::CSubSpaceExpander( const int siteIn, const bool movingRightIn, const SyBookkeeper * initBKUpIn, const SyBookkeeper * initBKDownIn, const SyBookkeeper * sseBKDownIn, const Problem * ProbIn )
    : site( siteIn ), movingRight( movingRightIn ), initBKUp( initBKUpIn ), initBKDown( initBKDownIn ), sseBKDown( sseBKDownIn ), prob( ProbIn ) {}

CheMPS2::CSubSpaceExpander::~CSubSpaceExpander() {}

void CheMPS2::CSubSpaceExpander::Expand( double * noise,
                                         CTensorT * in, CTensorT * out,
                                         CTensorL *** Ltensors, CTensorLT *** LtensorsT,
                                         CTensorOperator **** Atensors, CTensorOperator **** AtensorsT,
                                         CTensorOperator **** Btensors, CTensorOperator **** BtensorsT,
                                         CTensorOperator **** Ctensors, CTensorOperator **** CtensorsT,
                                         CTensorOperator **** Dtensors, CTensorOperator **** DtensorsT,
                                         CTensorS0 **** S0tensors, CTensorS0T **** S0Ttensors,
                                         CTensorS1 **** S1tensors, CTensorS1T **** S1Ttensors,
                                         CTensorF0 **** F0tensors, CTensorF0T **** F0Ttensors,
                                         CTensorF1 **** F1tensors, CTensorF1T **** F1Ttensors,
                                         CTensorQ *** Qtensors, CTensorQT *** QtensorsT,
                                         CTensorX ** Xtensors, CTensorO ** Otensors ) {
   out->Clear();

   if ( movingRight ) {
      ApplyRight( in, out, Ltensors, LtensorsT, Atensors, AtensorsT, Btensors, BtensorsT,
                  Ctensors, CtensorsT, Dtensors, DtensorsT, S0tensors, S0Ttensors, S1tensors,
                  S1Ttensors, F0tensors, F0Ttensors, F1tensors, F1Ttensors, Qtensors,
                  QtensorsT, Xtensors, Otensors );
   } else {
      ApplyLeft( in, out, Ltensors, LtensorsT, Atensors, AtensorsT, Btensors, BtensorsT,
                 Ctensors, CtensorsT, Dtensors, DtensorsT, S0tensors, S0Ttensors, S1tensors,
                 S1Ttensors, F0tensors, F0Ttensors, F1tensors, F1Ttensors, Qtensors,
                 QtensorsT, Xtensors, Otensors );
   }

   out->scale( *noise );
}

void CheMPS2::CSubSpaceExpander::ApplyRight( CTensorT * in, CTensorT * out,
                                             CTensorL *** Ltensors, CTensorLT *** LtensorsT,
                                             CTensorOperator **** Atensors, CTensorOperator **** AtensorsT,
                                             CTensorOperator **** Btensors, CTensorOperator **** BtensorsT,
                                             CTensorOperator **** Ctensors, CTensorOperator **** CtensorsT,
                                             CTensorOperator **** Dtensors, CTensorOperator **** DtensorsT,
                                             CTensorS0 **** S0tensors, CTensorS0T **** S0Ttensors,
                                             CTensorS1 **** S1tensors, CTensorS1T **** S1Ttensors,
                                             CTensorF0 **** F0tensors, CTensorF0T **** F0Ttensors,
                                             CTensorF1 **** F1tensors, CTensorF1T **** F1Ttensors,
                                             CTensorQ *** Qtensors, CTensorQT *** QtensorsT,
                                             CTensorX ** Xtensors, CTensorO ** Otensors ) {
   std::cout << "hi right" << std::endl;
}

void CheMPS2::CSubSpaceExpander::ApplyLeft( CTensorT * in, CTensorT * out,
                                            CTensorL *** Ltensors, CTensorLT *** LtensorsT,
                                            CTensorOperator **** Atensors, CTensorOperator **** AtensorsT,
                                            CTensorOperator **** Btensors, CTensorOperator **** BtensorsT,
                                            CTensorOperator **** Ctensors, CTensorOperator **** CtensorsT,
                                            CTensorOperator **** Dtensors, CTensorOperator **** DtensorsT,
                                            CTensorS0 **** S0tensors, CTensorS0T **** S0Ttensors,
                                            CTensorS1 **** S1tensors, CTensorS1T **** S1Ttensors,
                                            CTensorF0 **** F0tensors, CTensorF0T **** F0Ttensors,
                                            CTensorF1 **** F1tensors, CTensorF1T **** F1Ttensors,
                                            CTensorQ *** Qtensors, CTensorQT *** QtensorsT,
                                            CTensorX ** Xtensors, CTensorO ** Otensors ) {

   const int index    = site;
   const bool atLeft  = ( index == 0 ) ? true : false;
   const bool atRight = ( index == prob->gL() - 1 ) ? true : false;

   const int DIM_up   = std::max( initBKUp->gMaxDimAtBound( index ), initBKUp->gMaxDimAtBound( index + 1 ) );
   const int DIM_down = std::max( std::max( initBKDown->gMaxDimAtBound( index ), initBKDown->gMaxDimAtBound( index + 1 ) ), std::max( sseBKDown->gMaxDimAtBound( index ), sseBKDown->gMaxDimAtBound( index + 1 ) ) );

   char cotrans = 'C';
   char notrans = 'N';

   // // PARALLEL
   // #pragma omp parallel
   {
      dcomplex * temp  = new dcomplex[ DIM_up * DIM_down ];
      dcomplex * temp2 = new dcomplex[ DIM_up * DIM_down ];

      // #pragma omp for schedule( dynamic )
      for ( int ikappa = 0; ikappa < out->gNKappa(); ikappa++ ) {

         const int NL    = out->gNL( ikappa );
         const int TwoSL = out->gTwoSL( ikappa );
         const int IL    = out->gIL( ikappa );

         const int NR    = out->gNR( ikappa );
         const int TwoSR = out->gTwoSR( ikappa );
         const int IR    = out->gIR( ikappa );

         const int N    = NR - NL;
         const int TwoS = ( N == 1 ) ? 1 : 0;
         const int I    = initBKUp->gIrrep( index );

         int dimRU = initBKUp->gCurrentDim( index + 1, NR, TwoSR, IR );
         int dimRD = initBKDown->gCurrentDim( index + 1, NR, TwoSR, IR );
         int dimLD = sseBKDown->gCurrentDim( index, NL, TwoSL, IL );

         dcomplex * memLDxRU = new dcomplex[ dimLD * dimRU ];
         dcomplex * memLDxRD = new dcomplex[ dimLD * dimRD ];
         for ( int cnt = 0; cnt < dimLD * dimRD; cnt++ ) {
            memLDxRD[ cnt ] = 0.0;
         }
         for ( int cnt = 0; cnt < dimLD * dimRU; cnt++ ) {
            memLDxRU[ cnt ] = 0.0;
         }

         addDiagram0ALeft( ikappa, memLDxRD, in, out, prob->gEconst() );
         addDiagram1C( ikappa, memLDxRD, in, out, prob->gMxElement( index, index, index, index ) );

         if ( !atLeft ) {
            /*********************
            *  Diagrams group 1  *
            *********************/
            addDiagram1A( ikappa, memLDxRD, in, out, Xtensors[ index - 1 ] );

            /*********************
            *  Diagrams group 2  *
            *********************/
            addDiagram2b1and2b2Left( ikappa, memLDxRD, in, out, Atensors[ index - 1 ][ 0 ][ 0 ], AtensorsT[ index - 1 ][ 0 ][ 0 ] );
            addDiagram2b3spin0Left( ikappa, memLDxRD, in, out, CtensorsT[ index - 1 ][ 0 ][ 0 ] );
            addDiagram2b3spin1Left( ikappa, memLDxRD, in, out, DtensorsT[ index - 1 ][ 0 ][ 0 ] );

            /*********************
            *  Diagrams group 3  *
            *********************/
            addDiagram3Aand3DLeft( ikappa, memLDxRD, in, out, Qtensors[ index - 1 ][ 0 ], QtensorsT[ index - 1 ][ 0 ], Ltensors[ index - 1 ], LtensorsT[ index - 1 ], temp );
         }

         if ( ( !atLeft ) && ( !atRight ) ) {

            /*********************
            *  Diagrams group 2  *
            *********************/
            addDiagram2a1spin0Left( ikappa, memLDxRU, in, out, AtensorsT, S0Ttensors, temp );
            addDiagram2a2spin0Left( ikappa, memLDxRU, in, out, Atensors, S0tensors, temp );
            addDiagram2a1spin1Left( ikappa, memLDxRU, in, out, BtensorsT, S1Ttensors, temp );
            addDiagram2a2spin1Left( ikappa, memLDxRU, in, out, Btensors, S1tensors, temp );
            addDiagram2a3spin0Left( ikappa, memLDxRU, in, out, Ctensors, CtensorsT, F0tensors, F0Ttensors, temp );
            addDiagram2a3spin1Left( ikappa, memLDxRU, in, out, Dtensors, DtensorsT, F1tensors, F1Ttensors, temp );

            /*********************
            *  Diagrams group 3  *
            *********************/
            addDiagram3CLeft( ikappa, memLDxRU, in, out, Qtensors[ index - 1 ], QtensorsT[ index - 1 ], Ltensors[ index ], LtensorsT[ index ], temp );
            addDiagram3JLeft( ikappa, memLDxRU, in, out, Qtensors[ index ], QtensorsT[ index ], Ltensors[ index - 1 ], LtensorsT[ index - 1 ], temp );

            /*********************
            *  Diagrams group 4  *
            *********************/
            addDiagram4B1and4B2spin0Left( ikappa, memLDxRU, in, out, Atensors[ index - 1 ], AtensorsT[ index - 1 ], Ltensors[ index ], LtensorsT[ index ], temp );
            addDiagram4B1and4B2spin1Left( ikappa, memLDxRU, in, out, Btensors[ index - 1 ], BtensorsT[ index - 1 ], Ltensors[ index ], LtensorsT[ index ], temp );
            addDiagram4B3and4B4spin0Left( ikappa, memLDxRU, in, out, Ctensors[ index - 1 ], CtensorsT[ index - 1 ], Ltensors[ index ], LtensorsT[ index ], temp );
            addDiagram4B3and4B4spin1Left( ikappa, memLDxRU, in, out, Dtensors[ index - 1 ], DtensorsT[ index - 1 ], Ltensors[ index ], LtensorsT[ index ], temp );
            addDiagram4ELeft( ikappa, memLDxRU, in, out, Ltensors[ index - 1 ], LtensorsT[ index - 1 ], Ltensors[ index ], LtensorsT[ index ], temp, temp2 );
            addDiagram4L1and4L2spin0Left( ikappa, memLDxRU, in, out, Ltensors[ index - 1 ], LtensorsT[ index - 1 ], Atensors[ index ], AtensorsT[ index ], temp );
            addDiagram4L3and4L4spin0Left( ikappa, memLDxRU, in, out, Ltensors[ index - 1 ], LtensorsT[ index - 1 ], Ctensors[ index ], CtensorsT[ index ], temp );
            addDiagram4L1and4L2spin1Left( ikappa, memLDxRU, in, out, Ltensors[ index - 1 ], LtensorsT[ index - 1 ], Btensors[ index ], BtensorsT[ index ], temp );
            addDiagram4L3and4L4spin1Left( ikappa, memLDxRU, in, out, Ltensors[ index - 1 ], LtensorsT[ index - 1 ], Dtensors[ index ], DtensorsT[ index ], temp );
         }

         if ( !atRight && dimLD > 0 ) {

            /*********************
            *  Diagrams group 1  *
            *********************/
            addDiagram1B( ikappa, memLDxRU, in, out, Xtensors[ index ] );

            /*********************
            *  Diagrams group 2  *
            *********************/
            addDiagram2e1and2e2Left( ikappa, memLDxRU, in, out, Atensors[ index ][ 0 ][ 0 ], AtensorsT[ index ][ 0 ][ 0 ] );
            addDiagram2e3spin0Left( ikappa, memLDxRU, in, out, CtensorsT[ index ][ 0 ][ 0 ] );
            addDiagram2e3spin1Left( ikappa, memLDxRU, in, out, DtensorsT[ index ][ 0 ][ 0 ] );

            /*********************
            *  Diagrams group 3  *
            *********************/
            addDiagram3Kand3FLeft( ikappa, memLDxRU, in, out, Qtensors[ index ][ 0 ], QtensorsT[ index ][ 0 ], Ltensors[ index ], LtensorsT[ index ], temp );
         }

         int dimLDxRU = dimLD * dimRU;
         int dimLDxRD = dimLD * dimRD;

         dcomplex * BlockOut = out->gStorage() + out->gKappa2index( ikappa );

         dcomplex one = 1.0;
         int inc      = 1;
         zaxpy_( &dimLDxRU, &one, memLDxRU, &inc, BlockOut, &inc );

         if ( atRight ) {
            int dimLDxRD = dimLD * dimRD;
            zaxpy_( &dimLDxRD, &one, memLDxRD, &inc, BlockOut, &inc );
         } else {
            dcomplex * BlockOT = Otensors[ index ]->gStorage( NR, TwoSR, IR, NR, TwoSR, IR );
            zgemm_( &notrans, &cotrans, &dimLD, &dimRU, &dimRD, &one, memLDxRD, &dimLD, BlockOT, &dimRU, &one, BlockOut, &dimLD );
         }

         delete[] memLDxRU;
         delete[] memLDxRD;
      }

      delete[] temp;
      delete[] temp2;
   }
}

void CheMPS2::CSubSpaceExpander::AddNonExpandedToExpanded( CTensorT * expanded, CTensorT * nonExpanded ) {
   assert( expanded->gIndex() == nonExpanded->gIndex() );

   const int index = expanded->gIndex();

   for ( int ikappa = 0; ikappa < expanded->gNKappa(); ikappa++ ) {

      const int NL    = expanded->gNL( ikappa );
      const int TwoSL = expanded->gTwoSL( ikappa );
      const int IL    = expanded->gIL( ikappa );

      const int NR    = expanded->gNR( ikappa );
      const int TwoSR = expanded->gTwoSR( ikappa );
      const int IR    = expanded->gIR( ikappa );

      const int N    = NR - NL;
      const int TwoS = ( N == 1 ) ? 1 : 0;
      const int I    = initBKUp->gIrrep( index );

      int memNonExpKappa = nonExpanded->gKappa( NL, TwoSL, IL, NR, TwoSR, IR );
      if ( memNonExpKappa != -1 ) {

         int dimLEXP = expanded->gBK()->gCurrentDim( index, NL, TwoSL, IL );
         int dimREXP = expanded->gBK()->gCurrentDim( index + 1, NR, TwoSR, IR );

         int dimLNON = nonExpanded->gBK()->gCurrentDim( index, NL, TwoSL, IL );
         int dimRNON = nonExpanded->gBK()->gCurrentDim( index + 1, NR, TwoSR, IR );

         // If nonExpanded has the block it has to have the same size
         assert( dimLEXP == dimLNON );
         assert( dimREXP == dimRNON );

         int dim      = dimLEXP * dimREXP;
         int inc      = 1;
         dcomplex one = 1.0;

         dcomplex * BlockExp = expanded->gStorage() + expanded->gKappa2index( ikappa );
         dcomplex * BlockNon = nonExpanded->gStorage() + nonExpanded->gKappa2index( memNonExpKappa );

         zaxpy_( &dim, &one, BlockNon, &inc, BlockExp, &inc );
      }
   }
}

void CheMPS2::CSubSpaceExpander::decomposeMovingLeft( CTensorT * expandedLeft, SyBookkeeper * expandedLeftBK,
                                                      CTensorT * expandedRight, SyBookkeeper * expandedRightBK,
                                                      CTensorT * newLeft, SyBookkeeper * newLeftBK,
                                                      CTensorT * newRight, SyBookkeeper * newRightBK ) {
   assert( expandedLeftBK == expandedRightBK );
   assert( newLeftBK == newRightBK );
   int index = site;

   // std::cout << "hi" << std::endl;

   // std::cout << *expandedLeft << std::endl;
   // std::cout << *expandedRight << std::endl;
   // std::cout << *newLeft << std::endl;
   // std::cout << *newRight << std::endl;

   // PARALLEL
   int nSectors = 0;
   for ( int NL = expandedRightBK->gNmin( index ); NL <= expandedRightBK->gNmax( index ); NL++ ) {
      for ( int TwoSL = expandedRightBK->gTwoSmin( index, NL ); TwoSL <= expandedRightBK->gTwoSmax( index, NL ); TwoSL += 2 ) {
         for ( int IL = 0; IL < expandedRightBK->getNumberOfIrreps(); IL++ ) {
            int dimL = expandedRightBK->gCurrentDim( index, NL, TwoSL, IL );
            if ( dimL > 0 ) {
               int dimRtotal = 0;
               for ( int ikappa = 0; ikappa < expandedRight->gNKappa(); ikappa++ ) {
                  if ( ( NL == expandedRight->gNL( ikappa ) ) && ( TwoSL == expandedRight->gTwoSL( ikappa ) ) && ( IL == expandedRight->gIL( ikappa ) ) ) {
                     dimRtotal += expandedRightBK->gCurrentDim( index + 1, expandedRight->gNR( ikappa ), expandedRight->gTwoSR( ikappa ), expandedRight->gIR( ikappa ) );
                  }
               }
               if ( dimRtotal > 0 ) {
                  nSectors++;
               }
            }
         }
      }
   }

   int * SplitSectNL    = new int[ nSectors ];
   int * SplitSectTwoJL = new int[ nSectors ];
   int * SplitSectIL    = new int[ nSectors ];

   nSectors = 0;
   for ( int NL = expandedRightBK->gNmin( index ); NL <= expandedRightBK->gNmax( index ); NL++ ) {
      for ( int TwoSL = expandedRightBK->gTwoSmin( index, NL ); TwoSL <= expandedRightBK->gTwoSmax( index, NL ); TwoSL += 2 ) {
         for ( int IL = 0; IL < expandedRightBK->getNumberOfIrreps(); IL++ ) {
            int dimL = expandedRightBK->gCurrentDim( index, NL, TwoSL, IL );
            if ( dimL > 0 ) {
               int dimRtotal = 0;
               for ( int ikappa = 0; ikappa < expandedRight->gNKappa(); ikappa++ ) {
                  if ( ( NL == expandedRight->gNL( ikappa ) ) && ( TwoSL == expandedRight->gTwoSL( ikappa ) ) && ( IL == expandedRight->gIL( ikappa ) ) ) {
                     dimRtotal += expandedRightBK->gCurrentDim( index + 1, expandedRight->gNR( ikappa ), expandedRight->gTwoSR( ikappa ), expandedRight->gIR( ikappa ) );
                  }
               }
               if ( dimRtotal > 0 ) {
                  SplitSectNL[ nSectors ]    = NL;
                  SplitSectTwoJL[ nSectors ] = TwoSL;
                  SplitSectIL[ nSectors ]    = IL;
                  nSectors++;
               }
            }
         }
      }
   }

   double ** Lambdas = NULL;
   dcomplex ** Us    = NULL;
   dcomplex ** VTs   = NULL;
   int * DimLs       = NULL;
   int * DimMs       = NULL;
   int * DimRs       = NULL;

   Lambdas = new double *[ nSectors ];
   Us      = new dcomplex *[ nSectors ];
   VTs     = new dcomplex *[ nSectors ];
   DimLs   = new int[ nSectors ];
   DimMs   = new int[ nSectors ];
   DimRs   = new int[ nSectors ];

   // PARALLEL
   nSectors = 0;
   for ( int NL = expandedRightBK->gNmin( index ); NL <= expandedRightBK->gNmax( index ); NL++ ) {
      for ( int TwoSL = expandedRightBK->gTwoSmin( index, NL ); TwoSL <= expandedRightBK->gTwoSmax( index, NL ); TwoSL += 2 ) {
         for ( int IL = 0; IL < expandedRightBK->getNumberOfIrreps(); IL++ ) {
            int dimL = expandedRightBK->gCurrentDim( index, NL, TwoSL, IL );
            if ( dimL > 0 ) {
               int dimRtotal = 0;
               for ( int ikappa = 0; ikappa < expandedRight->gNKappa(); ikappa++ ) {
                  if ( ( NL == expandedRight->gNL( ikappa ) ) && ( TwoSL == expandedRight->gTwoSL( ikappa ) ) && ( IL == expandedRight->gIL( ikappa ) ) ) {
                     dimRtotal += expandedRightBK->gCurrentDim( index + 1, expandedRight->gNR( ikappa ), expandedRight->gTwoSR( ikappa ), expandedRight->gIR( ikappa ) );
                  }
               }
               if ( dimRtotal > 0 ) {

                  DimLs[ nSectors ] = dimL;
                  DimRs[ nSectors ] = dimRtotal;
                  DimMs[ nSectors ] = std::min( DimLs[ nSectors ], DimRs[ nSectors ] );

                  int mn = std::min( DimLs[ nSectors ], DimRs[ nSectors ] );
                  int mx = std::max( DimLs[ nSectors ], DimRs[ nSectors ] );

                  Lambdas[ nSectors ] = new double[ mn ];
                  Us[ nSectors ]      = new dcomplex[ mn * DimLs[ nSectors ] ];
                  VTs[ nSectors ]     = new dcomplex[ mn * DimRs[ nSectors ] ];

                  dcomplex * mem = new dcomplex[ dimRtotal * dimL ];
                  // Copy the relevant parts from storage to mem & multiply with factor !!
                  int dimRtotal2 = 0;
                  for ( int ikappa = 0; ikappa < expandedRight->gNKappa(); ikappa++ ) {
                     if ( ( NL == expandedRight->gNL( ikappa ) ) && ( TwoSL == expandedRight->gTwoSL( ikappa ) ) && ( IL == expandedRight->gIL( ikappa ) ) ) {
                        int dimR = expandedRightBK->gCurrentDim( index + 1, expandedRight->gNR( ikappa ), expandedRight->gTwoSR( ikappa ), expandedRight->gIR( ikappa ) );
                        if ( dimR > 0 ) {
                           double factor = sqrt( ( expandedRight->gTwoSR( ikappa ) + 1.0 ) / ( TwoSL + 1.0 ) );
                           for ( int l = 0; l < dimL; l++ ) {
                              for ( int r = 0; r < dimR; r++ ) {
                                 mem[ l + dimL * ( dimRtotal2 + r ) ] = factor * expandedRight->gStorage()[ expandedRight->gKappa2index( ikappa ) + l + dimL * r ];
                              }
                           }
                           dimRtotal2 += dimR;
                        }
                     }
                  }

                  // Now mem contains sqrt((2jR+1)/(2jM+1)) * (TT)^{jM nM IM) --> SVD per
                  // central symmetry
                  char jobz       = 'S'; // M x min(M,N) in U and min(M,N) x N in VT
                  int lwork       = mn * mn + 3 * mn;
                  int lrwork      = std::max( 5 * mn * mn + 5 * mn, 2 * mx * mn + 2 * mn * mn + mn );
                  dcomplex * work = new dcomplex[ lwork ];
                  double * rwork  = new double[ lrwork ];
                  int * iwork     = new int[ 8 * mn ];
                  int info;

                  // dgesdd is not thread-safe in every implementation ( intel MKL is safe, Atlas is not safe )
                  zgesdd_( &jobz, DimLs + nSectors, DimRs + nSectors, mem, DimLs + nSectors,
                           Lambdas[ nSectors ], Us[ nSectors ], DimLs + nSectors, VTs[ nSectors ],
                           &mn, work, &lwork, rwork, iwork, &info );

                  delete[] work;
                  delete[] rwork;
                  delete[] iwork;
                  delete[] mem;
                  nSectors++;
               }
            }
         }
      }
   }

   // for ( int i = 0; i < nSectors; i++ ) {
   //    std::cout << Lambdas[ i ][ 0 ] << std::endl;
   // }

   bool change              = true;
   double virtualdimensionD = 10;
   double cut_off           = 1e-100;

   double discardedWeight = 0.0; // Only if change==true; will the discardedWeight be meaningful and different from zero.
   int updateSectors      = 0;
   int * NewDims          = NULL;

   // If change: determine new virtual dimensions.
   if ( change ) {
      NewDims = new int[ nSectors ];

      // First determine the total number of singular values
      int totalDimSVD = 0;
      for ( int iSector = 0; iSector < nSectors; iSector++ ) {
         NewDims[ iSector ] = DimMs[ iSector ];
         totalDimSVD += NewDims[ iSector ];
      }
      std::cout << totalDimSVD << std::endl;

      // Copy them all in 1 array
      double * values = new double[ totalDimSVD ];
      totalDimSVD     = 0;
      int inc         = 1;
      for ( int iSector = 0; iSector < nSectors; iSector++ ) {
         if ( NewDims[ iSector ] > 0 ) {
            dcopy_( NewDims + iSector, Lambdas[ iSector ], &inc, values + totalDimSVD, &inc );
            totalDimSVD += NewDims[ iSector ];
         }
      }

      // Sort them in decreasing order
      char ID = 'D';
      int info;
      dlasrt_( &ID, &totalDimSVD, values, &info ); // Quicksort

      int maxD = 0;
      while ( maxD < totalDimSVD && maxD < virtualdimensionD && cut_off <= values[ maxD ] ) {
         maxD++;
      }
      std::cout << maxD << std::endl;

      // int maxD = virtualdimensionD;
      // If larger then the required virtualdimensionD, new virtual dimensions
      // will be set in NewDims.
      if ( totalDimSVD > maxD ) {

         // The D+1'th value becomes the lower bound Schmidt value. Every value
         // smaller than or equal to the D+1'th value is thrown out (hence Dactual // <= Ddesired).
         const double lowerBound = values[ maxD ];
         for ( int iSector = 0; iSector < nSectors; iSector++ ) {
            for ( int cnt = 0; cnt < NewDims[ iSector ]; cnt++ ) {
               if ( Lambdas[ iSector ][ cnt ] <= lowerBound ) {
                  NewDims[ iSector ] = cnt;
               }
            }
         }

         // Discarded weight
         double totalSum     = 0.0;
         double discardedSum = 0.0;
         for ( int iSector = 0; iSector < nSectors; iSector++ ) {
            for ( int iLocal = 0; iLocal < std::min( DimLs[ iSector ], DimRs[ iSector ] ); iLocal++ ) {
               double temp = ( expandedRight->gTwoSL( iSector ) + 1 ) *
                             Lambdas[ iSector ][ iLocal ] * Lambdas[ iSector ][ iLocal ];
               totalSum += temp;
               if ( Lambdas[ iSector ][ iLocal ] <= lowerBound ) {
                  discardedSum += temp;
               }
            }
         }
         discardedWeight = discardedSum / totalSum;
      }
      // Clean-up
      delete[] values;

      std::cout << discardedWeight << std::endl;

      // Check if there is a sector which differs
      updateSectors = 0;
      for ( int iSector = 0; iSector < nSectors; iSector++ ) {
         const int MPSdim = expandedRightBK->gCurrentDim( index + 1, expandedRight->gNL( iSector ), expandedRight->gTwoSL( iSector ), expandedRight->gIL( iSector ) );
         if ( NewDims[ iSector ] != MPSdim ) {
            updateSectors = 1;
         }
      }
   }

   if ( updateSectors == 1 ) {
      for ( int iSector = 0; iSector < nSectors; iSector++ ) {
         newLeftBK->SetDim( index, expandedRight->gNL( iSector ), expandedRight->gTwoSL( iSector ), expandedRight->gIL( iSector ), NewDims[ iSector ] );
      }
      newRight->Reset();
      newLeft->Reset();
   }

   if ( NewDims != NULL ) {
      delete[] NewDims;
   }

   // Copy first dimM per central symmetry sector to the relevant parts
   for ( int iCenter = 0; iCenter < nSectors; iCenter++ ) {
      const int dimM = newLeftBK->gCurrentDim( index, SplitSectNL[ iCenter ], SplitSectTwoJL[ iCenter ], SplitSectIL[ iCenter ] );
      if ( dimM > 0 ) {
         // U-part: copy
         int dimLtotal2 = 0;
         for ( int NL = SplitSectNL[ iCenter ] - 2; NL <= SplitSectNL[ iCenter ]; NL++ ) {
            const int TwoS1 = ( ( NL + 1 == SplitSectNL[ iCenter ] ) ? 1 : 0 );
            for ( int TwoSL = SplitSectTwoJL[ iCenter ] - TwoS1; TwoSL <= SplitSectTwoJL[ iCenter ] + TwoS1; TwoSL += 2 ) {
               if ( TwoSL >= 0 ) {
                  const int IL   = ( ( TwoS1 == 1 ) ? Irreps::directProd( newRightBK->gIrrep( index ), SplitSectIL[ iCenter ] ) : SplitSectIL[ iCenter ] );
                  const int dimL = newLeftBK->gCurrentDim( index - 1, NL, TwoSL, IL );
                  if ( dimL > 0 ) {
                     dcomplex * TleftBlock           = newLeft->gStorage( NL, TwoSL, IL,
                                                                SplitSectNL[ iCenter ],
                                                                SplitSectTwoJL[ iCenter ],
                                                                SplitSectIL[ iCenter ] );
                     const int dimension_limit_right = std::min( dimM, DimMs[ iCenter ] );
                     for ( int r = 0; r < dimension_limit_right; r++ ) {
                        const dcomplex factor = Lambdas[ iCenter ][ r ];
                        for ( int l = 0; l < dimL; l++ ) {
                           TleftBlock[ l + dimL * r ] *= factor * Us[ iCenter ][ dimLtotal2 + l + DimLs[ iCenter ] * r ];
                        }
                     }
                     for ( int r = dimension_limit_right; r < dimM; r++ ) {
                        for ( int l = 0; l < dimL; l++ ) {
                           TleftBlock[ l + dimL * r ] *= 1.0;
                        }
                     }
                     dimLtotal2 += dimL;
                  }
               }
            }
         }

         // VT-part: copy
         int dimRtotal2 = 0;
         for ( int NR = SplitSectNL[ iCenter ]; NR <= SplitSectNL[ iCenter ] + 2; NR++ ) {
            const int TwoS2 = ( ( NR == SplitSectNL[ iCenter ] + 1 ) ? 1 : 0 );
            for ( int TwoSR = SplitSectTwoJL[ iCenter ] - TwoS2; TwoSR <= SplitSectTwoJL[ iCenter ] + TwoS2; TwoSR += 2 ) {
               if ( TwoSR >= 0 ) {
                  const int IR   = ( ( TwoS2 == 1 ) ? Irreps::directProd( newRightBK->gIrrep( index ), SplitSectIL[ iCenter ] ) : SplitSectIL[ iCenter ] );
                  const int dimR = newLeftBK->gCurrentDim( index + 1, NR, TwoSR, IR );
                  if ( dimR > 0 ) {
                     dcomplex * TrightBlock         = newRight->gStorage( SplitSectNL[ iCenter ], SplitSectTwoJL[ iCenter ],
                                                                  SplitSectIL[ iCenter ], NR, TwoSR, IR );
                     const int dimension_limit_left = std::min( dimM, DimMs[ iCenter ] );
                     const dcomplex factor_base     = sqrt( ( SplitSectTwoJL[ iCenter ] + 1.0 ) / ( TwoSR + 1 ) );
                     for ( int l = 0; l < dimension_limit_left; l++ ) {
                        const dcomplex factor = factor_base;
                        for ( int r = 0; r < dimR; r++ ) {
                           TrightBlock[ l + dimM * r ] = factor * VTs[ iCenter ][ l + DimMs[ iCenter ] * ( dimRtotal2 + r ) ];
                        }
                     }
                     for ( int r = 0; r < dimR; r++ ) {
                        for ( int l = dimension_limit_left; l < dimM; l++ ) {
                           TrightBlock[ l + dimM * r ] = 0.0;
                        }
                     }
                     dimRtotal2 += dimR;
                  }
               }
            }
         }
      }
   }

   // std::cout << *newRight << std::endl;

   // std::cout << newRight->CheckRightNormal() << std::endl;

   // abort();
}