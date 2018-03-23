
#ifndef CHEFFNS_1S_CHEMPS2_H
#define CHEFFNS_1S_CHEMPS2_H

#include "CSobject.h"
#include "CTensorF0.h"
#include "CTensorF0T.h"
#include "CTensorF1.h"
#include "CTensorF1T.h"
#include "CTensorL.h"
#include "CTensorO.h"
#include "CTensorOperator.h"
#include "CTensorQ.h"
#include "CTensorQT.h"
#include "CTensorS0.h"
#include "CTensorS0T.h"
#include "CTensorS1.h"
#include "CTensorS1T.h"
#include "CTensorX.h"
#include "Options.h"
#include "Problem.h"
#include "SyBookkeeper.h"

namespace CheMPS2 {
   /** CHeffNS_1S class.
    \author Lars-Hendrik Frahm
    \date September 19, 2017

    The CHeffNS_1S class contains the sparse eigensolver routines and effective
   Hamiltonian construction needed for the DMRG algorithm. */
   class CHeffNS_1S {
      public:
      //! Constructor
      /** \param denBKIn The SyBookkeeper to get the dimensions
      \param ProbIn The Problem that contains the Hamiltonian
      \param dvdson_rtol_in The residual tolerance for the DMRG Davidson
     iterations */
      CHeffNS_1S( const SyBookkeeper * bk_upIn, const SyBookkeeper * bk_downIn, const Problem * ProbIn );

      void Apply( CTensorT * in, CTensorT * out,
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
                  CTensorX ** Xtensors, CTensorO ** Otensors, bool moveRight );
      // void Apply( CSobject * denS, CSobject * denP,
      //             CTensorL *** Ltensors,
      //             CTensorLT *** LtensorsT, CTensorOperator **** Atensors,
      //             CTensorOperator **** AtensorsT, CTensorOperator **** Btensors,
      //             CTensorOperator **** BtensorsT, CTensorOperator **** Ctensors,
      //             CTensorOperator **** CtensorsT, CTensorOperator **** Dtensors,
      //             CTensorOperator **** DtensorsT, CTensorS0 **** S0tensors,
      //             CTensorS0T **** S0Ttensors, CTensorS1 **** S1tensors,
      //             CTensorS1T **** S1Ttensors, CTensorF0 **** F0tensors,
      //             CTensorF0T **** F0Ttensors, CTensorF1 **** F1tensors,
      //             CTensorF1T **** F1Ttensors, CTensorQ *** Qtensors,
      //             CTensorQT *** QtensorsT, CTensorX ** Xtensors, CTensorO * OtensorsL,
      //             CTensorO * OtensorsR );

      //! Phase function
      /** \param TwoTimesPower Twice the power of the phase (-1)^{power}
      \return The phase (-1)^{TwoTimesPower/2} */
      static int phase( const int TwoTimesPower ) {
         return ( ( ( TwoTimesPower / 2 ) % 2 ) != 0 ) ? -1 : 1;
      }

      //! Destructor
      virtual ~CHeffNS_1S();

      private:
      // The SyBookkeeper
      const CheMPS2::SyBookkeeper * bk_up;

      // The SyBookkeeper
      const CheMPS2::SyBookkeeper * bk_down;

      // The Problem (and hence Hamiltonian)
      const CheMPS2::Problem * Prob;

      // const dcomplex offsetEnergy;

      // void makeHeff( CSobject * denS, CSobject * denP, CTensorL *** Ltensors,
      //                CTensorLT *** LtensorsT, CTensorOperator **** Atensors,
      //                CTensorOperator **** AtensorsT, CTensorOperator **** Btensors,
      //                CTensorOperator **** BtensorsT, CTensorOperator **** Ctensors,
      //                CTensorOperator **** CtensorsT, CTensorOperator **** Dtensors,
      //                CTensorOperator **** DtensorsT, CTensorS0 **** S0tensors,
      //                CTensorS0T **** S0tensorsT, CTensorS1 **** S1tensors,
      //                CTensorS1T **** S1tensorsT, CTensorF0 **** F0tensors,
      //                CTensorF0T **** F0tensorsT, CTensorF1 **** F1tensors,
      //                CTensorF1T **** F1tensorsT, CTensorQ *** Qtensors,
      //                CTensorQT *** QtensorsT, CTensorX ** Xtensors, CTensorO * OtensorsL,
      //                CTensorO * OtensorsR );

      // //The diagrams: Type 0/5
      // void addDiagram0A( const int ikappa, dcomplex * memHeff, CSobject * denS, CSobject * denP, dcomplex Helem_links );
      void addDiagram0A( const int ikappa, dcomplex * memHeff, CTensorT * in, CTensorT * out, dcomplex off_set );

      // //The diagrams: Type 1/5
      void addDiagram1A( const int ikappa, dcomplex * memHeff, CTensorT * in, CTensorT * out, CTensorX * Xleft );
      void addDiagram1B( const int ikappa, dcomplex * memHeff, CTensorT * in, CTensorT * out, CTensorOperator * Xright );
      void addDiagram1C( const int ikappa, dcomplex * memHeff, CTensorT * in, CTensorT * out, dcomplex Helem );

      // The diagrams: Type 2/5
      void addDiagram2a1spin0( const int ikappa, dcomplex * memHeff, CTensorT * in, CTensorT * out, CTensorOperator **** ATtensors, CTensorS0T **** S0Ttensors, dcomplex * workspace );
      void addDiagram2a2spin0( const int ikappa, dcomplex * memHeff, CTensorT * in, CTensorT * out, CTensorOperator **** Atensors, CTensorS0 **** S0tensors, dcomplex * workspace );
      void addDiagram2a1spin1( const int ikappa, dcomplex * memHeff, CTensorT * in, CTensorT * out, CTensorOperator **** BTtensors, CTensorS1T **** S1Ttensors, dcomplex * workspace );
      void addDiagram2a2spin1( const int ikappa, dcomplex * memHeff, CTensorT * in, CTensorT * out, CTensorOperator **** Btensors, CTensorS1 **** S1tensors, dcomplex * workspace );
      void addDiagram2a3spin0( const int ikappa, dcomplex * memHeff, CTensorT * in, CTensorT * out, CTensorOperator **** Ctensors, CTensorOperator **** CTtensors, CTensorF0 **** F0tensors, CTensorF0T **** F0Ttensors, dcomplex * workspace );
      void addDiagram2a3spin1( const int ikappa, dcomplex * memHeff, CTensorT * in, CTensorT * out, CTensorOperator **** Dtensors, CTensorOperator **** DTtensors, CTensorF1 **** F1tensors, CTensorF1T **** F1Ttensors, dcomplex * workspace );
      void addDiagram2b1and2b2( const int ikappa, dcomplex * memHeff, CTensorT * in, CTensorT * out, CTensorOperator * Atensor, CTensorOperator * ATtensor );
      void addDiagram2e1and2e2( const int ikappa, dcomplex * memHeff, CTensorT * in, CTensorT * out, CTensorOperator * Atensor, CTensorOperator * ATtensor );
      void addDiagram2b3spin0( const int ikappa, dcomplex * memHeff, CTensorT * in, CTensorT * out, CTensorOperator * CTtensor );
      void addDiagram2e3spin0( const int ikappa, dcomplex * memHeff, CTensorT * in, CTensorT * out, CTensorOperator * CTtensor );
      void addDiagram2b3spin1( const int ikappa, dcomplex * memHeff, CTensorT * in, CTensorT * out, CTensorOperator * DTtensor );
      void addDiagram2e3spin1( const int ikappa, dcomplex * memHeff, CTensorT * in, CTensorT * out, CTensorOperator * DTtensor );

      //The diagrams: Type 3/5
      void addDiagram3Aand3D( const int ikappa, dcomplex * memHeff, CTensorT * in, CTensorT * out, CTensorQ * Qleft, CTensorQT * QTleft, CTensorL ** Lleft, CTensorLT ** LTleft, dcomplex * temp );
      void addDiagram3C( const int ikappa, dcomplex * memHeff, CTensorT * in, CTensorT * out, CTensorQ ** Qleft, CTensorQT ** QTleft, CTensorL ** Lright, CTensorLT ** LTright, dcomplex * temp );
      void addDiagram3Kand3F( const int ikappa, dcomplex * memHeff, CTensorT * in, CTensorT * out, CTensorQ * Qright, CTensorQT * QTright, CTensorL ** Lright, CTensorLT ** LTright, dcomplex * temp );
      void addDiagram3J( const int ikappa, dcomplex * memHeff, CTensorT * in, CTensorT * out, CTensorQ ** Qright, CTensorQT ** QTright, CTensorL ** Lleft, CTensorLT ** LTleft, dcomplex * temp );

      // The diagrams: Type 4/5
      void addDiagram4B1and4B2spin0( const int ikappa, dcomplex * memHeff, CTensorT * in, CTensorT * out, CTensorOperator *** Aleft, CTensorOperator *** ATleft, CTensorL ** Lright, CTensorLT ** LTright, dcomplex * temp );
      void addDiagram4B1and4B2spin1( const int ikappa, dcomplex * memHeff, CTensorT * in, CTensorT * out, CTensorOperator *** Bleft, CTensorOperator *** BTleft, CTensorL ** Lright, CTensorLT ** LTright, dcomplex * temp );
      void addDiagram4B3and4B4spin0( const int ikappa, dcomplex * memHeff, CTensorT * in, CTensorT * out, CTensorOperator *** Cleft, CTensorOperator *** CTleft, CTensorL ** Lright, CTensorLT ** LTright, dcomplex * temp );
      void addDiagram4B3and4B4spin1( const int ikappa, dcomplex * memHeff, CTensorT * in, CTensorT * out, CTensorOperator *** Dleft, CTensorOperator *** DTleft, CTensorL ** Lright, CTensorLT ** LTright, dcomplex * temp );
      // void addDiagram4C1and4C2spin0( const int ikappa, dcomplex * memHeff, CSobject * denS, CSobject * denP, CTensorOperator *** Aleft, CTensorOperator *** ATleft, CTensorL ** Lright, CTensorLT ** LTright, dcomplex * temp );
      // void addDiagram4C1and4C2spin1( const int ikappa, dcomplex * memHeff, CSobject * denS, CSobject * denP, CTensorOperator *** Bleft, CTensorOperator *** BTleft, CTensorL ** Lright, CTensorLT ** LTright, dcomplex * temp );
      // void addDiagram4C3and4C4spin0( const int ikappa, dcomplex * memHeff, CSobject * denS, CSobject * denP, CTensorOperator *** Cleft, CTensorOperator *** CTleft, CTensorL ** Lright, CTensorLT ** LTright, dcomplex * temp );
      // void addDiagram4C3and4C4spin1( const int ikappa, dcomplex * memHeff, CSobject * denS, CSobject * denP, CTensorOperator *** Dleft, CTensorOperator *** DTleft, CTensorL ** Lright, CTensorLT ** LTright, dcomplex * temp );
      // void addDiagram4D( const int ikappa, dcomplex * memHeff, CSobject * denS, CSobject * denP, CTensorL ** Lleft, CTensorLT ** LTleft, dcomplex * temp );
      void addDiagram4E( const int ikappa, dcomplex * memHeff, CTensorT * in, CTensorT * out, CTensorL ** Lleft, CTensorLT ** LTleft, CTensorL ** Lright, CTensorLT ** LTright, dcomplex * temp, dcomplex * temp2 );
      // void addDiagram4F( const int ikappa, dcomplex * memHeff, CSobject * denS, CSobject * denP, CTensorL ** Lright, CTensorLT ** LTright, dcomplex * temp );
      // void addDiagram4G( const int ikappa, dcomplex * memHeff, CSobject * denS, CSobject * denP, CTensorL ** Lright, CTensorLT ** LTright, dcomplex * temp );
      // void addDiagram4H( const int ikappa, dcomplex * memHeff, CSobject * denS, CSobject * denP, CTensorL ** Lleft, CTensorLT ** LTleft, CTensorL ** Lright, CTensorLT ** LTright, dcomplex * temp, dcomplex * temp2 );
      // void addDiagram4I( const int ikappa, dcomplex * memHeff, CSobject * denS, CSobject * denP, CTensorL ** Lleft, CTensorLT ** LTleft, dcomplex * temp );
      // void addDiagram4J1and4J2spin0( const int ikappa, dcomplex * memHeff, CSobject * denS, CSobject * denP, CTensorOperator * Aright, CTensorOperator * ATright );
      // void addDiagram4J1and4J2spin1( const int ikappa, dcomplex * memHeff, CSobject * denS, CSobject * denP, CTensorOperator * Bright, CTensorOperator * BTright );
      // void addDiagram4J3and4J4spin0( const int ikappa, dcomplex * memHeff, CSobject * denS, CSobject * denP, CTensorOperator * Cright, CTensorOperator * CTright );
      // void addDiagram4J3and4J4spin1( const int ikappa, dcomplex * memHeff, CSobject * denS, CSobject * denP, CTensorOperator * Dright, CTensorOperator * DTright );
      // void addDiagram4K1and4K2spin0( const int ikappa, dcomplex * memHeff, CSobject * denS, CSobject * denP, CTensorL ** Lleft, CTensorLT ** LTleft, CTensorOperator *** Aright, CTensorOperator *** ATright, dcomplex * temp );
      void addDiagram4L1and4L2spin0( const int ikappa, dcomplex * memHeff, CTensorT * in, CTensorT * out, CTensorL ** Lleft, CTensorLT ** LTleft, CTensorOperator *** Aright, CTensorOperator *** ATright, dcomplex * temp );
      // void addDiagram4K1and4K2spin1( const int ikappa, dcomplex * memHeff, CSobject * denS, CSobject * denP, CTensorL ** Lleft, CTensorLT ** LTleft, CTensorOperator *** Bright, CTensorOperator *** BTright, dcomplex * temp );
      void addDiagram4L1and4L2spin1( const int ikappa, dcomplex * memHeff, CTensorT * in, CTensorT * out, CTensorL ** Lleft, CTensorLT ** LTleft, CTensorOperator *** Bright, CTensorOperator *** BTright, dcomplex * temp );
      // void addDiagram4K3and4K4spin0( const int ikappa, dcomplex * memHeff, CSobject * denS, CSobject * denP, CTensorL ** Lleft, CTensorLT ** LTleft, CTensorOperator *** Cright, CTensorOperator *** CTright, dcomplex * temp );
      void addDiagram4L3and4L4spin0( const int ikappa, dcomplex * memHeff, CTensorT * in, CTensorT * out, CTensorL ** Lleft, CTensorLT ** LTleft, CTensorOperator *** Cright, CTensorOperator *** CTright, dcomplex * temp );
      // void addDiagram4K3and4K4spin1( const int ikappa, dcomplex * memHeff, CSobject * denS, CSobject * denP, CTensorL ** Lleft, CTensorLT ** LTleft, CTensorOperator *** Dright, CTensorOperator *** DTright, dcomplex * temp );
      void addDiagram4L3and4L4spin1( const int ikappa, dcomplex * memHeff, CTensorT * in, CTensorT * out, CTensorL ** Lleft, CTensorLT ** LTleft, CTensorOperator *** Dright, CTensorOperator *** DTright, dcomplex * temp );

      // // The diagrams: type 5/5
      // void addDiagram5A( const int ikappa, dcomplex * memHeff, CSobject * denS, CSobject * denP, CTensorL ** Lleft, CTensorLT ** LTleft, CTensorL ** Lright, CTensorLT ** LTright, dcomplex * temp, dcomplex * temp2 );
      // void addDiagram5B( const int ikappa, dcomplex * memHeff, CSobject * denS, CSobject * denP, CTensorL ** Lleft, CTensorLT ** LTleft, CTensorL ** Lright, CTensorLT ** LTright, dcomplex * temp, dcomplex * temp2 );
      // void addDiagram5C( const int ikappa, dcomplex * memHeff, CSobject * denS, CSobject * denP, CTensorL ** Lleft, CTensorLT ** LTleft, CTensorL ** Lright, CTensorLT ** LTright, dcomplex * temp, dcomplex * temp2 );
      // void addDiagram5D( const int ikappa, dcomplex * memHeff, CSobject * denS, CSobject * denP, CTensorL ** Lleft, CTensorLT ** LTleft, CTensorL ** Lright, CTensorLT ** LTright, dcomplex * temp, dcomplex * temp2 );
      // void addDiagram5E( const int ikappa, dcomplex * memHeff, CSobject * denS, CSobject * denP, CTensorL ** Lleft, CTensorLT ** LTleft, CTensorL ** Lright, CTensorLT ** LTright, dcomplex * temp, dcomplex * temp2 );
      // void addDiagram5F( const int ikappa, dcomplex * memHeff, CSobject * denS, CSobject * denP, CTensorL ** Lleft, CTensorLT ** LTleft, CTensorL ** Lright, CTensorLT ** LTright, dcomplex * temp, dcomplex * temp2 );
   };
} // namespace CheMPS2

#endif
