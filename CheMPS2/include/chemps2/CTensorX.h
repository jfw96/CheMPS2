#ifndef CTENSORX_CHEMPS2_H
#define CTENSORX_CHEMPS2_H

#include "CTensor.h"
#include "CTensorF0.h"
#include "CTensorF1.h"
#include "CTensorL.h"
#include "CTensorLT.h"
#include "CTensorOperator.h"
#include "CTensorQ.h"
#include "CTensorQT.h"
#include "Problem.h"
#include "SyBookkeeper.h"

namespace CheMPS2 {
   /** CTensorX class.
    \author Lars-Hendrik Frahm
    \date September 18, 2018

    The CTensorX class is a storage and manipulation class for completely
   contracted Hamiltonian terms. */
   class CTensorX : public CTensorOperator {
      public:
      //! Constructor
      /** \param boundary_index The boundary index
      \param moving_right If true: sweep from left to right. If false: sweep
     from right to left
      \param denBK The symmetry bookkeeper with symmetry sector virtual
     dimensions
      \param Prob The Problem containing the Hamiltonian matrix elements */
      CTensorX( const int boundary_index, const bool moving_right, const SyBookkeeper * bk_up,
                const SyBookkeeper * bk_down, const Problem * Prob );

      //! Destructor
      virtual ~CTensorX();

      //! Clear and add the relevant terms to the CTensorX
      /** \param denT TensorT from which the new CTensorX should be made
      \param Ltensors Array with the TensorL's
      \param Xtensor The previous CTensorX
      \param Qtensor The previous TensorQ
      \param Atensor The previous A-tensor
      \param Ctensor The previous C-tensor
      \param Dtensor The previous D-tensor */
      void update( CTensorT * denTup, CTensorT * denTdown, CTensorO * overlap, CTensorL ** Ltensors,
                   CTensorLT ** LtensorsT, CTensorOperator * Xtensor, CTensorQ * Qtensor, CTensorQT * QtensorT,
                   CTensorOperator * Atensor, CTensorOperator * ATtensor, CTensorOperator * CtensorT,
                   CTensorOperator * DtensorT );

      //! Clear and add the relevant terms to the CTensorX
      /** \param denT TensorT from which the new CTensorX should be made */
      void update( CTensorT * denTup, CTensorT * denTdown );

      private:
      // Problem containing the matrix elements
      const Problem * Prob;

      // helper functions
      void makenewRight( const int ikappa, CTensorT * denTup, CTensorT * denTdown, CTensorO * overlap, dcomplex * workmem );

      void makenewLeft( const int ikappa, CTensorT * denTup, CTensorT * denTdown, CTensorO * overlap, dcomplex * workmem );

      void addTermQLRight( const int ikappa, CTensorT * denTup, CTensorT * denTdown, CTensorL ** Lprev,
                           CTensorQ * Qprev, dcomplex * workmemRR, dcomplex * workmemLR, dcomplex * workmemLL );

      void addTermQTLTRight( const int ikappa, CTensorT * denTup, CTensorT * denTdown, CTensorLT ** LprevT,
                             CTensorQT * QprevT, dcomplex * workmemRR, dcomplex * workmemLR, dcomplex * workmemLL );

      void addTermQLLeft( const int ikappa, CTensorT * denTup, CTensorT * denTdown, CTensorL ** Lprev,
                          CTensorQ * Qprev, dcomplex * workmemLL, dcomplex * workmemLR, dcomplex * workmemRR );

      void addTermQTLTLeft( const int ikappa, CTensorT * denTup, CTensorT * denTdown, CTensorLT ** Lprev,
                            CTensorQT * Qprev, dcomplex * workmemLL, dcomplex * workmemLR, dcomplex * workmemRR );

      void addTermALeft( const int ikappa, CTensorT * denTup, CTensorT * denTdown,
                         CTensorOperator * Aprev, dcomplex * workmemLR, dcomplex * workmemLL );

      void addTermATLeft( const int ikappa, CTensorT * denTup, CTensorT * denTdown,
                          CTensorOperator * Aprev, dcomplex * workmemLR, dcomplex * workmemLL );

      void addTermARight( const int ikappa, CTensorT * denTup, CTensorT * denTdown,
                          CTensorOperator * Aprev, dcomplex * workmemRR, dcomplex * workmemLR );

      void addTermATRight( const int ikappa, CTensorT * denTup, CTensorT * denTdown,
                           CTensorOperator * Aprev, dcomplex * workmemRR, dcomplex * workmemLR );

      void addTermCRight( const int ikappa, CTensorT * denTup, CTensorT * denTdown,
                          CTensorOperator * denC, dcomplex * workmemLR );

      void addTermCLeft( const int ikappa, CTensorT * denTup, CTensorT * denTdown,
                         CTensorOperator * denC, dcomplex * workmemLR );

      void addTermDRight( const int ikappa, CTensorT * denTup, CTensorT * denTdown,
                          CTensorOperator * denD, dcomplex * workmemLR );

      void addTermDLeft( const int ikappa, CTensorT * denTup, CTensorT * denTdown,
                         CTensorOperator * denD, dcomplex * workmemLR );
   };
} // namespace CheMPS2

#endif
