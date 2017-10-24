
#ifndef CTENSORQ_CHEMPS2_H
#define CTENSORQ_CHEMPS2_H

#include "CTensorF0.h"
#include "CTensorF1.h"
#include "CTensorL.h"
#include "CTensorLT.h"
#include "CTensorOperator.h"
#include "CTensorT.h"
#include "Problem.h"
#include "SyBookkeeper.h"

namespace CheMPS2 {
   /** CTensorQ class.
    \author Lars-Hendrik Frahm
    \date September 15, 2017

    The CTensorQ class is a storage and manipulation class for the complementary
   operator of three contracted creators/annihilitors. */
   class CTensorQ : public CTensorOperator {
      public:
      //! Constructor
      /** \param boundary_index The boundary index
      \param Idiff The irrep of the one creator ( sandwiched if TensorL ; to
     sandwich if CTensorQ )
      \param moving_right If true: sweep from left to right. If false: sweep
     from right to left
      \param denBK Symmetry bookkeeper of the problem at hand
      \param Prob Problem containing the matrix elements
      \param site The site on which the last crea/annih should work */
      CTensorQ( const int boundary_index, const int Idiff, const bool moving_right, const SyBookkeeper * book_up, const SyBookkeeper * book_down, const Problem * Prob, const int site );

      //! Destructor
      virtual ~CTensorQ();

      //! Add terms after update/clear without previous tensors
      /** \param denT TensorT to construct the Q-term without previous tensors */
      void AddTermSimple( CTensorT * denTup, CTensorT * denTdown, CTensorO * previous, dcomplex * workmem );

      //! Add terms after update/clear with previous TensorL's
      /** \param Ltensors The TensorL's to construct the Q-term
      \param denT TensorT to construct the Q-term with previous TensorL's
      \param workmem Work memory
      \param workmem2 Work memory */
      void AddTermsL( CTensorL ** Ltensors, CTensorLT ** LtensorsT, CTensorT * denTup, CTensorT * denTdown, dcomplex * workmem, dcomplex * workmem2 );

      //! Add terms after update/clear with previous A-tensors and B-tensors
      /** \param denA The A-tensor to construct the Q-term
      \param denB The B-tensor to construct the Q-term
      \param denT TensorT to construct the Q-term with previous TensorL's
      \param workmem Work memory
      \param workmem2 Work memory */
      void AddTermsAB( CTensorOperator * denA, CTensorOperator * denB, CTensorT * denTup, CTensorT * denTdown, dcomplex * workmem, dcomplex * workmem2 );

      //! Add terms after update/clear with previous C-tensors and D-tensors
      /** \param denC The C-tensor to construct the Q-term
      \param denD The D-tensor to construct the Q-term
      \param denT TensorT to construct the Q-term with previous TensorL's
      \param workmem Work memory
      \param workmem2 Work memory */
      void AddTermsCD( CTensorOperator * denC, CTensorOperator * denD, CTensorT * denTup, CTensorT * denTdown, dcomplex * workmem, dcomplex * workmem2 );

      private:
      //! Pointer to the problem (contains the matrix elements)
      const Problem * Prob;

      //! Site on which the last crea/annih works
      const int site;

      // Internal stuff
      void AddTermSimpleRight( CTensorT * denTup, CTensorT * denTdown, CTensorO * previous, dcomplex * workmem );

      void AddTermSimpleLeft( CTensorT * denTup, CTensorT * denTdown, CTensorO * previous, dcomplex * workmem );

      void AddTermsLRight( CTensorL ** Ltensors, CTensorLT ** LtensorsT, CTensorT * denTup, CTensorT * denTdown, dcomplex * workmem, dcomplex * workmem2 );

      void AddTermsLLeft( CTensorL ** Ltensors, CTensorLT ** LtensorsT, CTensorT * denTup, CTensorT * denTdown, dcomplex * workmem, dcomplex * workmem2 );

      void AddTermsABRight( CTensorOperator * denA, CTensorOperator * denB, CTensorT * denTup, CTensorT * denTdown, dcomplex * workmem, dcomplex * workmem2 );

      void AddTermsABLeft( CTensorOperator * denA, CTensorOperator * denB, CTensorT * denTup, CTensorT * denTdown, dcomplex * workmem, dcomplex * workmem2 );

      void AddTermsCDRight( CTensorOperator * denC, CTensorOperator * denD, CTensorT * denTup, CTensorT * denTdown, dcomplex * workmem, dcomplex * workmem2 );

      void AddTermsCDLeft( CTensorOperator * denC, CTensorOperator * denD, CTensorT * denTup, CTensorT * denTdown, dcomplex * workmem, dcomplex * workmem2 );
   };
}

#endif
