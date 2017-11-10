
#ifndef CTENSORS1_CHEMPS2_H
#define CTENSORS1_CHEMPS2_H

#include "CTensor.h"
#include "CTensorL.h"
#include "CTensorOperator.h"
#include "CTensorT.h"
#include "SyBookkeeper.h"

namespace CheMPS2 {
   /** CTensorS1 class.
    \author Lars-Hendrik Frahm
    \date September 15, 2017

    The CTensorS1 class is a storage and manipulation class for the spin-1
   component of two contracted creators or two contracted annihilators. */
   class CTensorS1 : public CTensorOperator {
      public:
      //! Constructor
      /** \param boundary_index The boundary index
      \param Idiff Direct product of irreps of the two 2nd quantized operators;
     both sandwiched & to sandwich
      \param moving_right If true: sweep from left to right. If false: sweep
     from right to left
      \param denBK The symmetry sector partitioning */
      CTensorS1( const int boundary_index, const int Idiff, const bool moving_right, const SyBookkeeper * book_up, const SyBookkeeper * book_down );

      //! Destructor
      virtual ~CTensorS1();

      // Make new CTensorS1 (vs update)
      /** \param denL TensorL from which the new CTensorS1 should be made.
      \param denT TensorT from which the new CTensorS1 should be made.
      \param workmem Work memory */
      void makenew( CTensorL * denL, CTensorT * denTup, CTensorT * denTdown, dcomplex * workmem );

      private:
      // makenew when movingright
      void makenewRight( const int ikappa, CTensorL * denL, CTensorT * denTup, CTensorT * denTdown, dcomplex * workmem );

      // makenew when movingleft
      void makenewLeft( const int ikappa, CTensorL * denL, CTensorT * denTup, CTensorT * denTdown, dcomplex * workmem );
   };
} // namespace CheMPS2

#endif
