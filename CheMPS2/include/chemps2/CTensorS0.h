
#ifndef TENSORS0_CHEMPS2_H
#define TENSORS0_CHEMPS2_H

#include "CTensorL.h"
#include "CTensorOperator.h"
#include "CTensorT.h"
#include "SyBookkeeper.h"
#include "Tensor.h"

namespace CheMPS2 {
   /** CTensorS0 class.
    \author Lars-Hendrik Frahm
    \date September 14, 2017

    The CTensorS0 class is a storage and manipulation class for the spin-0
   component of two contracted creators or two contracted annihilators. */
   class CTensorS0 : public CTensorOperator {
      public:
      //! Constructor
      /** \param boundary_index The boundary index
      \param Idiff Direct product of irreps of the two 2nd quantized operators;
     both sandwiched & to sandwich
      \param moving_right If true: sweep from left to right. If false: sweep
     from right to left
      \param denBK The symmetry sector partitioning */
      CTensorS0( const int boundary_index, const int Idiff, const bool moving_right, const SyBookkeeper * book_up, const SyBookkeeper * book_down );

      //! Destructor
      virtual ~CTensorS0();

      // Make new CTensorS0 (vs update)
      /** \param denT CTensorT from which the new CTensorS0 should be made. */
      void makenew( CTensorT * denTup, CTensorT * denTdown, CTensorO * previous, dcomplex * workmem );

      // Make new CTensorS0 (vs update)
      /** \param denL CTensorL from which the new CTensorS0 should be made.
      \param denT CTensorT from which the new CTensorS0 should be made.
      \param workmem Work memory */
      void makenew( CTensorL * denL, CTensorT * denTup, CTensorT * denTdown, dcomplex * workmem );

      private:
      // makenew when movingright
      void makenewRight( const int ikappa, CTensorT * denTup, CTensorT * denTdown, CTensorO * previous, dcomplex * workmem );

      // makenew when movingleft
      void makenewLeft( const int ikappa, CTensorT * denTup, CTensorT * denTdown, CTensorO * previous, dcomplex * workmem );

      // makenew when movingright
      void makenewRight( const int ikappa, CTensorL * denL, CTensorT * denTup, CTensorT * denTdown, dcomplex * workmem );

      // makenew when movingleft
      void makenewLeft( const int ikappa, CTensorL * denL, CTensorT * denTup, CTensorT * denTdown, dcomplex * workmem );
   };
}

#endif
