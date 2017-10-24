
#ifndef TENSORS0T_CHEMPS2_H
#define TENSORS0T_CHEMPS2_H

#include "CTensor.h"
#include "CTensorLT.h"
#include "CTensorOperator.h"
#include "CTensorT.h"
#include "SyBookkeeper.h"

namespace CheMPS2 {
   /** CTensorS0T class.
    \author Lars-Hendrik Frahm
    \date September 15, 2017

    The CTensorS0T class is a storage and manipulation class for the spin-0
   component of two contracted creators or two contracted annihilators. */
   class CTensorS0T : public CTensorOperator {
      public:
      //! Constructor
      /** \param boundary_index The boundary index
      \param Idiff Direct product of irreps of the two 2nd quantized operators;
     both sandwiched & to sandwich
      \param moving_right If true: sweep from left to right. If false: sweep
     from right to left
      \param denBK The symmetry sector partitioning */
      CTensorS0T( const int boundary_index, const int Idiff, const bool moving_right, const SyBookkeeper * book_up, const SyBookkeeper * book_down );

      //! Destructor
      virtual ~CTensorS0T();

      // Make new CTensorS0T (vs update)
      /** \param denT TensorT from which the new CTensorS0T should be made. */
      void makenew( CTensorT * denTup, CTensorT * denTdown, CTensorO * previous, dcomplex * workmem );

      // Make new CTensorS0T (vs update)
      /** \param denL TensorL from which the new CTensorS0T should be made.
      \param denT TensorT from which the new CTensorS0T should be made.
      \param workmem Work memory */
      void makenew( CTensorLT * denL, CTensorT * denTup, CTensorT * denTdown, dcomplex * workmem );

      private:
      // makenew when movingright
      void makenewRight( const int ikappa, CTensorT * denTup, CTensorT * denTdown, CTensorO * previous, dcomplex * workmem );

      // makenew when movingleft
      void makenewLeft( const int ikappa, CTensorT * denTup, CTensorT * denTdown, CTensorO * previous, dcomplex * workmem );

      // makenew when movingright
      void makenewRight( const int ikappa, CTensorLT * denL, CTensorT * denTup, CTensorT * denTdown, dcomplex * workmem );

      // makenew when movingleft
      void makenewLeft( const int ikappa, CTensorLT * denL, CTensorT * denTup, CTensorT * denTdown, dcomplex * workmem );
   };
}

#endif
