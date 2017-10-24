
#ifndef CTENSORF0_HAMMPS_H
#define CTENSORF0_HAMMPS_H

#include "CTensor.h"
#include "CTensorL.h"
#include "CTensorOperator.h"
#include "CTensorT.h"
#include "SyBookkeeper.h"

namespace CheMPS2 {
   /** CTensorF0 class.
    \author Lars-Hendrik Frahm
    \date September 14, 2017

    The CTensorF0 class is a storage and manipulation class for the spin-0
   component of a contracted creator & annihilator. */
   class CTensorF0 : public CTensorOperator {
      public:
      //! Constructor
      /** \param boundary_index The boundary index
      \param Idiff Direct product of irreps of the two 2nd quantized operators;
     both sandwiched & to sandwich
      \param moving_right If true: sweep from left to right. If false: sweep
     from right to left
      \param denBK The symmetry sector bookkeeper */
      CTensorF0( const int boundary_index, const int Idiff, const bool moving_right, const CheMPS2::SyBookkeeper * book_up, const CheMPS2::SyBookkeeper * book_down );

      //! Destructor
      virtual ~CTensorF0();

      // Make new CTensorF0 (vs update)
      /** \param denT CTensorT from which the new CTensorF0 should be made. */
      void makenew( CTensorT * denTup, CTensorT * denTdown, CTensorO * previous, dcomplex * workmem );

      // Make new CTensorF0 (vs update)
      /** \param denL TensorL from which the new CTensorF0 should be made.
      \param denT CTensorT from which the new CTensorF0 should be made.
      \param workmem Work memory */
      void makenew( CTensorL * denL, CTensorT * denTup, CTensorT * denTdown, dcomplex * workmem );

      private:
      // makenew when movingright
      void makenewRight( const int ikappa, CTensorL * denL, CTensorT * denTup, CTensorT * denTdown, dcomplex * workmem );

      // makenew when movingleft
      void makenewLeft( const int ikappa, CTensorL * denL, CTensorT * denTup, CTensorT * denTdown, dcomplex * workmem );

      // makenew when movingright
      void makenewRight( const int ikappa, CTensorT * denTup, CTensorT * denTdown, CTensorO * previous, dcomplex * workmem );

      // makenew when movingleft
      void makenewLeft( const int ikappa, CTensorT * denTup, CTensorT * denTdown, CTensorO * previous, dcomplex * workmem );
   };
}

#endif
