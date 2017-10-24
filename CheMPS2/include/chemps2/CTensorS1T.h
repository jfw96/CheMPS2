
#ifndef CTENSORS1T_HAMMPS_H
#define CTENSORS1T_HAMMPS_H

#include "CTensor.h"
#include "CTensorLT.h"
#include "CTensorOperator.h"
#include "CTensorT.h"
#include "SyBookkeeper.h"

namespace CheMPS2 {
   /** CTensorS1T class.
    \author Lars-Hendrik Frahm
    \date September 15, 2017

    The CTensorS1T class is a storage and manipulation class for the spin-1
   component of two contracted creators or two contracted annihilators. */
   class CTensorS1T : public CTensorOperator {
      public:
      //! Constructor
      /** \param boundary_index The boundary index
      \param Idiff Direct product of irreps of the two 2nd quantized operators;
     both sandwiched & to sandwich
      \param moving_right If true: sweep from left to right. If false: sweep
     from right to left
      \param denBK The symmetry sector partitioning */
      CTensorS1T( const int boundary_index, const int Idiff, const bool moving_right, const SyBookkeeper * book_up, const SyBookkeeper * book_down );

      //! Destructor
      virtual ~CTensorS1T();

      // Make new CTensorS1T (vs update)
      /** \param denL TensorL from which the new CTensorS1T should be made.
      \param denT TensorT from which the new CTensorS1T should be made.
      \param workmem Work memory */
      void makenew( CTensorLT * denL, CTensorT * denTup, CTensorT * denTdown, dcomplex * workmem );

      private:
      // makenew when movingright
      void makenewRight( const int ikappa, CTensorLT * denL, CTensorT * denTup, CTensorT * denTdown, dcomplex * workmem );

      // makenew when movingleft
      void makenewLeft( const int ikappa, CTensorLT * denL, CTensorT * denTup, CTensorT * denTdown, dcomplex * workmem );
   };
}

#endif
