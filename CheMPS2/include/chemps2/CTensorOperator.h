
#ifndef CTENSOROPERATOR_H
#define CTENSOROPERATOR_H

#include "CTensor.h"
#include "CTensorT.h"
#include "SyBookkeeper.h"

namespace CheMPS2 {
   /** CTensorOperator class.
    \author Lars-Hendrik Frahm
    \date September 12, 2017

    The CTensorOperator class is a storage and update class for tensor operators with a given:\n
    - spin (two_j)
    - particle number (n_elec)
    - point group irrep (n_irrep).

    It replaces the previous classes TensorDiag, TensorSwap, TensorS0Abase,
   TensorS1Bbase, TensorF0Cbase, TensorF1Dbase, TensorA, TensorB, TensorC, and
   TensorD. Their storage and update functions have a common origin. The boolean
   prime_last denotes whether in which convention the tensor operator is
   stored:\n
    - \f$ \braket{ j m J M | j' m' } \braket{ j ( N, I ) || J ( n\_elec,
   n\_irrep ) || j' ( N + n\_elec, I \times n\_irrep ) } \f$ (prime_last ==
   true)
    - \f$ \braket{ j' m' J M | j m } \braket{ j ( N, I ) || J ( n\_elec,
   n\_irrep ) || j' ( N + n\_elec, I \times n\_irrep ) } \f$ (prime_last ==
   false).

    This determines the specific reduced update formulae when contracting with
   the Clebsch-Gordan coefficients of the reduced MPS tensors. */
   class CTensorOperator : public CTensor {
      public:
      //! Constructor
      /** \param boundary_index The boundary index
      \param two_j Twice the spin of the tensor operator
      \param n_elec How many electrons there are more in the symmetry sector of
     the lower leg compared to the upper leg
      \param n_irrep The (real-valued abelian) point group irrep difference
     between the symmetry sectors of the lower and upper legs (see Irreps.h)
      \param moving_right If true: sweep from left to right. If false: sweep
     from right to left.
      \param prime_last Convention in which the tensor operator is stored (see
     class information)
      \param jw_phase Whether or not to include a Jordan-Wigner phase due to the
     fermion anti-commutation relations
      \param bk_up   Symmetry bookkeeper of the upper MPS
      \param bk_down Symmetry bookkeeper of the lower MPS */
      CTensorOperator( const int boundary_index, const int two_j, const int n_elec, const int n_irrep, const bool moving_right,
                       const bool prime_last, const bool jw_phase, const CheMPS2::SyBookkeeper * bk_up, const CheMPS2::SyBookkeeper * bk_down );

      //! Destructor
      virtual ~CTensorOperator();

      //! Get the number of symmetry blocks
      /** \return The number of symmetry blocks */
      int gNKappa() const;

      //! Get the pointer to the storage
      /** return pointer to the storage */
      dcomplex * gStorage();

      //! Get the index corresponding to a certain tensor block
      /** \param N1 The up particle number sector
      \param TwoS1 The up spin symmetry sector
      \param I1 The up irrep sector
      \param N2 The down particle number sector
      \param TwoS2 The down spin symmetry sector
      \param I2 The down irrep sector
      \return The kappa corresponding to the input parameters; -1 means no
      such block */
      int gKappa( const int N1, const int TwoS1, const int I1, const int N2, const int TwoS2, const int I2 ) const;

      //! Get the storage jump corresponding to a certain tensor block
      /** \param kappa The symmetry block
      \return kappa2index[kappa], the memory jumper to a certain block */
      int gKappa2index( const int kappa ) const;

      //! Get the pointer to the storage of a certain tensor block
      /** \param N1 The up particle number sector
      \param TwoS1 The up spin symmetry sector
      \param I1 The up irrep sector
      \param N2 The down particle number sector
      \param TwoS2 The down spin symmetry sector
      \param I2 The down irrep sector
      \return Pointer to the storage of the specified tensor block; NULL
      means no such block */
      dcomplex * gStorage( const int N1, const int TwoS1, const int I1, const int N2, const int TwoS2, const int I2 );

      //! Get the boundary index
      /** \return the index */
      int gIndex() const;

      //! Get twice the spin of the tensor operator
      /** \return Twice the spin of the tensor operatorg */
      int get_2j() const;

      //! Get how many electrons there are more in the symmetry sector of the lower
      //! leg compared to the upper leg
      /** \return How many electrons there are more in the symmetry sector of the
   * lower leg compared to the upper leg */
      int get_nelec() const;

      //! Get the (real-valued abelian) point group irrep difference between the
      //! symmetry sectors of the lower and upper legs (see Irreps.h)
      /** \return The (real-valued abelian) point group irrep difference between the
   * symmetry sectors of the lower and upper legs (see Irreps.h) */
      int get_irrep() const;

      //! Clear and update
      /** \param previous The previous CTensorOperator needed for the update
      \param den_up   The upper MPS tensor needed for the update
      \param den_down The lower MPS tensor needed for the update
      \param workmem Work memory */
      void update( CTensorOperator * previous, CTensorT * den_up, CTensorT * den_down, dcomplex * workmem );

      //! daxpy for CTensorOperator
      /** \param alpha The prefactor
      \param to_add The CTensorOperator x which should be added:
      this <-- this + alpha * to_add */
      void zaxpy( dcomplex alpha, CTensorOperator * to_add );

      void zaxpy_tensorCD( dcomplex alpha, CTensorOperator * to_add );

      void zaxpy_tensorCTDT( dcomplex alpha, CTensorOperator * to_add );

      //! daxpy_transpose for C- and D-tensors (with special spin-dependent
      //! factors)
      /** \param alpha The prefactor
        \param to_add The CTensorOperator x which should be added: this <-- this 
               + alpha * special_spin_dependent_factor * to_add^T */
      void zaxpy_transpose_tensorCD( dcomplex alpha, CTensorOperator * to_add );

      //! Set all storage variables to 0.0
      void clear();

      //! Make the in-product of two CTensorOperator
      /** \param buddy The second tensor
        \param trans If trans == 'N' a regular ddot is taken. If trans == 'T'
        and n_elec==0, the in-product with buddy's transpose is made.
        \return The in-product */
      dcomplex inproduct( CTensorOperator * buddy, const char trans ) const;

      friend std::ostream & operator<<( std::ostream & os, const CTensorOperator & tns );

      protected:
      //! The bookkeeper of the upper MPS
      const CheMPS2::SyBookkeeper * bk_up;

      //! The bookkeeper of the lower MPS
      const CheMPS2::SyBookkeeper * bk_down;

      //! Twice the spin of the tensor operator
      int two_j;

      //! How many electrons there are more in the symmetry sector of the lower
      //! leg
      //! compared to the upper leg
      int n_elec;

      //! The (real-valued abelian) point group irrep difference between the
      //! symmetry sectors of the lower and upper legs (see Irreps.h)
      int n_irrep;

      //! Whether or not moving right
      bool moving_right;

      //! The up particle number sector
      int * sector_nelec_up;

      //! The up spin symmetry sector
      int * sector_irrep_up;

      //! The up spin symmetry sector
      int * sector_spin_up;

      //! The down spin symmetry sector (pointer points to sectorTwoS1 if two_j ==
      //! 0)
      int * sector_spin_down;

      //! Update moving right
      /** \param ikappa The tensor block which should be updated
        \param previous The previous CTensorOperator needed for the update
        \param den_up   The upper MPS tensor needed for the update
        \param den_down The lower MPS tensor needed for the update
        \param workmem Work memory */
      void update_moving_right( const int ikappa, CTensorOperator * previous, CTensorT * den_up, CTensorT * den_down, dcomplex * workmem );

      //! Update moving left
      /** \param ikappa The tensor block which should be updated
        \param previous The previous CTensorOperator needed for the update
        \param den_up   The upper MPS tensor needed for the update
        \param den_down The lower MPS tensor needed for the update
        \param workmem Work memory */
      void update_moving_left( const int ikappa, CTensorOperator * previous, CTensorT * den_up, CTensorT * den_down, dcomplex * workmem );

      //! Convention in which the tensor operator is stored (see classinformation)
      bool prime_last;

      //! Whether or not to include a Jordan-Wigner phase due to the fermion
      //! anti-commutation relations
      bool jw_phase;

      private:
   };

   std::ostream & operator<<( std::ostream & os, const CTensorOperator & tns );
}

#endif
