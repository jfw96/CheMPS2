
#ifndef CSOBJECT_CHEMPS2_H
#define CSOBJECT_CHEMPS2_H

#include "CTensorO.h"
#include "CTensorT.h"
#include "SyBookkeeper.h"

namespace CheMPS2 {
   /** CSobject class.
    \author Lars-Hendrik Frahm
    \date September 19, 2017

    Handles the storage of a spin-coupled two-site object, as is needed for the
   effective Hamiltonian DMRG equations. Extra functionalities are:
     - the formation of an S-object out of two neighbouring TensorT's
     - the decomposition into two TensorT's */
   class CSobject {
      public:
      //! Constructor
      /** \param index The first index ( S spans index & index + 1 ).
      \param denBK Contains the virtual dimensions. Not constant as CSobject is
     allowed to set the virtual dimensions of symmetry sectors based on the
     Schmidt spectrum. */
      CSobject( const int index, SyBookkeeper * denBK );

      //! Destructor
      virtual ~CSobject();

      //! Get the number of symmetry blocks
      /** \return The number of symmetry blocks */
      int gNKappa() const;

      //! Get the pointer to the storage
      /** return pointer to the storage */
      std::complex< double > * gStorage();

      //! Get the index corresponding to a certain symmetry block
      /** \param NL The left particle number sector
      \param TwoSL The left spin symmetry sector
      \param IL The left irrep sector
      \param N1 The first site particle number sector
      \param N2 The second site particle number sector
      \param TwoJ The recoupled two-site spin symmetry sector
      \param NR The right particle number sector
      \param TwoSR The right spin symmetry sector
      \param IR The right irrep sector
      \return The kappa corresponding to the input parameters; -1 means no such
     block */
      int gKappa( const int NL, const int TwoSL, const int IL, const int N1,
                  const int N2, const int TwoJ, const int NR, const int TwoSR,
                  const int IR ) const;

      //! Get the storage jump corresponding to a certain symmetry block
      /** \param kappa The symmetry block
      \return kappa2index[kappa], the memory jumper to a certain block */
      int gKappa2index( const int kappa ) const;

      //! Get the pointer to the storage of a certain symmetry block
      /** \param NL The left particle number sector
      \param TwoSL The left spin symmetry sector
      \param IL The left irrep sector
      \param N1 The first site particle number sector
      \param N2 The second site particle number sector
      \param TwoJ The recoupled two-site spin symmetry sector
      \param NR The right particle number sector
      \param TwoSR The right spin symmetry sector
      \param IR The right irrep sector
      \return Pointer to the requested block; NULL means no such block ( kappa
     == -1 ) */
      std::complex< double > * gStorage( const int NL, const int TwoSL, const int IL,
                                         const int N1, const int N2, const int TwoJ,
                                         const int NR, const int TwoSR, const int IR );

      //! Get the location index
      /** \return the index */
      int gIndex() const;

      //! Join two sites to form a composite S-object.
      /** \param Tleft Left TensorT to form the composite S-object
      \param Tright Right TensorT to form the composite S-object */
      void Join( CTensorT * Tleft, CTensorT * Tright );

      void Join( CTensorO * Oleft, CSobject * innerS, CTensorO * Oright );

      void Add( std::complex< double > alpha, CSobject * to_add );
      //! SVD an S-object into 2 TensorT's.
      /** \param Tleft Left TensorT storage space. At output left normalized.
      \param Tright Right TensorT storage space. At output right normalized.
      \param virtualdimensionD The virtual dimension which is partitioned over
     the different symmetry blocks based on the Schmidt spectrum
      \param movingright When true, the singular values are multiplied into V^T,
     when false, into U.
      \param change Whether or not the symmetry virtual dimensions are allowed
     to change (when false: D doesn't matter)
      \return the discarded weight if change==true ; else 0.0 */
      double Split( CTensorT * Tleft, CTensorT * Tright, const int virtualdimensionD, const double cut_off, const bool movingright, const bool change );

      //! Add noise to the current S-object
      /** \param NoiseLevel The noise added to the S-object is of size (-0.5 <
   * random number < 0.5) * NoiseLevel / infinity-norm(gStorage()) */
      void addNoise( const double NoiseLevel );

      //! Convert the storage from diagram convention to symmetric Hamiltonian
      //! convention
      void prog2symm();

      //! Convert the storage from symmetric Hamiltonian convention to diagram
      //! convention
      void symm2prog();

      //! Get the left particle number symmetry of block ikappa
      /** \param ikappa The tensor block number
      \return the left particle number symmetry of block ikappa */
      int gNL( const int ikappa ) const;

      //! Get the left spin symmetry of block ikappa
      /** \param ikappa The tensor block number
      \return the left spin symmetry of block ikappa */
      int gTwoSL( const int ikappa ) const;

      //! Get the left irrep symmetry of block ikappa
      /** \param ikappa The tensor block number
      \return the left irrep symmetry of block ikappa */
      int gIL( const int ikappa ) const;

      //! Get the left local particle number symmetry of block ikappa
      /** \param ikappa The tensor block number
      \return the left local particle number symmetry of block ikappa */
      int gN1( const int ikappa ) const;

      //! Get the right local particle number symmetry of block ikappa
      /** \param ikappa The tensor block number
      \return the right local particle number symmetry of block ikappa */
      int gN2( const int ikappa ) const;

      //! Get the central spin symmetry of block ikappa
      /** \param ikappa The tensor block number
      \return the central spin symmetry of block ikappa */
      int gTwoJ( const int ikappa ) const;

      //! Get the right particle number symmetry of block ikappa
      /** \param ikappa The tensor block number
      \return the right particle number symmetry of block ikappa */
      int gNR( const int ikappa ) const;

      //! Get the right spin symmetry of block ikappa
      /** \param ikappa The tensor block number
      \return the right spin symmetry of block ikappa */
      int gTwoSR( const int ikappa ) const;

      //! Get the right irrep symmetry of block ikappa
      /** \param ikappa The tensor block number
      \return the right irrep symmetry of block ikappa */
      int gIR( const int ikappa ) const;

      //! Get the blocks from large to small:
      //! blocksize(reorder[i])>=blocksize(reorder[i+1])
      /** \param ikappa The index which counts from large to small
      \return The original block index */
      int gReorder( const int ikappa ) const;

      //! Get the pointer to the symmetry bookkeeper
      /** \return the pointer to the symmetry bookkeeper */
      const CheMPS2::SyBookkeeper * gBK() const { return denBK; };

      // void print() const;

      private:
      //! First site index
      const int index;

      //! The local irrep of site index
      const int Ilocal1;

      //! The local irrep of site index+1
      const int Ilocal2;

      //! Pointer to the symmetry BK
      SyBookkeeper * denBK;

      //! The number of symmetry blocks
      int nKappa;

      //! Symmetry sector arrays; length nKappa
      int * sectorNL;
      int * sectorTwoSL;
      int * sectorIL;
      int * sectorN1;
      int * sectorN2;
      int * sectorTwoJ;
      int * sectorNR;
      int * sectorTwoSR;
      int * sectorIR;

      //! kappa2index[ kappa ] indicates the start of tensor block kappa in storage.
      //! kappa2index[ nKappa ] gives the size of storage.
      int * kappa2index;

      //! The actual variables. Symmetry block kappa begins at storage +
      //! kappa2index[ kappa ] and ends at storage + kappa2index[ kappa + 1 ].
      std::complex< double > * storage;

      //! The array reorder: blocksize( reorder[ i ] ) >= blocksize( reorder[ i + 1
      //! ] ), with blocksize( k ) = kappa2index[ k + 1 ] - kappa2index[ k ]
      int * reorder;
   };
}

#endif
