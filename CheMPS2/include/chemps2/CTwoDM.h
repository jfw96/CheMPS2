/*
   CheMPS2: a spin-adapted implementation of DMRG for ab initio quantum chemistry
   Copyright (C) 2013-2017 Sebastian Wouters

   This program is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation; either version 2 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License along
   with this program; if not, write to the Free Software Foundation, Inc.,
   51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
*/

#ifndef CTWODM_CHEMPS2_H
#define CTWODM_CHEMPS2_H

#include "CTensorF0.h"
#include "CTensorF1.h"
#include "CTensorL.h"
#include "CTensorS0.h"
#include "CTensorS1.h"
#include "CTensorT.h"
#include "SyBookkeeper.h"

namespace CheMPS2 {
   /** CTwoDM class.
    \author Sebastian Wouters <sebastianwouters@gmail.com>
    \date June 13, 2013
    
    The CTwoDM class stores the result of a converged DMRG calculation. With the 2DM \n
    \f$ \Gamma_{(\alpha \sigma) (\beta \tau) ; (\gamma \sigma) (\delta \tau)} = \braket{ a^{\dagger}_{\alpha \sigma} a^{\dagger}_{\beta \tau} a_{\delta \tau} a_{\gamma \sigma} } \f$\n
    we can define two spin-reduced versions of interest:\n
    \f$ \Gamma^{2A}_{\alpha \beta ; \gamma \delta} = \sum_{\sigma \tau} \Gamma_{(\alpha \sigma) (\beta \tau) ; (\gamma \sigma) (\delta \tau)} \f$ \n
    \f$ \Gamma^{2B}_{\alpha \beta ; \gamma \delta} = \sum_{\sigma} \left( \Gamma_{(\alpha \sigma) (\beta \sigma) ; (\gamma \sigma) (\delta \sigma)} - \Gamma_{(\alpha \sigma) (\beta -\sigma) ; (\gamma \sigma) (\delta -\sigma)} \right) \f$. \n
    Because the wave-function belongs to a certain Abelian irrep, \f$ I_{\alpha} \otimes I_{\beta} \otimes I_{\gamma} \otimes I_{\delta} = I_{trivial} \f$ must be valid before the corresponding element \f$ \Gamma^{A,B}_{\alpha \beta ; \gamma \delta} \f$ is non-zero. \n
\n
    We can also define spin-densities in the spin-ensemble as:\n
    \f{eqnarray*}{
       \Gamma^{spin}_{ij} & = & \frac{3(1 - \delta_{S,0})}{(S+1)(2S+1)} \sum_{S^z} S^z \braket{ S S^z \mid a^{\dagger}_{i \uparrow} a_{j \uparrow} - a^{\dagger}_{i \downarrow} a_{j \downarrow} \mid S S^z } \\
                          & = & \frac{3(1 - \delta_{S,0})}{2(S+1)} \left[ ( 2 - N_{elec} ) \Gamma^1_{ij} - \sum_r \Gamma^{2A}_{ir;rj} - \sum_r \Gamma^{2B}_{ir;rj} \right]
    \f} \n
    The normalization factor is chosen so that \f$ trace( \Gamma^{spin} ) = 2 S \f$.
*/
   class CTwoDM {

      public:
      //! Constructor
      /** \param denBKIn Symmetry sector bookkeeper
             \param ProbIn The problem to be solved */
      CTwoDM( const SyBookkeeper * denBKIn, const Problem * ProbIn );

      //! Destructor
      virtual ~CTwoDM();

      //! Get a 2DM_A term, using the DMRG indices
      /** \param cnt1 the first index
               \param cnt2 the second index
               \param cnt3 the third index
               \param cnt4 the fourth index
               \return the desired value */
      dcomplex getTwoDMA_DMRG( const int cnt1, const int cnt2, const int cnt3, const int cnt4 ) const;

      //! Get a 2DM_B term, using the DMRG indices
      /** \param cnt1 the first index
               \param cnt2 the second index
               \param cnt3 the third index
               \param cnt4 the fourth index
               \return the desired value */
      dcomplex getTwoDMB_DMRG( const int cnt1, const int cnt2, const int cnt3, const int cnt4 ) const;

      //! Get a 1-RDM term, using the DMRG indices
      /** \param cnt1 the first index
               \param cnt2 the second index
               \return the desired value */
      dcomplex get1RDM_DMRG( const int cnt1, const int cnt2 ) const;

      //! Get a spin-density term, using the DMRG indices
      /** \param cnt1 the first index
               \param cnt2 the second index
               \return the desired value  */
      dcomplex spin_density_dmrg( const int cnt1, const int cnt2 ) const;

      //! Get a 2DM_A term, using the HAM indices
      /** \param cnt1 the first index
               \param cnt2 the second index
               \param cnt3 the third index
               \param cnt4 the fourth index
               \return the desired value */
      dcomplex getTwoDMA_HAM( const int cnt1, const int cnt2, const int cnt3, const int cnt4 ) const;

      //! Get a 2DM_B term, using the HAM indices
      /** \param cnt1 the first index
               \param cnt2 the second index
               \param cnt3 the third index
               \param cnt4 the fourth index
               \return the desired value */
      dcomplex getTwoDMB_HAM( const int cnt1, const int cnt2, const int cnt3, const int cnt4 ) const;

      //! Get a 1-RDM term, using the HAM indices
      /** \param cnt1 the first index
               \param cnt2 the second index
               \return the desired value */
      dcomplex get1RDM_HAM( const int cnt1, const int cnt2 ) const;

      //! Get a spin-density term, using the HAM indices
      /** \param cnt1 the first index
               \param cnt2 the second index
               \return the desired value  */
      dcomplex spin_density_ham( const int cnt1, const int cnt2 ) const;

      //! Fill the 2DM terms with as second site index denT->gIndex()
      /** \param denT DMRG site-matrices
               \param Ltens Ltensors
               \param F0tens F0tensors
               \param F1tens F1tensors
               \param S0tens S0tensors
               \param S1tens S1tensors*/
      void FillSite( CTensorT * denT, CTensorL *** Ltens, CTensorF0 **** F0tens, CTensorF1 **** F1tens, CTensorS0 **** S0tens, CTensorS1 **** S1tens );

      //! After the whole 2-RDM is filled, a prefactor for higher multiplicities should be applied
      void correct_higher_multiplicities();

      //! Return the dcomplex trace of 2DM-A (should be N(N-1))
      /** \return dcomplex trace of 2DM-A */
      dcomplex trace() const;

      //! Calculate the energy based on the 2DM-A
      /** \return The energy calculated as 0.5*trace(2DM-A * Ham) */
      dcomplex energy() const;

      //   //! Print the natural orbital occupation numbers
      //   void print_noon() const;

      //   //! Save the CTwoDMs to disk
      //   void save() const;

      //   //! Load the CTwoDMs from disk
      //   void read();

      //   //! Save the 2-RDM-A to disk in Hamiltonian indices
      //   /** \param filename The filename to store the 2-RDM at */
      //   void save_HAM( const string filename ) const;

      //   //! Write the 2-RDM-A to a file
      //   /** param filename where to write the 2-RDM-A elements to */
      //   void write2DMAfile( const string filename ) const;

      //   //! Add the 2-RDM elements of all MPI processes
      //   void mpi_allreduce();

      private:
      //The BK containing all the irrep information
      const SyBookkeeper * denBK;

      //The problem containing orbital reshuffling and symmetry information
      const Problem * Prob;

      //The chain length
      int L;

      //Two 2DM^{A,B} objects
      dcomplex * two_rdm_A;
      dcomplex * two_rdm_B;

      // Set 2DM terms, using the DMRG indices
      void set_2rdm_A_DMRG( const int cnt1, const int cnt2, const int cnt3, const int cnt4, const dcomplex value );
      void set_2rdm_B_DMRG( const int cnt1, const int cnt2, const int cnt3, const int cnt4, const dcomplex value );

      //Helper functions
      dcomplex doD1( CTensorT * denT );
      dcomplex doD2( CTensorT * denT, CTensorL * Lright, dcomplex * workmem );
      dcomplex doD3( CTensorT * denT, CTensorS0 * S0right, dcomplex * workmem );
      dcomplex doD4( CTensorT * denT, CTensorF0 * F0right, dcomplex * workmem );
      dcomplex doD5( CTensorT * denT, CTensorF0 * F0right, dcomplex * workmem );
      dcomplex doD6( CTensorT * denT, CTensorF1 * F1right, dcomplex * workmem );
      dcomplex doD7( CTensorT * denT, CTensorL * Lleft, dcomplex * workmem );
      dcomplex doD8( CTensorT * denT, CTensorL * Lleft, CTensorL * Lright, dcomplex * workmem, dcomplex * workmem2, int Irrep_g );
      void doD9andD10andD11( CTensorT * denT, CTensorL * Lleft, CTensorL * Lright, dcomplex * workmem, dcomplex * workmem2, dcomplex * d9, dcomplex * d10, dcomplex * d11, int Irrep_g );
      dcomplex doD12( CTensorT * denT, CTensorL * Lleft, CTensorL * Lright, dcomplex * workmem, dcomplex * workmem2, int Irrep_g );
      dcomplex doD13( CTensorT * denT, CTensorL * Lleft, CTensorS0 * S0right, dcomplex * workmem, dcomplex * workmem2, int Irrep_g );
      dcomplex doD14( CTensorT * denT, CTensorL * Lleft, CTensorS0 * S0right, dcomplex * workmem, dcomplex * workmem2, int Irrep_g );
      dcomplex doD15( CTensorT * denT, CTensorL * Lleft, CTensorS1 * S1right, dcomplex * workmem, dcomplex * workmem2, int Irrep_g );
      dcomplex doD16( CTensorT * denT, CTensorL * Lleft, CTensorS1 * S1right, dcomplex * workmem, dcomplex * workmem2, int Irrep_g );
      dcomplex doD17orD21( CTensorT * denT, CTensorL * Lleft, CTensorF0 * F0right, dcomplex * workmem, dcomplex * workmem2, int Irrep_g, bool shouldIdoD17 );
      dcomplex doD18orD22( CTensorT * denT, CTensorL * Lleft, CTensorF0 * F0right, dcomplex * workmem, dcomplex * workmem2, int Irrep_g, bool shouldIdoD18 );
      dcomplex doD19orD23( CTensorT * denT, CTensorL * Lleft, CTensorF1 * F1right, dcomplex * workmem, dcomplex * workmem2, int Irrep_g, bool shouldIdoD19 );
      dcomplex doD20orD24( CTensorT * denT, CTensorL * Lleft, CTensorF1 * F1right, dcomplex * workmem, dcomplex * workmem2, int Irrep_g, bool shouldIdoD20 );
   };
} // namespace CheMPS2

#endif
