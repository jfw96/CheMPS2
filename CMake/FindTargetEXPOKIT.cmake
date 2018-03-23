# FindTargetEXPOKIT.cmake
# ----------------------
#
# EXPOKIT cmake module to wrap FindEXPOKIT.cmake in a target.
#
# This module sets the following variables in your project: ::
#
#   TargetEXPOKIT_FOUND - true if BLAS/LAPACK found on the system
#   TargetEXPOKIT_MESSAGE - status message with BLAS/LAPACK library path list
#

# 1st precedence - libraries passed in through -DEXPOKIT_LIBRARY

if (EXPOKIT_LIBRARY)
    if (NOT ${PN}_FIND_QUIETLY)
        message (STATUS "LAPACK detection suppressed.")
    endif()

    add_library (tgt::expokit INTERFACE IMPORTED)
    set_property (TARGET tgt::expokit PROPERTY INTERFACE_LINK_LIBRARIES ${EXPOKIT_LIBRARY})
else()
    message( FATAL_ERROR "You need to specify -DEXPOKIT_LIBRARY to proceed.")
endif()  