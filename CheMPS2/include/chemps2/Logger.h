#ifndef LOGGER_CHEMPS2_H
#define LOGGER_CHEMPS2_H

#include <ctime>
#include <fstream>
#include <iostream>
#include <string>

namespace CheMPS2 {

   const std::string hashline = "#################################################################\n";

   class Logger {
      public:
      static const int TOFILE    = 0;
      static const int TOCONSOLE = 1;

      private:
      const int storeType;
      std::ostream * os;
      std::time_t start;
      std::string fileName;

      void
      PrintWelcome();

      void PrintGoodbye();

      public:
      Logger( const int storeTypeIn = TOCONSOLE );

      ~Logger();

      void TextWithDate( const std::string & input, const std::time_t & time );

      const std::string & gHashLine() const { return hashline; };

      template < typename T >
      Logger & operator<<( const T & value );
   };

   template < typename T >
   Logger & Logger::operator<<( const T & value ) {
      if ( storeType == TOFILE ) {
         os = new std::ofstream( fileName.c_str(), std::ofstream::out | std::ofstream::app );
         os->precision( 15 );
      }

      ( *os ) << value;

      if ( storeType == TOFILE ) {
         static_cast< std::ofstream * >( os )->close();
      }

      return ( *this );
   }
}

#endif
