#ifndef LOGGER_CHEMPS2_H
#define LOGGER_CHEMPS2_H

#include <iostream>
#include <ctime>
#include <string>

namespace CheMPS2{

class Logger {
public:
  static const int TOFILE = 0;
  static const int TOCONSOLE = 1;
 private:
  const int storeType;
  static const std::string hasline;
  std::ostream *os;
  std::time_t start;

  void PrintWelcome();
  void PrintGoodbye();
  void TextWithDate(const std::string& input, const std::time_t& time);
  
public:
  Logger(const int storeTypeIn = TOCONSOLE);
  ~Logger();
  
  template <typename T>
  Logger& operator<<(const T& value);
  
};

template <typename T>
Logger& Logger::operator<<(const T& value){
  (*os) << value;
  return (*this);
}
}

#endif
