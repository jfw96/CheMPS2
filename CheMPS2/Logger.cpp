
#include "Logger.h"

#include <fstream>
#include <sstream>
#include <iomanip>
#include <limits.h>
#include <unistd.h>

const std::string hashline = "#################################################################\n";

CheMPS2::Logger::Logger(const int storeTypeIn): storeType(storeTypeIn) {
  if(storeType == TOCONSOLE){
    os = &std::cout;
 } else if (storeType == TOFILE){
    os = new std::ofstream("test.txt", std::ofstream::out);
  }
  (*os).precision(15);

  start = time(NULL);
  PrintWelcome();
  
}

void CheMPS2::Logger::PrintWelcome(){
  TextWithDate("Starting to run CheMPS2 code", start);
  
  char hostname[HOST_NAME_MAX];
  gethostname(hostname, HOST_NAME_MAX);
  (*this) << "### Running on " << hostname << "\n" << hashline;
  
  
}

void CheMPS2::Logger::TextWithDate(const std::string& input, const std::time_t& time){
  tm* tmstart = localtime(&time);
  std::ostringstream text;
  
  text << hashline;
  text << "### " << input << " on " << tmstart->tm_year + 1900 << "-";
  text << std::setfill('0') << std::setw(2) << tmstart->tm_mon + 1 << "-";
  text << std::setfill('0') << std::setw(2) << tmstart->tm_mday << " ";
  text << std::setfill('0') << std::setw(2) << tmstart->tm_hour << ":";
  text << std::setfill('0') << std::setw(2) << tmstart->tm_min  << ":";
  text << std::setfill('0') << std::setw(2) << tmstart->tm_sec << "\n";
  
  (*this) << text.str();
}

void CheMPS2::Logger::PrintGoodbye(){
  TextWithDate("Finished to run CheMPS2 code", time(NULL));
}

CheMPS2::Logger::~Logger(){
  PrintGoodbye();
  (*this) << "### Calculation took " << (time(NULL) - start) / 60.0 << " minutes\n" << hashline;
  
  if(storeType == TOFILE){
    (*static_cast<std::ofstream*>(os)).close();
    delete os;
  }
}
