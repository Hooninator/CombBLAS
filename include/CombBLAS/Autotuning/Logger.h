
#ifndef LOGGER_H
#define LOGGER_H


#include <fstream>
#include <iostream>
#include <string>


namespace combblas {
namespace autotuning {
class Logger {
public:
    
    Logger(int rank, std::string name): rank(rank) {
        ofs.open(name, std::ofstream::out);
    }

    
    template <typename T>
    void Print(T msg, bool allRanks=false) {
        if (rank==0 || allRanks) std::cout<<msg<<std::endl;
    }
    
    
    template <typename T>
    void Log(T msg, bool allRanks=false) {
        if (rank==0 || allRanks) ofs<<msg<<std::endl;
    }

    ~Logger(){ofs.close();}

private:
    int rank;
    std::ofstream ofs;

};

}//autotuning
}//combblas

#endif

