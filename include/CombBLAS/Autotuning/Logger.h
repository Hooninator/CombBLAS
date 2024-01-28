
#ifndef LOGGER_H
#define LOGGER_H


#include <fstream>
#include <iostream>
#include <string>


namespace combblas {
namespace autotuning {
class Logger {
public:
    
    Logger(int rank, std::string name, bool allRanks): rank(rank), _allRanks(allRanks) {
        ofs.open(name, std::ofstream::out);
    }

    
    template <typename T>
    void Print(T msg) {
        if (rank==0 ||_allRanks ) std::cout<<msg<<std::endl;
    }
    
    
    template <typename T>
    void Log(T msg) {
        if (rank==0 || _allRanks) ofs<<msg<<std::endl;
    }

    ~Logger(){ofs.close();}

private:
    int rank;
    bool _allRanks;
    std::ofstream ofs;

};

}//autotuning
}//combblas

#endif

