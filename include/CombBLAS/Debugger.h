



#include <fstream>
#include <iostream>



namespace combblas {
namespace autotuning {
class Debugger {
public:
    
    Debugger(int rank): rank(rank) {
        ofs.open("logfile"+std::to_string(rank)+".out",std::ofstream::out);
    }

    
    void Print(std::string msg) {
        if (rank==0) std::cout<<msg<<std::endl;
    }
    

    void Log(std::string msg) {ofs<<msg<<std::endl;}


private:
    int rank;
    std::ofstream ofs;

};

}//autotuning
}//combblas

