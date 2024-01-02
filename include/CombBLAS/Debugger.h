



#include <fstream>
#include <iostream>



namespace combblas {
namespace autotuning {
class Debugger {
public:
    
    static void Init(int rank) {
        rank = rank;
        ofs.open("logfile"+std::to_string(rank)+".out",std::ofstream::out);
    }
    
    static void Print(std::string msg) {
        if (rank==0) std::cout<<msg<<std::endl;
    }
    
    static void Log(std::string msg) {ofs<<msg<<std::endl;}

private:
    static int rank;
    static std::ofstream ofs;

};

}//autotuning
}//combblas

