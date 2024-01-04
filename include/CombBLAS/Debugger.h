



#include <fstream>
#include <iostream>



namespace combblas {
namespace autotuning {
class Debugger {
public:
    
    Debugger(int rank): rank(rank) {
        ofs.open("logfile"+std::to_string(rank)+".out",std::ofstream::out);
    }

    
    void Print(std::string msg, bool allRanks=false) {
        if (rank==0 || allRanks) std::cout<<msg<<std::endl;
    }
    
    
    template <typename T>
    void Log(T msg) {ofs<<msg<<std::endl;}

    ~Debugger(){ofs.close();}

private:
    int rank;
    std::ofstream ofs;

};

}//autotuning
}//combblas

