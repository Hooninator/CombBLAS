



#include <fstream>
#include <iostream>



namespace combblas {
namespace autotuning {
class Debugger {
public:
    
    Debugger(){}
    

    Debugger(int rank): rank(rank) {
        ofs.open("logfile"+std::to_string(rank)+".out",std::ofstream::out);
    }
    
    Debugger& operator=(const Debugger& other) {
        rank = other.rank;
        ofs.open("logfile"+std::to_string(rank)+".out",std::ofstream::out);
        return *this;
    }
    
    void Print(std::string msg) {
        if (rank==0) std::cout<<msg<<std::endl;
    }
    
    void Log(std::string msg) {ofs<<msg<<std::endl;}
    
    ~Debugger(){ofs.close();}

private:
    int rank;
    std::ofstream ofs;

};

}//autotuning
}//combblas

