#ifndef INFOLOG_H
#define INFOLOG_H

#include <fstream>
#include <map>

namespace combblas{
namespace autotuning {

class InfoLog {
public:
    InfoLog(const std::string fileName, int rank) : rank(rank) {
        ofs.open(fileName, std::ofstream::out);
    }


    /* Timer utilities */
    void StartTimer(const std::string label) { 
        double stime = MPI_Wtime(); 
        infoMap.emplace(label, std::to_string(stime));
    }

    void EndTimer(const std::string label) {
        double etime = MPI_Wtime();
        double totalTime = etime - std::stod(infoMap[label]);
        infoMap.emplace(label, std::to_string(totalTime));
    }

    void StartTimerGlobal(const std::string label) { 
        double stime = MPI_Wtime(); 
        infoMapGlobal.emplace(label, std::to_string(stime));
    }

    void EndTimerGlobal(const std::string label) {
        double etime = MPI_Wtime();
        double totalTime = etime - std::stod(infoMapGlobal[label]);
        infoMapGlobal.emplace(label, std::to_string(totalTime));
    }
    

    /* Write all info into file */
    void WriteInfo() {
        std::for_each(infoMap.begin(), infoMap.end(),
            [this](auto const& elem) {
                this->ofs<<elem.first<<":"<<elem.second;
            }
        );
        ofs<<std::endl;
    }

    void WriteInfoGlobal() {
        std::for_each(infoMapGlobal.begin(), infoMapGlobal.end(),
            [this](auto const& elem) {
                this->ofs<<elem.first<<":"<<elem.second;
            }
        );
        ofs<<std::endl;
    }


    /* Print somerhing */
    void Print(const std::string label) {
        if (rank==0) {
            std::cout<<"["<<label<<"]"<<": "<<infoMap[label]<<std::endl;
        }
    }

    void PrintGlobal(const std::string label) {
        if (rank==0) {
            std::cout<<"["<<label<<"]"<<": "<<infoMapGlobal[label]<<std::endl;
        }
    }


    /* Put and get utilities */
    std::string Get(const std::string label) {
        return infoMap[label];
    }

    void Put(const std::string key, const std::string value) {
        infoMap.emplace(key, value);
    }

    std::string GetGlobal(const std::string label) {
        return infoMapGlobal[label];
    }

    void PutGlobal(const std::string key, const std::string value) {
        infoMapGlobal.emplace(key, value);
    }


    /* Clear maps */
    void Clear() {infoMap.clear();}
    void ClearGlobal() {infoMapGlobal.clear();}


    ~InfoLog() {ofs.close();}


private:
    std::ofstream ofs;
    std::map<std::string, std::string> infoMap;
    std::map<std::string, std::string> infoMapGlobal;
    int rank;

};

}
}




#endif




