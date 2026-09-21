// others.h -- file I/O helpers shared by the simulation modules.
#pragma once
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

// Read a directed edge list: one "i j" pair per line (whitespace separated), meaning
// user i follows user j. Pairs are appended flat to `links` as i, j, i, j, ...
void read_lins_from_txt(std::string fname, std::vector<int> &links)
{
    std::ifstream ifs(fname);
    std::string line;
    while (std::getline(ifs, line))
    {
        std::stringstream ss(line);
        int i, j;
        ss >> i >> j;
        links.push_back(i);
        links.push_back(j);
    }
}

// Write a vector as one comma-separated line; append = true adds a line to an existing file.
template <typename T>
void output_vector(std::vector<T> &v, std::string path, bool append = false)
{
    std::ofstream ofs;
    if (append) ofs.open(path, std::ios::app);
    else        ofs.open(path);

    bool first = true;
    for (auto &x : v) {
        if (!first) ofs << ",";
        ofs << x;
        first = false;
    }
    ofs << '\n';
    ofs.close();
}

void print_partition(int length = 30)
{
    for (int i = 0; i < length; i++) std::cout << '=';
    std::cout << std::endl;
}
