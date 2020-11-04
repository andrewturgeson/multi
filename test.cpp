#include <cstdio>
#include <typeinfo>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <ctime>
#include <stdlib.h>
#include <stdio.h>
#include <cstring>

int main(int argc, char* argv[]) {
	
	
	int count = 0;
	int iter = 100;
	std::string filename = "test3.mtx";
	
    int display = 0;
	int argNum = 1;
	while (argv[argNum] != NULL) 
    {
        if (argv[argNum][0] == '-') 
        {   
			if (strcmp(argv[argNum],"-i")==0) {iter = std::stoi(argv[argNum+1]); argNum++;}
			if (strcmp(argv[argNum],"-d")==0) {display = 1;}
		} else {
            filename = std::string(argv[argNum]);
        } 
        argNum++;
    }
    
    std::ifstream fin(filename);

	int M, N, L;

    while (fin.peek() == '%') fin.ignore(2048, '\n');

    fin >> M >> N >> L;
    std::cout << M << " " << N << " " << L << std::endl;
      
	double matrix[M][N] = {};
	double B[M];
	double P[M];
	double sum = 0.0;

	for (int i = 0; i < L; i++) {
	    int m, n;
	    double data;
	    fin >> m >> n >> data;
	    matrix[m-1][n-1] = data;
    }
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            std::cout << matrix[i][j] << " "; 
        }
       std::cout << std::endl;
    }

    for (int i = 0; i < M; i++) {
        B[i] = 20*( ( ( (double) rand() ) / RAND_MAX ) -0.5);
        //std::cout << "B(" << i << ") = " << B[i] << std::endl;
    }

    for (int c = 0; c < iter; c++) {
        for (int i = 0; i < M; i++) {
            sum = 0;
            for (int j = 0; j < N; j++) {
                sum += matrix[i][j] * B[j];
            }
            P[i] = sum;
        }
        sum = 0;
        for (int i = 0; i < M; i++) {
            sum += P[i]*P[i];
        }
        for (int i = 0; i < M; i++) {
            B[i] = (P[i]/sqrt(sum));
        }
    }

    for (int i = 0; i < M; i++) {
        //B[i] = 20*( ( ( (double) rand() ) / RAND_MAX ) -0.5);
        std::cout << "B(" << i << ") = " << B[i] << std::endl;
    }


}

