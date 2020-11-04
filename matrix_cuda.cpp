//Matrix Frobenius Norm Kokkos - Andy Turgeson
#include <Kokkos_Core.hpp>
#include <cstdio>
#include <typeinfo>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <ctime>
#include <stdlib.h>
#include <stdio.h>

using view_type = Kokkos::View<double**>;
using view_vector = Kokkos::View<double*>;
using host_view_type = view_type::HostMirror;
using host_view_vector = view_vector::HostMirror;
		//using ExecSpace = Kokkos::Cuda;
		//using MemSpace = Kokkos::CudaUVMSpace;
		//using Layout = Kokkos::LayoutRight;
		//using range_policy = Kokkos::RangePolicy<ExecSpace>;
	

struct Norm {
	view_vector P;
	Norm(view_vector P_) : P(P_) {}
	using value_type = double;  // Specify type for reduction value, lsum

 	KOKKOS_INLINE_FUNCTION
  	void operator()(int i, double &lsum) const {
		lsum += P(i);
  	}
};

struct NormalizePtoB {
	view_vector P;
	view_vector B;
	view_vector norm;
	NormalizePtoB(view_vector P_, view_vector B_, view_vector norm_) : P(P_), B(B_), norm(norm_) {}
	using value_type = double;  // Specify type for reduction value, lsum

 	KOKKOS_INLINE_FUNCTION
  	void operator()(int i) const {
		B(i) = P(i)/norm(0);
  	}
};

struct NormalizeP {
	view_vector P;
	view_vector norm;
	NormalizeP(view_vector P_, view_vector norm_) : P(P_), norm(norm_) {}
	using value_type = double;  // Specify type for reduction value, lsum

 	KOKKOS_INLINE_FUNCTION
  	void operator()(int i) const {
		P(i) = P(i)/norm(0);
  	}
};

struct MatrixMult {
	view_type matrix;
	view_vector B;
	view_vector P;
	MatrixMult(view_type matrix_, view_vector B_, view_vector P_) : matrix(matrix_), B(B_), P(P_) {}
	using value_type = double;  // Specify type for reduction value, lsum

 	KOKKOS_INLINE_FUNCTION
  	void operator()(int i) const {
		double dotsum = 0.0;
		for (int j = 0; j < matrix.extent(1); j++) {	  
    		dotsum += matrix(i,j)*B(j);
		}
		P(i) = dotsum;
  	}
};


struct Matrix_init {
	view_vector B;
	Matrix_init(view_vector B_) : B(B_) {}
	using value_type = double;  // Specify type for reduction value, lsum
 	KOKKOS_INLINE_FUNCTION
  	void operator()(int i) const {	  
	 	B(i) = i%10;
  	}
};

int main(int argc, char* argv[]) {
	
	
	int count = 0;
	int iter = 10;
	std::string filename = "test.mtx";
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
	
	cudaGetDeviceCount(&count);	
	Kokkos::initialize(argc, argv);
	{
        
		{

			std::ifstream fin(filename);

	        	int M, N, L;

        		while (fin.peek() == '%') fin.ignore(2048, '\n');

      	 		fin >> M >> N >> L;
        		std::cout << M << " " << N << " " << L << std::endl;
      
			const int size = N;
			
			view_type matrix("matrix", M, N);
			view_type::HostMirror host_matrix = Kokkos::create_mirror_view( matrix );
			view_vector B("B", N);
			view_vector P("P", N);
			view_vector norm("norm", 1);
			view_vector::HostMirror host_norm = Kokkos::create_mirror_view( norm );
        		
			double sum = 0.0;

			for (int i = 0; i < L; i++) {
			int m, n;
			double data;
			fin >> m >> n >> data;
			host_matrix(m-1,n-1) = data;
			//sum += data*data;
			}
			
				Kokkos::deep_copy( matrix, host_matrix);			
		        std::cout << "size: " << size << std::endl;
				//std::cout << "nonparallel: " << sqrt(sum) << std::endl;
	
				Kokkos::parallel_for("createArrayB", size, Matrix_init(B));

			//try {
				for (int i = 0; i < iter; i++) {
					sum = 0.0;

					Kokkos::parallel_for("MatrixMultiplication", size, MatrixMult(matrix, B, P));

					Kokkos::parallel_reduce("GetNormSum", size, Norm(P), sum);
					//std::cout << sum << std::endl;
					host_norm(0)=std::sqrt(sum);
					Kokkos::deep_copy( norm, host_norm);

					Kokkos::parallel_for("NormalizeB", size, NormalizePtoB(P, B, norm));
				}

				Kokkos::parallel_for("NormalizeP", size, NormalizeP(P, norm));

		        std::cout << "norm of P: " << sqrt(sum) << std::endl;
				if (display) {
				view_vector::HostMirror host_P = Kokkos::create_mirror_view( P );
					Kokkos::deep_copy( host_P, P);
					for (int i = 0 ; i < N ; i++)  {
						std::cout << " P(" <<i <<") = " << host_P(i);
					}
					std::cout << std::endl;
				}
			//}

			Kokkos::fence();
			fin.close();
		}
	}

	Kokkos::finalize();
	return 0;
}


