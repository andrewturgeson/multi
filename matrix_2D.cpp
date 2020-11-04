//Matrix Frobenius Norm Kokkos - Andy Turgeson
#include <Kokkos_Core.hpp>
#include <cstdio>
#include <typeinfo>
#include <iostream>
#include <cstdio>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <string.h>
using view_type = Kokkos::View<double**>;



struct ReduceFunctor {
	view_type matrix;
	ReduceFunctor(view_type matrix_) : matrix(matrix_) {}
	using value_type = double;  // Specify type for reduction value, lsum

 	KOKKOS_INLINE_FUNCTION
  	void operator()(int i, double &lsum) const {
		for (int j = 0; j < matrix.extent(1); j++) {	  
    		lsum += matrix(i,j)*matrix(i,j);
		}
  	}
};

int main(int argc, char* argv[]) {
	int count = 0;
	cudaGetDeviceCount(&count);	
	Kokkos::initialize(argc, argv);
	{
        using host_view_type = view_type::HostMirror;
		using ExecSpace = Kokkos::Cuda;
		using MemSpace = Kokkos::CudaUVMSpace;
		using Layout = Kokkos::LayoutRight;
		using range_policy = Kokkos::RangePolicy<ExecSpace>;
		using ViewMatrixType = Kokkos::View<double**>;
		{
             	std::ifstream fin(argv[1]);
                int M, N, L;

        		while (fin.peek() == '%') fin.ignore(2048, '\n');

      	 		fin >> M >> N >> L;
        		std::cout << M << " " << N << " " << L << std::endl;
      
			const int size = N;
			
			ViewMatrixType matrix("matrix", M, N);
			ViewMatrixType::HostMirror host_matrix = Kokkos::create_mirror_view( matrix );
			
        		double sum = 0.0;

			for (int i = 0; i < L; i++) {
				int m, n;
				double data;
				fin >> m >> n >> data;
				host_matrix(m,n) = data;
				sum += data*data;
			}
			
			Kokkos::deep_copy( matrix, host_matrix);			
	        	std::cout << "size: " << size << std::endl;
			std::cout << "nonparallel: " << sqrt(sum) << std::endl;

			sum = 0.0;
		
			Kokkos::parallel_reduce("reduction", size, ReduceFunctor(matrix), sum);

	        	std::cout << "parallel frobenius norm from array is: " << sqrt(sum) << std::endl;
			Kokkos::fence();
			fin.close();
		}
	}

	Kokkos::finalize();
	return 0;
}


