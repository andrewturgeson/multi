//Matrix Frobenius Norm Kokkos - Andy Turgeson
#include <Kokkos_Core.hpp>
#include <cstdio>
#include <typeinfo>
#include <iostream>
#include <cstdio>
#include <fstream>
#include <algorithm>
#include <cmath>


int main(int argc, char* argv[]) {
	int count = 0;
	cudaGetDeviceCount(&count);	
	Kokkos::initialize(argc, argv);
	{
        	
		using ExecSpace = Kokkos::Cuda;
		using MemSpace = Kokkos::CudaUVMSpace;
		using Layout = Kokkos::LayoutRight;
		using range_policy = Kokkos::RangePolicy<ExecSpace>;
		using ViewVectorType = Kokkos::View<double*>;
		//using ViewMatrixType = Kokkos::View<double**>;
		{

			std::ifstream fin("matrix.mtx");

	        	int M, N, L;

        		while (fin.peek() == '%') fin.ignore(2048, '\n');

      	 		fin >> M >> N >> L;
        		std::cout << M << " " << N << " " << L << std::endl;
      
			const int size = M*N;
			ViewVectorType matrix("matrix", size); 
			//maybe make a matrix...i
			//ViewMatrixType matrix("matrix", N, M);
			ViewVectorType::HostMirror host_matrix = Kokkos::create_mirror_view( matrix );
			
        		double sum = 0.0;

			for (int i = 0; i < L; i++) {
				int m, n;
				double data;
				fin >> m >> n >> data;
				host_matrix((m-1) + (n-1)*M) = data;
				sum += data*data;
			}
			
			Kokkos::deep_copy( matrix, host_matrix);			
	        	std::cout << "size: " << size << std::endl;
			std::cout << "nonparallel: " << sqrt(sum) << std::endl;

			sum = 0.0;
		
			Kokkos::parallel_reduce("reduction", size, KOKKOS_LAMBDA (const int i, double& add) { 
				add += matrix(i)*matrix(i);
			 }, sum);

	        	std::cout << "parallel frobenius norm from array is: " << sqrt(sum) << std::endl;
			Kokkos::fence();
			fin.close();
		}
	}

	Kokkos::finalize();
	return 0;
}


