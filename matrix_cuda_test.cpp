//Matrix Multiplication Power iteration Eiganvalues - Andy Turgeson
#include <Kokkos_Core.hpp>
#include <cstdio>
#include <typeinfo>
#include <iostream>
#include <cstdio>
#include <fstream>
#include <algorithm>
#include <cmath>

struct squaresum {
    using value_type = int;

    KOKKOS_INLINE_FUNCTION
    void operator()(const int i, const int j, int& lsum) const {
        for (int k = 0; k < j; k++) {
            lsum += matrix(i,k);  // compute the sum of squares
        }
    }
};



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
            if (argv[1] == NULL) { char file = "matrix.mtx"} 
			std::ifstream fin(file);

	        	int M, N, L;

        		while (fin.peek() == '%') fin.ignore(2048, '\n');

      	 		fin >> M >> N >> L;
        		std::cout << M << " " << N << " " << L << std::endl;
      
			const int size = M*N;
			//ViewVectorType matrix("matrix", size); 
			ViewMatrixType matrix("matrix", N, M);
			ViewVectorType::HostMirror host_matrix = Kokkos::create_mirror_view( matrix );
			
        		

			for (int i = 0; i < L; i++) {
				int m, n;
				double data;
				fin >> m >> n >> data;
				host_matrix(m, n) = data;
			}
			
			Kokkos::deep_copy( matrix, host_matrix);			
	        	std::cout << "size: " << size << std::endl;

			double sum = 0.0;
		
			Kokkos::parallel_reduce("reduction", M, N, (const int i, const int j, double& add) { 

				add += matrix(i)*matrix(j);
			 }, sum);

	        	std::cout << "parallel frobenius norm from array is: " << sqrt(sum) << std::endl;
			Kokkos::fence();
			fin.close();
		}
	}

	Kokkos::finalize();
	return 0;
}