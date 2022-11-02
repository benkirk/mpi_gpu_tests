hello_world_mpi_cuda: hello_world_mpi_cuda.o fillprint.o
	mpicxx -o hello_world_mpi_cuda hello_world_mpi_cuda.o fillprint.o -lcuda -lcudart -fopenmp

hello_world_mpi_cuda.o: hello_world_mpi_cuda.C
	mpicxx -c hello_world_mpi_cuda.C

fillprint.o: fillprint.cu
	nvcc -c fillprint.cu

run_CPU: hello_world_mpi_cuda
	@echo "CPU Memory:"
	@mpiexec -n 2 ./hello_world_mpi_cuda && echo "Success" || echo "Failed"

run_GPUm: hello_world_mpi_cuda
	@echo "GPU Managed Memory:"
	@mpiexec -n 2 ./hello_world_mpi_cuda -m && echo "Success" || echo "Failed"

run_GPUd: hello_world_mpi_cuda
	@echo "GPU Device Memory:"
	@mpiexec -n 2 ./hello_world_mpi_cuda -d && echo "Success" || echo "Failed"

run:
	$(MAKE) run_CPU
	$(MAKE) run_GPUm
	$(MAKE) run_GPUd

clean:
	rm -f *~ hello_world_mpi_cuda *.o core
