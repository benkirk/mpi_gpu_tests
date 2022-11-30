all: hello_world_mpi_cpu hello_world_mpi_cuda

hello_world_mpi_cuda: hello_world_mpi_cuda.o fillprint.o
	mpicxx -g -o hello_world_mpi_cuda hello_world_mpi_cuda.o fillprint.o -lcuda -lcudart

hello_world_mpi_cuda.o: hello_world_mpi.C
	mpicxx -g -c -DHAVE_CUDA -o hello_world_mpi_cuda.o hello_world_mpi.C

# no-CUDA / CPU only variant
hello_world_mpi_cpu: hello_world_mpi.C
	mpicxx -g -o hello_world_mpi_cpu hello_world_mpi.C

fillprint.o: fillprint.cu
	nvcc -c fillprint.cu

run_CPU: hello_world_mpi_cpu
	@echo "------------------"
	@echo "CPU Memory:"
	@echo "------------------"
	mpiexec $(MPI_ARGS) ./hello_world_mpi_cpu || echo "Failed"
	@echo && echo && echo

run_GPUd: hello_world_mpi_cuda
	@echo "------------------"
	@echo "GPU Device Memory:"
	@echo "------------------"
	mpiexec $(MPI_ARGS) ./hello_world_mpi_cuda -d || echo "Failed"
	@echo && echo && echo

run_GPUm: hello_world_mpi_cuda
	@echo "------------------"
	@echo "GPU Managed Memory:"
	@echo "------------------"
	mpiexec $(MPI_ARGS) ./hello_world_mpi_cuda -m || echo "Failed"
	@echo && echo && echo

run_GPUdc: hello_world_mpi_cuda
	@echo "---------------------------------"
	@echo "GPU Device Memory + Copy-to-Host:"
	@echo "---------------------------------"
	mpiexec $(MPI_ARGS) ./hello_world_mpi_cuda -d -c || echo "Failed"
	@echo && echo && echo

run_GPUmc: hello_world_mpi_cuda
	@echo "----------------------------------"
	@echo "GPU Managed Memory + Copy-to-Host:"
	@echo "----------------------------------"
	mpiexec $(MPI_ARGS) ./hello_world_mpi_cuda -m -c || echo "Failed"
	@echo && echo && echo

run:
	$(MAKE) run_CPU
	$(MAKE) run_GPUd
	$(MAKE) run_GPUm
	$(MAKE) run_GPUdc
	$(MAKE) run_GPUmc

clean:
	rm -f *~ hello_world_mpi_cuda hello_world_mpi_cpu *.o core

clobber:
	$(MAKE) clean
	rm -f run*.sh.o*
