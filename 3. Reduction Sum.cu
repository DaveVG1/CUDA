
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#define N 512 // Falla a partir de 2048 porque 2048/2 es el número máximo de hebras por bloque

// Realiza la suma secuencial de los valores del vector
float sumaSecuencial(float *vector)
{
	float fResultado = 0.0;

	for (int iPos = 0; iPos < N; iPos++)
		fResultado += vector[iPos];
	
	return fResultado;
}

// Kernell CUDA para la suma de los valores del vector
__global__ void sumaParalela(float *vector, int n)
{
	__shared__ float vectorComp[N];

	vectorComp[threadIdx.x] = vector[threadIdx.x];
	if (threadIdx.x + blockDim.x < n)
		vectorComp[threadIdx.x + blockDim.x] = vector[threadIdx.x + blockDim.x];

	__syncthreads();

	for (unsigned int iPos = n >> 1; iPos >= 1; iPos = iPos >> 1)
	{
		if (threadIdx.x < iPos)
			vectorComp[threadIdx.x] += vectorComp[threadIdx.x + iPos];

		__syncthreads();
	}

	if (threadIdx.x == 0){
		vector[0] = vectorComp[0];
	}
}



int main(void)
{
	float host_v[N];
	float fResultadoParalelo, fResultadoSecuencial;
	float *dev_v;

	// Se llena de forma aleatoria el vector sobre el que se realiza la suma
	srand((unsigned) time(NULL));
	for (int i = 0; i < N; i++)
		host_v[i] = floorf(100*(rand()/(float)RAND_MAX));

	// Pedir memoria en el Device para el vector a sumar (dev_v)
	/* COMPLETAR */

	cudaMalloc((void **)&dev_v, N*sizeof(float));

	// Transferir el vector del Host al Device
	/* COMPLETAR */

	cudaMemcpy(dev_v, host_v, N*sizeof(float), cudaMemcpyHostToDevice);

	int threads = (N / 2) + N % 2;
	
	// Llamar al kernell CUDA
	/* COMPLETAR */
	sumaParalela<<< 1, threads>>>(dev_v, N);
	// Copiar el resultado de la operación del Device al Host
	/* COMPLETAR */
	cudaMemcpy(&fResultadoParalelo, dev_v, sizeof(float), cudaMemcpyDeviceToHost);
	// Se comprueba que el resultado es correcto y se muestra un mensaje
	fResultadoSecuencial = sumaSecuencial(host_v);
	if (fResultadoParalelo == fResultadoSecuencial)
		printf("Operacion correcta\nDevice = %f\nHost   = %f\n", fResultadoParalelo, fResultadoSecuencial);
	else
		printf("Operacion INCORRECTA\nDevice = %f\nHost =   %f\n", fResultadoParalelo, fResultadoSecuencial);

	// Librerar la memoria solicitada en el Device
	/* COMPLETAR */
	cudaFree(dev_v);
	return 0;
}