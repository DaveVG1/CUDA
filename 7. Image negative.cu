
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <assert.h>

#define ERROR_CHECK { cudaError_t err; \
  if ((err = cudaGetLastError()) != cudaSuccess) { \
    printf("CUDA error: %s, line %d\n", cudaGetErrorString(err), __LINE__);}}

// Hebras por dimensión de cada bloque
#define THREADS 32


// Cabecera de fichero BMP sin la signature
struct BMPHEADER
{
	unsigned int iFileSize;
	unsigned int iReserved;
	unsigned int iFileOffsetToPixelArray;

	unsigned int iDIBHeaderSize;
	unsigned int iWidth;
	unsigned int iHeigth;
	unsigned short iColorPlanes;
	unsigned short iBitsPerPixel;
	unsigned int iCompression;
	unsigned int iSizeOfDataWithPadding;
	unsigned int iPrintResolutionHorizontal;
	unsigned int iPrintResolutionVertical;
	unsigned int iColorsInPalete;
	unsigned int iImportantColors;
};

// Estructura para un pixel (B G R) (1 byte x 1 byte x 1 byte)
struct PIXEL
{
	unsigned char B;
	unsigned char G;
	unsigned char R;
};



// Negativo paralelo
__global__ void negativoParalelo(PIXEL *dev_input, PIXEL *dev_output, unsigned iWidth)
{
	// COMPLETAR
	unsigned int iCol = threadIdx.x + blockDim.x*blockIdx.x;
	unsigned int iFila = threadIdx.y + blockDim.y*blockIdx.y;

	unsigned pos = iFila*iWidth + iCol;

	dev_output[pos].R = 255 - dev_input[pos].R;
	dev_output[pos].G = 255 - dev_input[pos].G ;
	dev_output[pos].B = 255 - dev_input[pos].B ;
}


// Negativo secuencial
void negativoSecuencial(PIXEL *host_input, PIXEL *sec_output, unsigned iWidth, unsigned int iHeigth)
{
	// COMPLETAR
	for (unsigned int i = 0; i < iWidth*iHeigth; i++)
	{
		sec_output[i].B = 255- host_input[i].B;
		sec_output[i].G = 255-host_input[i].G;
		sec_output[i].R = 255-host_input[i].R ;
	}
}


int main(int argc, char **argv)
{
	struct BMPHEADER BMPheader;
	char signature[2];
	size_t dataToRead;
	struct PIXEL pixel;

	// Variables para las imagenes en el HOST
	struct PIXEL *host_input, *host_output, *sec_output;

	// Variables para las imagenes en el DEVICE
	struct PIXEL *dev_input, *dev_output;

	if (argc != 4)
	{
		printf("USO:\nFiltro2D <fichero imagen> <fichero salida paralelo> <fichero salida secuencial\n");
		return -1;
	}

	// Se abre el fichero de la imagen
	FILE *fData = fopen(argv[1], "rb");
	if (fData == NULL)
	{
		printf("Error abriendo el fichero de datos\n");
		return -1;
	}

	// Se lee la signature y se comprueba que realmente es un fichero BMP
	fread((void *)signature, sizeof(char), 2, fData);
	if (signature[0] != 'B' || signature[1] != 'M')
	{
		printf("El fichero de la imagen no es BMP\n");
		fclose(fData);
		return -1;
	}

	// Se lee la cabecera del fichero BMP
	fread((void *)&BMPheader, sizeof(BMPHEADER), 1, fData);

	// Los datos a leer serán el alto x ancho
	dataToRead = BMPheader.iWidth * BMPheader.iHeigth;

	// Se pide memoria para las variables en el host
	// COMPLETAR
	//
	host_input = (PIXEL *)malloc(dataToRead* 3 * sizeof(unsigned char));
	host_output = (PIXEL *)malloc(dataToRead * 3 * sizeof(unsigned char));
	sec_output = (PIXEL *)malloc(dataToRead * 3 * sizeof(unsigned char));
	// Se cargan los datos de cada pixel en memoria del Host
	for (int iPos = 0; iPos < (int)dataToRead; iPos++)
	{
		fread((void *)&pixel, sizeof(PIXEL), 1, fData);
		host_input[iPos] = pixel;
	}
	fclose(fData);


	// Pedir memoria en el Device
	// COMPLETAR
	cudaMalloc((void **)&dev_input, dataToRead * 3 * sizeof(unsigned char));
	cudaMalloc((void **)&dev_output, dataToRead * 3 * sizeof(unsigned char));
	ERROR_CHECK;

	// Transferir los datos del Host al Device
	// COMPLETAR
	cudaMemcpy(dev_input, host_input, dataToRead * 3 * sizeof(unsigned char), cudaMemcpyHostToDevice);
	ERROR_CHECK;
	
	 dim3 blocks(THREADS, THREADS);
	 dim3 grid(BMPheader.iWidth / THREADS, BMPheader.iHeigth / THREADS);
	
	// Se llama al filtrado paralelo CUDA
	// COMPLETAR
	ERROR_CHECK;
	negativoParalelo <<< grid, blocks >>>(dev_input, dev_output, BMPheader.iWidth);
	// Copiar el resultado final del Device al Host
	// COMPLETAR
	cudaMemcpy(host_output, dev_output, dataToRead * 3 * sizeof(unsigned char), cudaMemcpyDeviceToHost);
	ERROR_CHECK;

	// Llamada al negativo secuencial
	negativoSecuencial(host_input, sec_output, BMPheader.iWidth, BMPheader.iHeigth);
	
	// Se abre el fichero de salida paralela
	FILE *fDataOut = fopen(argv[2], "wb");
	if (fDataOut == NULL)
	{
		printf("Error abriendo el fichero de salida\n");
		return -1;
	}

	// Se escribe la signature
	fwrite((void *)&signature, sizeof(char), 2, fDataOut);
	// Se escribe la cabecera
	fwrite((void *)&BMPheader, sizeof(BMPHEADER), 1, fDataOut);
	// Se escriben los datos del negativo paralelo
	for (int iPos = 0; iPos < (int)dataToRead; iPos++)
	{
		pixel.R = host_output[iPos].R;
		pixel.G = host_output[iPos].G;
		pixel.B = host_output[iPos].B;
		fwrite((void *)&pixel, sizeof(PIXEL), 1, fDataOut);
	}
	fclose(fDataOut);
	// Se abre el fichero de salida secuencial
	fDataOut = fopen(argv[3], "wb");
	if (fDataOut == NULL)
	{
		printf("Error abriendo el fichero de salida\n");
		return -1;
	}

	// Se escribe la signature
	fwrite((void *)&signature, sizeof(char), 2, fDataOut);
	// Se escribe la cabecera
	fwrite((void *)&BMPheader, sizeof(BMPHEADER), 1, fDataOut);
	// Se escriben los datos del negativo secuencial
	for (int iPos = 0; iPos < (int)dataToRead; iPos++)
	{
		pixel.R = sec_output[iPos].R;
		pixel.G = sec_output[iPos].G;
		pixel.B = sec_output[iPos].B;
		fwrite((void *)&pixel, sizeof(PIXEL), 1, fDataOut);
	}
	fclose(fDataOut);

	// Liberar la memoria del Host
	// COMPLETAR
	free(host_input);
	free(host_output);
	free(sec_output);
	// Librerar la memoria solicitada en el Device
	// COMPLETAR
	cudaFree(dev_input);
	cudaFree(dev_output);
	cudaDeviceReset();

	return 0;
}
