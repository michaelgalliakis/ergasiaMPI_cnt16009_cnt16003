#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"
#define N 4
#define ROOT 0
int main(int argc, char** argv)
{
    int *matrixA ;
    int *matrixB ;
    int *matrixC ;
    
    int *matrixA_oneRow = malloc(N*sizeof(int)) ;
    int *matrixB_oneRow = malloc(N*sizeof(int)) ;        
    int *matrixC_oneRow = malloc(N*sizeof(int)) ;
    
    int rank,size,root;    
    int tag1 = 100 ;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);          
    MPI_Status status ;	
    int k,j,i ;    
    
    if (rank == ROOT) 
    {
        int rowIndex ;
        matrixA = malloc(N * N *sizeof(int)) ;
        matrixB = malloc(N * N *sizeof(int)) ;
        matrixC = malloc(N * N *sizeof(int)) ;
        for (k=0; k<N; k++) 
        {           
            rowIndex = k*N ;
            for (j=0; j<N; j++) {
                matrixA[rowIndex+j] = k+j;
                matrixB[rowIndex+j] = k+j;   
                //matrixC[rowIndex+j] = 0 ;
                printf("mA[%d][%d]=%d,", k,j, matrixA[rowIndex+j]);
                printf("mB[%d][%d]=%d | ", k,j, matrixB[rowIndex+j]);
            }
            printf("\n");		
        }			
    }
    
    MPI_Scatter(matrixA, N, MPI_INT, matrixA_oneRow, N, MPI_INT, ROOT, MPI_COMM_WORLD);
    MPI_Scatter(matrixB, N, MPI_INT, matrixB_oneRow, N, MPI_INT, ROOT, MPI_COMM_WORLD);
    
    for(i=0;i<N;i++)                     
        matrixC_oneRow[i] = matrixA_oneRow[rank] * matrixB_oneRow[i];        
    
    for(i=1;i<N;i++)
    {
        MPI_Send(matrixB_oneRow,N,MPI_INT, (rank>0)?(rank-1):size-1,tag1, MPI_COMM_WORLD);
        MPI_Recv(matrixB_oneRow,N,MPI_INT, (rank+1)%size,tag1,MPI_COMM_WORLD,&status);
        for(j=0; j<N; j++)
            matrixC_oneRow[j] += matrixA_oneRow[(rank+i)%size]*matrixB_oneRow[j];                                                      		        
    }	
    
    MPI_Gather(matrixC_oneRow, N, MPI_INT,matrixC, N, MPI_INT, ROOT, MPI_COMM_WORLD);
    
    if (rank == ROOT)
    {
        int rowIndex ;
        for (k=0; k<N; k++) {
            rowIndex = k*N ;
            for (j=0; j<N; j++) 		
                printf("matrixC[%d][%d]=%d | ", k,j, matrixC[rowIndex+j]);
            printf("\n") ;
	}
    }
    
    return 0 ;
}
