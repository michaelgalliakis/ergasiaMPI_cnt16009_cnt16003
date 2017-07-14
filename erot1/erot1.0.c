#include <stdio.h>
#include "mpi.h"

int main(int argc, char** argv)
{
    int N = 4 ;
    int matrixA[N][N] ;
    int matrixB[N][N] ;
    int matrixA_oneRow[N] ;
    int matrixB_oneRow[N] ;    
    int matrixC[N][N] ;
    int matrixC_oneRow[N] ;
    
    int rank,size,root;    
    int tag1 = 100 ;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);   
    root = 0;    
    MPI_Status status ;	
    int k,j,i ;
    
    if (rank == root) 
    {
        for (k=0; k<N; k++) 
        {
            for (j=0; j<N; j++) {
                matrixA[k][j] = k+j;
                matrixB[k][j] = k+j;   
                matrixC[k][j] = 0 ;
                printf("mA[%d][%d]=%d,", k,j, matrixA[k][j]);
                printf("mB[%d][%d]=%d | ", k,j, matrixB[k][j]);
            }
            printf("\n");		
        }			
    }
    
    MPI_Scatter(matrixA, N, MPI_INT, matrixA_oneRow, N, MPI_INT, root, MPI_COMM_WORLD);
    MPI_Scatter(matrixB, N, MPI_INT, matrixB_oneRow, N, MPI_INT, root, MPI_COMM_WORLD);
    
    for(i=0;i<N;i++) 
        matrixC_oneRow[i] = matrixA_oneRow[rank] * matrixB_oneRow[i];
    
    for(i=1;i<N;i++)
    {
        MPI_Send(matrixB_oneRow,N,MPI_INT, (rank>0)?(rank-1):size-1,tag1, MPI_COMM_WORLD);
        MPI_Recv(matrixB_oneRow,N,MPI_INT, (rank+1)%size,tag1,MPI_COMM_WORLD,&status);
        for(j=0; j<N; j++){
            matrixC_oneRow[j] += matrixA_oneRow[(rank+i)%size]*matrixB_oneRow[j];            
        }
        //printf("Rank:%d, Proigoumenos:%d epomenos:%d",rank,(rank>0)?(rank-1):size-1,(rank+1)%size) ;        
    }		/*   
                      for(j=0; j<N; j++){
                      matrixC_oneRow[i]+= matrixA_oneRow[rank]*matrixB_oneRow[j];
                      printf("MyRand %d matrixC[%d]=%d\n",rank,rank,matrixA_oneRow[rank]*matrixB_oneRow[j]);		
                      }*/
    MPI_Gather(matrixC_oneRow, N, MPI_INT,matrixC, N, MPI_INT, root, MPI_COMM_WORLD);
    if (rank == root)
    {
        for (k=0; k<N; k++) {
            for (j=0; j<N; j++) 		
                printf("matrixC[%d][%d]=%d | ", k,j, matrixC[k][j]);
            printf("\n") ;
	}
    }
    return 0 ;
}
