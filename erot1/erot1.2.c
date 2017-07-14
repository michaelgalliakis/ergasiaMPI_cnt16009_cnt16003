/*
 * Athanasios Rokopoulos (cnt16009)
 * Michael Galliakis (cnt16003)
 * MPI - Erotima A
 * Pollaplasiasmos C=AxB dyo 
 * pinakon diastasis NxN (row-based)
 * Date: 5/06/2017 
 */

#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"
#define N 12     //Σταθερά N για τις διαστάσεις των πινάκων.(Το Ν πρέπει να είναι πολλαπλάσιο του p)
#define ROOT 0   //Σταθερή τιμή για το rank του 0 ώστε να αναφερόμαστε σε αυτόν με το όνομα ROOT
#define TAG1 100 //Σταθερή τιμή για το TAG που χρειάζεται για τα send-receive...
int main(int argc, char** argv)
{              
    //Δηλώνουμε τους 3 πίνακες:
    int *matrixA ; 
    int *matrixB ;
    int *matrixC ;       
    
    int rank,size,root;     
    //Αρχικοποιείται το MPI 
    MPI_Init(&argc, &argv);
    //Παίρνει ο κάθε επεξεργαστής-υπολογιστής το rank του
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    //Παίρνει ο κάθε επεξεργαστής-υπολογιστής το πλήθος των επεξεργαστών.
    MPI_Comm_size(MPI_COMM_WORLD, &size);          
    //Δηλώνουμε πιο rank έχει ο τελευταίος επεξεργαστής στην σειρά.
    const int LAST = size-1 ;
    MPI_Status status ;	
    int k,j,i ;    
    int rowIndex ;
    
    if (rank == ROOT) //Ο "0" μόνο
    {        
        //Δημιουργεί 3 πίνακες N x N
        matrixA = malloc(N * N *sizeof(int)) ;
        matrixB = malloc(N * N *sizeof(int)) ;
        matrixC = malloc(N * N *sizeof(int)) ;
        //Και στην συνέχεια τους καταχωρεί τιμές, της μορφής: //0,1,2,3
        for (k=0; k<N; k++)                                   //1,2,3,4
        {                                                     //2,3,4,5
            rowIndex = k*N ;                                  //3,4,5,6
            for (j=0; j<N; j++) {
                matrixA[rowIndex+j] = k+j;
                matrixB[rowIndex+j] = k+j;   
                //matrixC[rowIndex+j] = 0 ;
                printf("[%d] ", matrixA[rowIndex+j]);
                //printf("mB[%d][%d]=%d | ", k,j, matrixB[rowIndex+j]);
            }
            printf("\n");		
        }			
    }

    int quota = N/size ;    //Υπολογίζεται το μερίδιο που θα έχει ο κάθε επεξεργαστής
    int quotaCrowd = quota*N ; //Υπολογίζεται το πλήθος αριθμών που θα έχει ο κάθε επεξεργαστής
    int q,w ;
    
    //Δημιουργεί κάθε επεξεργαστής 3 πίνακες μεγέθους quota X N
    int *matrixA_quotaRows = malloc(quotaCrowd*sizeof(int)) ; 
    int *matrixB_quotaRows = malloc(quotaCrowd*sizeof(int)) ;        
    int *matrixC_quotaRows = malloc(quotaCrowd*sizeof(int)) ;
    
    //O "0" μοιράζει τα περιεχόμενα του matrixA και matrixB στους τοπικούς πίνακες του κάθε επεξεργαστή...
    MPI_Scatter(matrixA, quotaCrowd, MPI_INT, matrixA_quotaRows, quotaCrowd, MPI_INT, ROOT, MPI_COMM_WORLD);
    MPI_Scatter(matrixB, quotaCrowd, MPI_INT, matrixB_quotaRows, quotaCrowd, MPI_INT, ROOT, MPI_COMM_WORLD);
    int rowIndexAandC,rowIndexB ;    
    
    //Με βάση τον αλγόριθμο, ο κάθε επεξεργαστής υπολογίζει τις τιμές του δικού του τοπικού (υπο) πίνακα C[matrixC_quotaRows]
    for(k=0;k<quota;k++){        
        rowIndexB = k*N ;
        for(j=0;j<quota;j++){
            rowIndexAandC = j*N ;            
            for(i=0;i<N;i++){
                matrixC_quotaRows[rowIndexAandC+i] += matrixA_quotaRows[rowIndexAandC+k+(rank*quota)] * matrixB_quotaRows[rowIndexB+i] ;
                //printf("{%d}[%d] ",rank,matrixC_quotaRows[rowIndexAandC+i]);
            }
            //printf("\n") ;        
        }
    }          
    
    for(w=1;w<size;w++) //Για όσο είναι το πλήθος των επεξεργαστών -1 (Γιατί έγινε στο προηγούμενο βήμα...)
    {
        //Στέλνει ο κάθε επεξεργαστής στον προηγούμενο του γείτονα και παραλαμβάνει από τον επόμενο του γείτονα
        //(Τοπολογία δακτυλίου), το περιεχόμενο των γραμμών που αναλογούν σε κάθε επεξεργαστή από τον πίνακα B (με βάση τον row-based).
        MPI_Send(matrixB_quotaRows,quotaCrowd,MPI_INT, (rank>0)?(rank-1):LAST,TAG1, MPI_COMM_WORLD);
        MPI_Recv(matrixB_quotaRows,quotaCrowd,MPI_INT, (rank+1)%size,TAG1,MPI_COMM_WORLD,&status);
        //Με βάση τον αλγόριθμο, ο κάθε επεξεργαστής υπολογίζει τις τιμές του δικού του τοπικού (υπο) πίνακα C[matrixC_quotaRows]
        //με βάση τις νέες τιμές που έχει μόλις πάρει από τον γείτονα του όπου αφορούν τον πίνακα B.
        for(k=0;k<quota;k++){        
            rowIndexB = k*N ;
            for(j=0;j<quota;j++){
                rowIndexAandC = j*N ;            
                for(i=0;i<N;i++){
                    matrixC_quotaRows[rowIndexAandC+i] += matrixA_quotaRows[rowIndexAandC+k+((rank*quota)+(w*quota))%N] * matrixB_quotaRows[rowIndexB+i] ;
                    //printf("{%d}[%d] ",rank,k+((rank*quota)+(w*quota))%N);
                }
                //printf("\n") ;        
            }
        }                                
    }	
    //Εφόσον έχουν τελειώσει όλα τα προηγούμενα βήματα, ο "0" μαζεύει τα περιεχόμενα όλων των (υπο) πινάκων C[matrixC_quotaRows] του κάθε
    //επεξεργαστή στον δικό του πίνακα C.
    MPI_Gather(matrixC_quotaRows, quotaCrowd, MPI_INT,matrixC, quotaCrowd, MPI_INT, ROOT, MPI_COMM_WORLD);
    
    if (rank == ROOT) ////Ο "0" μόνο
    {        
        printf("Result:\n");          //Εμφανίζει τα αποτελέσματα.
        for (k=0; k<N; k++) {         //Δηλαδή τα περιεχόμενα του πίνακα C, ο οποίος έχει τις υπολογισμένες 
            rowIndex = k*N ;          //(παράλληλα) τιμές από τον πολλαπλασιασμό του πίνακα Α με τον πίνακα B.
            for (j=0; j<N; j++) 		
                printf("[%d] ",matrixC[rowIndex+j]);
            printf("\n") ;
    	}
    }
    
    MPI_Finalize(); //Τερματίζει το MPI
    return 0 ;
}
