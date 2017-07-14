/*
 * Athanasios Rokopoulos (cnt16009)
 * Michael Galliakis (cnt16003)
 * MPI - Erotima B
 * Methodos Jacobi gia epilysi
 * grammikon systimaton
 * Date: 7/06/2017 
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "mpi.h"
#define ROOT 0 //Σταθερή τιμή για το rank του 0 ώστε να αναφερόμαστε σε αυτόν με το όνομα ROOT

//Συνάρτηση για να γεμίσουμε με αρχικές τιμές τα l,n,ex,A,b,x, είτε από αρχείο είτε με default.
void fillInitialValues(const char[],int*, int*, float*,float *[],float *[], float *[]) ;
//Συναρτήσεις για διάφορα print:
void printFinalVector(float[], int) ;
void printVector(float[], int, float) ;
void printVectorDebug(int, int, float[], int);
void printNorma(int, int, float, float, float);

int main(int argc, char** argv)
{             
    int rank,size;      
    //Αρχικοποιείται το MPI 
    MPI_Init(&argc, &argv);
    //Παίρνει ο κάθε επεξεργαστής-υπολογιστής το rank του
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    //Παίρνει ο κάθε επεξεργαστής-υπολογιστής το πλήθος των επεξεργαστών.
    MPI_Comm_size(MPI_COMM_WORLD, &size);                  
    
    //Δηλώνουμε τους 6 πίνακες που χρειαζόμαστε:
    float *matrix_A ;
    float *matrix_A_RowsLoc ;
    float *vector_b ;    
    float *vector_b_elements ;    
    float *vector_x ;    
    float *vector_x_new ;            
    int k,i,j,q ; //Διάφοροι μετρητές που χρειάζονται
    int l,n ; //Το λ και n του Jacobi αλγορίθμου
    float ex ; //Το "ξ" του Jacobi αλγορίθμου              
    
    if (rank == ROOT) //Ο "0" μόνο
    {    
        //Δηλώνουμε όνομα αρχείου για να διαβάσουμε από εκεί τις αρχικοποιήσεις
        const char path[] = "jacobiInput.txt" ;
        //Αρχικοποιεί τα n,l,ex,matrix_A,vector_b,vector_x
        fillInitialValues(path,&n, &l, &ex, &matrix_A, &vector_b, &vector_x) ;                              
    }
    
    //Ο "0" στέλνει σε όλους τις τιμές του l,ex,n
    MPI_Bcast(&l, 1, MPI_INT, ROOT, MPI_COMM_WORLD);
    MPI_Bcast(&ex, 1, MPI_REAL, ROOT, MPI_COMM_WORLD);    
    MPI_Bcast(&n, 1, MPI_INT, ROOT, MPI_COMM_WORLD);
    
    if(rank!=ROOT) //Κάθε επεξεργαστής, εκτός του "0"
        vector_x = malloc(n *sizeof(float)) ; //Δεσμεύει μνήμη για το διάνυσμα vector_x
    
    //Ο "0" στέλνει σε όλους τις τιμές του πίνακα vector_x
    MPI_Bcast(vector_x, n, MPI_REAL, ROOT, MPI_COMM_WORLD);            
    
    int quota = n/size ; //Υπολογίζεται το μερίδιο που θα έχει ο κάθε επεξεργαστής
    int quotaCrowd = quota*n ; //Υπολογίζεται το πλήθος αριθμών που θα έχει ο κάθε επεξεργαστής
    
    //Δεσμεύετε μνήμη για το τοπικό διάνυσμα vector_x_new για κάθε επεξεργαστή
    vector_x_new = malloc(quota *sizeof(float)) ; 
    
    //Δεσμεύετε μνήμη για τον πίνακα matrix_A_RowsLoc (που είναι τοπικός [υπο]πίνακας του matrix_A για κάθε επεξεργαστή...)
    matrix_A_RowsLoc = malloc(quotaCrowd *sizeof(float)) ;
    //O "0" μοιράζει τα περιεχόμενα του matrixA στους τοπικούς [υπο]πίνακες του κάθε επεξεργαστή...
    MPI_Scatter(matrix_A, quotaCrowd, MPI_REAL, matrix_A_RowsLoc, quotaCrowd, MPI_REAL, ROOT, MPI_COMM_WORLD);    
    
    //Δεσμέυετε μνήμη για το διάνυσμα vector_b_elements (που είναι τοπικό [υπο]διάνυσμα του vector_b για κάθε επεξεργαστή...)
    vector_b_elements = malloc(quota *sizeof(float)) ;       
    //O "0" μοιράζει τα περιεχόμενα του vector_b στα τοπικά [υπο]διανύσματα του κάθε επεξεργαστή...    
    MPI_Scatter(vector_b, quota, MPI_REAL, vector_b_elements, quota, MPI_REAL, ROOT, MPI_COMM_WORLD);                       
    
    float sum ;
    int rowIndex ;    
    float norma_loc, norma_all ;    
    for (k=0;k<l;k++)
    {         
        norma_loc = 0 ; //Μηδενίζει στην αρχή της κάθε επανάληψης η τοπική νόρμα
        for (q=0;q<quota;q++) // Για όσο είναι το "μερίδιο" του κάθε επεξεργαστή
        {
            rowIndex = q*n ; //Υπολογίζετε ο δείκτης της κάθε γραμμής για τον πίνακα matrix_A_RowsLoc
            
            //Στην συνέχεια γίνονται οι απαραίτητοι υπολογισμοί, με βάση τον αλγόριθμο του Jacobi
            //ώστε να βρει ο κάθε επεξεργαστής τοπικά τις νέες δικές του τιμές για τον πίνακα x.
            sum = -matrix_A_RowsLoc[rowIndex+(q+(rank*quota))] * vector_x[q+(rank*quota)] ;
            for(j=0;j<n;j++)
                sum += matrix_A_RowsLoc[rowIndex+j] * vector_x[j];
            vector_x_new[q] = (vector_b_elements[q]-sum)/matrix_A_RowsLoc[rowIndex+q+(rank*quota)] ;            
            
            //Σύμφωνα με τον αλγόριθμο, υπολογίζει κάθε επεξεργαστής τοπικά την νόρμα του.
            norma_loc += pow((vector_x_new[q]-vector_x[q+(rank*quota)]),2) ; //power 2
        }      
        
        //Με την Allgather, γίνεται gather και broadcast, ώστε κάθε επεξεργαστής να έχει 
        //τo νέο πίνακα x, με τις νέες υπολογισμένες τιμές που έχει βρει ο κάθε επεξεργαστής,
        //ώστε αν έχουμε βρει την λύση, να εμφανιστεί το διάνυσμα στον χρήστη και αν όχι
        //ακόμη, να χρησιμοποιηθεί αυτό το νέο διάνυσμα x στην επόμενη επανάληψη του βρόχου.
        MPI_Allgather(vector_x_new, quota, MPI_REAL, vector_x, quota, MPI_REAL,MPI_COMM_WORLD);        
        
        //Με την Allreduce, γίνεται reduce και broadcast, ώστε κάθε επεξεργαστής να έχει 
        //την συνολική νόρμα (χωρίς να έχει υπολογιστεί ακόμη η ρίζα...)
        MPI_Allreduce(&norma_loc, &norma_all, 1, MPI_REAL, MPI_SUM,MPI_COMM_WORLD);                               
        
        //Κάθε επεξεργαστής βρίσκει την ρίζα της συνολικής νόρμας 
        norma_all = sqrt(norma_all) ;        
        
        if(rank==ROOT){//Για λόγους επαλήθευσης, ο "0" μόνο, εμφανίζει κάποια μηνύματα
            //printVector(vector_x,n,norma_all) ;            
            printNorma(k,rank,norma_loc,norma_all,ex) ;   
            printVectorDebug(k,rank,vector_x,n) ;
        }
        //Εάν η νόρμα είναι μικρότερη από το "ξ" που έχει δοθεί, σημαίνει ότι υπάρχει σύγκλιση
        if (norma_all<=ex)                        
            break ;//Εάν υπάρχει σύγκλιση, τότε κάνουν όλοι οι επεξεργαστές break για να βγουν από το loop                                  
    }      
    
    if (rank==ROOT) //ο "0" μόνο
    {
        //Εμφανίζει αν υπάρχει ή δεν υπάρχει σύγκλιση και εφόσον υπάρχει
        //εμφανίζει και τον διάνυσμα x με τη λύση του γραμμικού συστήματος
        if(k==l) //Αν k=l, σημαίνει ότι δεν έγινε κάποιο break μέσα στο loop...
            printf("Δεν υπάρχει σύγκλιση!\n") ;  
        else{                       
            printf("Υπάρχει σύγκλιση! και η λύση του συστήματος είναι:\n") ;     
            printFinalVector(vector_x,n) ;
        }                
    }        
    
    MPI_Finalize(); //Τερματίζει το MPI
    return 0 ;
}

void fillInitialValues(const char path[],int *n, int *l, float *ex, float *matrix_A[],float *vector_b[], float *vector_x[])
{
    FILE *f;
    int i,j,rowIndex ;
    
    f = fopen(path,"r");
    if (f){
        printf("Βρέθηκε το αρχείο και ξεκινάμε να το διαβάζουμε!\n") ;        
        fscanf(f, "%d %d %f", n,l,ex);
        printf("n=%d, l=%d, ex=%.*f\n",*n,*l,8,*ex) ;
        
        *matrix_A = (float*)malloc( *n * *n * sizeof(float)) ;        
        printf("* * * Matrix A * * *\n") ;
        for(i=0;i<*n;i++){
            rowIndex = i* *n ;
            for(j=0;j<*n;j++){
                fscanf(f, "%f",&(*matrix_A)[rowIndex+j]);                
                printf("[%f]",(*matrix_A)[rowIndex+j]) ;
            }
            printf("\n") ;
        }
        *vector_b = (float*)malloc(*n *sizeof(float)) ;
        *vector_x = (float*)malloc(*n *sizeof(float)) ;
        printf("* * * Vector b * * *\n") ;
        for(j=0;j<*n;j++){
            fscanf(f, "%f",&(*vector_b)[j]);
            printf("[%f]",(*vector_b)[j]) ;
        }            
        printf("\n* * * Vector X * * *\n") ;
        for(j=0;j<*n;j++){
            fscanf(f, "%f",&(*vector_x)[j]);
            printf("[%f]",(*vector_x)[j]) ;
        }        
        printf("\nΤέλος αρχικοποίησης\n") ;      
        fclose(f);
    }    
    else
    {
        printf("Δεν βρέθηκε το αρχείο και γεμίζουμε με default τιμές!\n") ;  
        *n = 4;
        *l = 20;
        *ex = 0.000001 ;
        printf("n=%d, l=%d, ex=%.*f\n",*n,*l,8,*ex) ;
        
        *matrix_A = (float*) malloc(*n * *n *sizeof(float)) ;
        (*matrix_A)[0] = 10; (*matrix_A)[1] = -1 ;(*matrix_A)[2] = 2 ;(*matrix_A)[3] = 0 ;
        (*matrix_A)[4] = -1 ; (*matrix_A)[4+1] = 11 ;(*matrix_A)[4+2] = -1 ;(*matrix_A)[4+3] = 3 ;
        (*matrix_A)[8] = 2 ; (*matrix_A)[8+1] = -1 ;(*matrix_A)[8+2] = 10 ;(*matrix_A)[8+3] = -1 ;
        (*matrix_A)[12] = 0 ; (*matrix_A)[12+1] = 3 ;(*matrix_A)[12+2] = -1 ;(*matrix_A)[12+3] = 8 ;
        
        printf("* * * Matrix A * * *\n") ;
        for(i=0;i<*n;i++){
            rowIndex = i* *n ;
            for(j=0;j<*n;j++)        
                printf("[%f]",(*matrix_A)[rowIndex+j]) ;        
            printf("\n") ;
        }        
        
        *vector_b = (float*) malloc(*n *sizeof(float)) ;
        *vector_x = (float*) malloc(*n *sizeof(float)) ;
        
        (*vector_b)[0] = 6 ;
        (*vector_b)[1] = 25 ;
        (*vector_b)[2] = -11 ;
        (*vector_b)[3] = 15 ;
        
        printf("* * * Vector b * * *\n") ;
        for(j=0;j<*n;j++)            
            printf("[%f]",(*vector_b)[j]) ;                
        
        (*vector_x)[0] = 0 ;
        (*vector_x)[1] = 0 ;
        (*vector_x)[2] = 0 ;
        (*vector_x)[3] = 0 ;         
        
        printf("\n* * * Vector X * * *\n") ;
        for(j=0;j<*n;j++)        
            printf("[%f]",(*vector_x)[j]) ;
        
        printf("\nΤέλος αρχικοποίησης\n") ;        
    }
}

void printFinalVector(float vector[], int length)
{
    int i = 0;    
    for(;i<length;i++)
        printf("[%f] ",vector[i]) ;
    printf("\n") ;
}
void printVector(float vector[], int length,float nAll)
{
    int i = 0;    
    for(;i<length;i++)
        printf("[%f] ",vector[i]) ;
    printf("nAll=%.*f\n",8,nAll) ;
}
void printVectorDebug(int k, int rank,float vector[], int length)
{
    int i = 0;
    printf("k=%d, Rank=%d",k,rank) ;
    for(;i<length;i++)
        printf("[%f] ",vector[i]) ;
    printf("\n") ;
}
void printNorma(int k, int rank, float nLoc,float nAll,float ex)
{
    printf("k=%d, Rank=%d, Norma_loc=%.*f, Norma_all=%.*f, Ex=%.*f \n",k, rank,8,nLoc,8,nAll,8,ex) ;
}
