#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include "math.h"
#include <float.h>
#include PY_SSIZE_T_CLEAN
#include <Python.h>

#define LINESIZE 1000

void resetMat(int row,int col,double** mat);
double** createMat(int col, int row);
int calculateRows(char* fileName);
int calculateCol(char* fileName);
void fillMat(char* fileName,double** inputMat);
void freeMemory(double** matrix ,int n);
void update(double* toChage,double* GroupOfClusters,double** inputMat);
void sumVectors(double* vector1,double* vector2);
void avgVectors(double* vector1,int cnt);
void algorithm(double** clusters, double ** inputMat, double ** GroupOfClusters);
int minIndex(double** clusters, double* victor);
double distance(double* vector1 , double* vector2);
double calculateNorm(double* vector);
int normMat(double** matrix);
double** calculateCentroids(int k, int max_itr, double epsilon, int d, int n, double** matrix, double** clusters);


/*
 *  max_itr := the maximum number of iterations of the K-means algorithm, the default value is 200.
 *  k := the number of clusters required.
 *  d := number of column of an input file.
 *  n := number of line of an input file , = number of vectors.
 */
int max_itr1;
int k1;
int d1;
int n1;

static PyObject* fit(PyObject *self, PyObject *args) {
    int max_itr;
    int k;
    int d;
    int n;
    double epsilon;
    double** matrix;
    double** clusters;

    if (!PyArg_ParseTuple(args, "iidiiOO", &k, &max_itr, &epsilon, &d, &n, &matrix, &clusters)) {
        return NULL;
    }

    max_itr1 = max_itr;
    k1 = k;
    d1 = d;
    n1 = n;
    return Py_BuildValue("i", main(k, max_itr, epsilon, d, n, matrix, clusters))
}

static PyMethodDef myMethods[] = {
        { "fit", 
        (PyCFunction)fit, METH_VARARGS, PyDoc_STR("Input: Points, Centroids, Iterations and Clusters. Output: Centroids") },
        { NULL, NULL, 0, NULL }
};

static struct PyModuleDef mykmeanssp = {
        PyModuleDef_HEAD_INIT,
        "mykmeanssp",
        "kmeans module",
        -1,
        myMethods
};

PyMODINIT_FUNC PyInit_mykmeanssp(void) {
    PyObject *m;
    m = PyModule_Create(&mykmeanssp);
    if (!m) {
        return NULL;
    }
    return m;
}

double** calculateCentroids(int k, int max_itr, double epsilon, int d, int n, double** inputMat, double** clusters) {
    int i,j;
    char* useless;
    /*
     *  groupOfClusters := group of clusters by S1,...,SK, each cluster Sj is represented by it’s
     *    centroid  which is the mean µj ∈ Rd of the cluster’s members.
     */
    double** groupOfClusters = NULL;

    /*
     * bug
     */
    if(k>n){
        printf("Invalid Input! \n");
        return 0;
    }

    assert(inputMat!=NULL);
    assert(clusters != NULL);

    /*
     *  groupOfClusters := [[0.0,0.0,0.0,0,0,0.0]
     *                     ,[0.0,0.0,0.0,0,0,0.0]]
    */
    groupOfClusters = createMat(k, n);
    assert(groupOfClusters != NULL);

    for(i=0; i<k; i++){
        for(j=0;j<n;j++){
            groupOfClusters[i][j] = 0.0;
        }
    }
    algorithm(clusters,inputMat,groupOfClusters, epsilon);

    /*
     * step 4: freeing memory
     */
    freeMemory(groupOfClusters, k);
    
    return inputMat;
}

void algorithm(double** clusters, double** inputMat, double** GroupOfClusters, double epsilon){
    int numOfItr=0;
    int x_i;
    int m_i;
    int index;
    while (numOfItr < max_itr1){
        if(normMat(clusters, epsilon)==1){
            printf("break\n");
            break;
        }
        resetMat(k1,n1,GroupOfClusters);
        /*
         * for xi, 0 < i ≤ N:
         *  Assign xi to the closest cluster Sj : argmin_Sj(xi − µj )^2 , ∀j 1 ≤ j ≤ K
         *  0 < index ≤ K
         */
        for(x_i=0;x_i<n1;x_i++){
            index = minIndex(clusters, inputMat[x_i]);
            GroupOfClusters[index][x_i] = 10;
        }
        /*
         * for µk, 0 < i ≤ K:
         *  we want to change clusters[i] , 0< i <= k
         */
        for(m_i=0;m_i<k1;m_i++){
            update(clusters[m_i], GroupOfClusters[m_i], inputMat);
        }
        numOfItr++;
    }

}
int normMat(double** matrix, double epsilon){
    int i;
    for(i=0;i<k1;i++){
        if(calculateNorm(matrix[i]) > epsilon){
            return 0;
        }
    }
    return 1;
}
double calculateNorm(double* vector){
    int i;
    double sum=0.0;
    double value;
    for(i=0;i<d1;i++){
        sum = sum + pow(vector[i],2.0);
    }
    value = sqrt(sum);
    return value;
}


void resetMat(int row,int col,double** mat){
    int i,j;
    for(i=0;i<row;i++){
        for(j=0;j<col;j++){
            mat[i][j] = 0.0;
        }
    }
}

void sumVectors(double* vector1,double* vector2){
    int i;
    for(i=0;i<d1;i++){
        vector1[i] = vector1[i] + vector2[i];
    }
}
void avgVectors(double* vector1,int cnt){
    int i;
    if(cnt == 0){
        return;
    }
    for(i=0;i<d1;i++){
        vector1[i] = vector1[i]/cnt;
    }
}
void update(double* toChage,double* GroupOfClusters,double** inputMat){
    int i;
    int cnt=0;
    /*
     * fill tochange with 0.0
     */
    for(i=0;i<d1;i++){
        toChage[i]= 0.0;
    }
    for(i=0;i<n1;i++){
        if(GroupOfClusters[i] > 5){
            cnt++;
            sumVectors(toChage,inputMat[i]);
        }
    }
    avgVectors(toChage,cnt);
}

int minIndex(double** clusters, double* victor){
    int minIndex=0;
    double minDistance=DBL_MAX;
    double tempMinDistance;
    int i;

    for(i=0;i<k1;i++){
        tempMinDistance = distance(victor, clusters[i]);
        if(tempMinDistance<minDistance) {
            minIndex = i;
            minDistance = tempMinDistance;
        }
    }
    return minIndex;
}
double distance(double* vector1 , double* vector2){
    int i;
    double sum=0.0;
    for(i=0; i < d1; i++){
        /*
         * argmin_Sj(xi − µj)^2
         */
        sum = sum + pow((vector1[i] - vector2[i]), 2.0);
    }
    return sum;
}


/*
 *  freeing 2-dimensional arrays
 */
void freeMemory(double** matrix ,int len){
    int i;
    if(matrix == NULL){
        return;
    }
    for(i = 0; i < len ; i++){
        if(matrix[i] == NULL){
            continue;
        }
        free(matrix[i]);
    }
    free(matrix);
}

/*
 *  create 2-dimensional arrays
 */
double** createMat(int col, int row){
    int i;
    double ** matrix = (double**)malloc(col* sizeof(double *));
    assert(matrix != NULL);
    for(i=0;i<col;i++){
        matrix[i]= (double*)malloc(row* sizeof(double ));
        assert(matrix[i] != NULL);
    }
    return matrix;
}
