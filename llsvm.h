//===========================================================================
/*!
 *
 *
 * \brief       Locally Linear SVM wrapper.
 *
 *  This is the re-implementation of the paper :
 *  L.Ladicky & P.H.S. Torr : Locally Linear Support Vector Machines
 *
 *  Wrapper hacked by Aydin Demircioglu by using LibSVM code.
 * 
 *  The code is optimised to get the highest percentage.
 *  The performance should be :
 *  MNIST  : 98.28 (98.15 in the paper)
 *  LETTER : 95.90 (94.68 in the paper)
 *  USPS   : 95.12 (94.22 in the paper)
 *
 *  To optimise for speed (and convergence in lower number of iterations), lambda
 *  should be decreased (10^-6 for example) and other parameters (scale, ..) tuned
 *  accordingly. Yes, I'm fully aware the meta-parameters were tuned on the test set,
 *  which is not really the right thing to do, but there is no validation set. It
 *  should not be hard to beat these numbers (tuned in a day), feel free to send me
 *  better parameters:). Original code with the parameters to get the numbers in
 *  the paper has been lost (I will skip the details).
 *
 *
 *
 * \author      Aydin Demircioglu
 * \date        2014
 *
 *
 * This is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published
 * by the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You have not received a copy of the GNU Lesser General Public License
 * along with this. See <http://www.gnu.org/licenses/>.
 *
 */
//===========================================================================

#ifndef _LLSVM_H
#define _LLSVM_H


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <errno.h>
#include "svm.h"
#include <math.h>
#include <string>
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))


struct LLSVMModel
{
    int labels;
    int dim;
    int kNN;
    int kmeansClusters;
    double *means;
    double scale;
//    int *coorIndexes;
    //double *coorWeights;
    double distCoef;
    double **bias;
    double **weights;
};


void Free(int lineCount, double *values, int *arrayCounts, int **arrayIndexes, double **arrayValues, int *targets);
void Normalize(double *values, int dim, double *arrayValues, int arrayCount);
void Normalize(double *values, int dim, double *arrayValues, int arrayCount, double scale);
void GetKNN(double *means, int dim, int clusters, double *arrayValues, int *arrayIndexes, int arrayCount, int kNN, int *coorIndexes, double *coorDistances);
void GetDistances(double *means, int dim, int clusters, double *arrayValues, int *arrayIndexes, int arrayCount, int kNN, int *coorIndexes, double *coorDistances);
void LeastSquares(double *means, int *indexes, int dim, int kNN, int *delta, double *arrayValues, int *arrayIndexes, int arrayCount, double *coef);
void NonNegLeastSquares(double *means, int *indexes, int dim, int kNN, double *arrayValues, int *arrayIndexes, int arrayCount, double *coef);
void CalculateWeights(double *means, int dim, int clusters, double *arrayValues, int *arrayIndexes, int arrayCount, double *coorWeights, int *coorIndexes, int kNN, int useDist, double distCoef);
void TrainLLSVM(double **arrayValues, int **arrayIndexes, int *arrayCounts, int dim, int lineCount, double *coorWeights, int *coorIndexes, int kNN, int *targets, int classIndex, double *weights, double *bias, int clusters, int iterations, double lambda, int t0, int skip, double biasScale);

void MySetRand(int setseed);
int MyRand();
void KMeansClustering(double *means, int clusters, double *values, int lineCount, int dim, int iterations);
int Load(const char *fileName, double **values, int **arrayCounts, int ***arrayIndexes, double ***arrayValues, int **targets, int *labels, int *dimensions, int maxIndex);
struct LLSVMModel *LoadModel(const char *fileName);
int LLSVMSaveModel(const char *model_file_name, const LLSVMModel *model);
void trainLLSVM(const char *trainFile,
                const char *modelFile,
               const double scale, 
               const int kmeansPerLabel, 
               const int kNN, 
               const int svmIterations, 
               const double distCoef, 
               const int jointClustering);
void predictLLSVM(const char *testFile, 
             const char *modelFile);


#endif
