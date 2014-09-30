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
 * \par Copyright 1995-2014 Shark Development Team
 *
 * <BR><HR>
 * This file is part of Shark.
 * <http://image.diku.dk/shark/>
 *
 * Shark is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published
 * by the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * Shark is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with Shark.  If not, see <http://www.gnu.org/licenses/>.
 *
 */
//===========================================================================

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <errno.h>
#include "svm.h"
#include <math.h>
#include <string>
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

#include "llsvm.h"


void Free(int lineCount, double *values, int *arrayCounts, int **arrayIndexes, double **arrayValues, int *targets)
{
    for(int i = 0; i < lineCount; i++)
    {
        if(arrayIndexes[i] != NULL) delete[] arrayIndexes[i];
        if(arrayValues[i] != NULL) delete[] arrayValues[i];
    }
    if(arrayIndexes != NULL) delete[] arrayIndexes;
    if(arrayValues != NULL) delete[] arrayValues;
    if(arrayCounts != NULL) delete[] arrayCounts;
    if(targets != NULL) delete[] targets;
    if(values != NULL) delete[] values;
}



void Normalize(double *values, int dim, double *arrayValues, int arrayCount)
{
    double sum = 0;
    int j;

    for(j = 0; j < arrayCount; j++) sum += arrayValues[j] * arrayValues[j];
    sum = 1.0 / sqrt(sum);
    for(j = 0; j < dim; j++) values[j] *= sum;
    for(j = 0; j < arrayCount; j++) arrayValues[j] *= sum;
}



void Normalize(double *values, int dim, double *arrayValues, int arrayCount, double scale)
{
    int j;
    for(j = 0; j < dim; j++) values[j] *= scale;
    for(j = 0; j < arrayCount; j++) arrayValues[j] *= scale;
}



void GetKNN(double *means, int dim, int clusters, double *arrayValues, int *arrayIndexes, int arrayCount, int kNN, int *coorIndexes, double *coorDistances)
{
    int count = 0;
    int j, k;

    for(j = 0; j < clusters; j++)
    {
        double dist = 0;
        int arrayIndex = 0;
        double *thisMeans = means + j * dim;

        for(k = 0; k < dim; k++)
        {
            if((arrayIndex < arrayCount) && (arrayIndexes[arrayIndex] == k))
            {
                dist += (arrayValues[arrayIndex] - thisMeans[k]) * (arrayValues[arrayIndex] - thisMeans[k]);
                arrayIndex++;
            }
            else dist += thisMeans[k] * thisMeans[k]; 
        }

        int pos = count;
        while((pos > 0) && (dist < coorDistances[pos - 1])) pos--;
        
        if(pos < kNN)
        {
            if(count == kNN) count--;
            for(k = count; k >= pos + 1; k--) coorIndexes[k] = coorIndexes[k - 1], coorDistances[k] = coorDistances[k - 1];
            coorDistances[pos] = dist, coorIndexes[pos] = j, count++;
        }
    }
}



void GetDistances(double *means, int dim, int clusters, double *arrayValues, int *arrayIndexes, int arrayCount, int kNN, int *coorIndexes, double *coorDistances)
{
    int count = 0;
    int j, k;

    for(j = 0; j < kNN; j++)
    {
        double dist = 0;
        int arrayIndex = 0;
        double *thisMeans = means + coorIndexes[j] * dim;

        for(k = 0; k < dim; k++)
        {
            if((arrayIndex < arrayCount) && (arrayIndexes[arrayIndex] == k))
            {
                dist += (arrayValues[arrayIndex] - thisMeans[k]) * (arrayValues[arrayIndex] - thisMeans[k]);
                arrayIndex++;
            }
            else dist += thisMeans[k] * thisMeans[k]; 
        }
        coorDistances[j] = dist;
    }
}



void LeastSquares(double *means, int *indexes, int dim, int kNN, int *delta, double *arrayValues, int *arrayIndexes, int arrayCount, double *coef)
{
    int i, j, l;
    double *A = new double[(kNN + 1) * (kNN + 2)];
    double *b = A + (kNN + 1) * (kNN + 1);

    for(j = 0; j < kNN; j++) if(delta[j])
    {
        double *vm = means + indexes[j] * dim;

        for(i = 0; i < kNN; i++) if(delta[i])
        {
            double *x = A + (kNN + 1) * j + i;
            *x = 0;

            double *vk = means + indexes[i] * dim;
            for(l = 0; l < dim; l++) (*x) += vk[l] * vm[l];
            (*x) *= 2;
        }
        b[j] = 0;
        for(l = 0; l < arrayCount; l++) b[j] += arrayValues[l] * vm[arrayIndexes[l]];
        b[j] *= 2;
        A[(kNN + 1) * j + kNN] = 1;
    }
    for(j = 0; j < kNN; j++) if(delta[j]) A[(kNN+ 1) * kNN + j] = 1;
    A[(kNN + 2) * kNN] = 0, b[kNN] = 1;

    for(i = 0; i < kNN; i++) if(delta[i])
    {
        for(j = i + 1; j < kNN + 1; j++) if((j == kNN) || (delta[j]))
        {
            double r = A[j * (kNN + 1) + i] / A[i * (kNN + 1) + i];
            for(l = i; l < (kNN + 1); l++) if((l == kNN) || (delta[l])) A[j * (kNN + 1) + l] -= r * A[i * (kNN + 1) + l];
            b[j] -= r * b[i];
        }
    }
    memset(coef, 0, kNN * sizeof(double));
    double lambda = b[kNN] / A[kNN * (kNN + 2)];
    for(i = kNN - 1; i >= 0; i--) if(delta[i])
    {
        double t = b[i] - A[i * (kNN + 1) + kNN] * lambda;
        for(j = i + 1; j < kNN; j++) if(delta[j]) t -= A[i * (kNN + 1) + j] * coef[j];
        coef[i] = t / A[i * (kNN + 1) + i];
    }
    if(A != NULL) delete[] A;
}



void NonNegLeastSquares(double *means, int *indexes, int dim, int kNN, double *arrayValues, int *arrayIndexes, int arrayCount, double *coef)
{
    int i, j;
    int *P = new int[kNN];
    double *z = new double[2 * kNN + dim];
    double *w = z + kNN;
    double *resid = w + kNN;

    memset(coef, 0, kNN * sizeof(double));
    memset(P, 0, kNN * sizeof(int));
    memset(z, 0, 2 * kNN * sizeof(double));

    for(i = 0; i < kNN; i++)
    {
        double *mean = means + indexes[i] * dim;
        for(j = 0; j < arrayCount; j++) w[i] += mean[arrayIndexes[j]] * arrayValues[i];
    }

    int iter = 0, itmax = 3 * kNN;
    double tol = 1e-8;

    int stop = 0;
    while((!stop) && (iter < itmax))
    {
        iter++;

        int max = -1;
        for(i = 0; i < kNN; i++) if((!P[i]) && ((max == -1) || (w[i] > w[max]))) max = i;
        if(max == -1) max = 0;

        P[max] = 1;
        LeastSquares(means, indexes, dim, kNN, P, arrayValues, arrayIndexes, arrayCount, z);

        int innerstop = 1;
        for(i = 0; i < kNN; i++) if((P[i]) && (z[i] <= tol)) innerstop = 0;

        while((!innerstop) && (iter < itmax))
        {
            iter++;
            double alpha = 1e30;
            for(i = 0; i < kNN; i++) if((P[i]) && (z[i] <= tol))
            {
                double thisAlpha = coef[i] / (coef[i] - z[i]);
                if(thisAlpha < alpha) alpha = thisAlpha;
            }
            for(i = 0; i < kNN; i++) coef[i] += alpha * (z[i] - coef[i]);
            for(i = 0; i < kNN; i++) if((P[i]) && (fabs(coef[i]) < tol)) P[i] = 0;
            LeastSquares(means, indexes, dim, kNN, P, arrayValues, arrayIndexes, arrayCount, z);

            innerstop = 1;
            for(i = 0; i < kNN; i++) if((P[i]) && (z[i] <= tol)) innerstop = 0;
        }

        memcpy(coef, z, kNN * sizeof(double));
        memset(resid, 0, dim * sizeof(double));
        for(i = 0; i < arrayCount; i++) resid[arrayIndexes[i]] = arrayValues[i];
        for(i = 0; i < kNN; i++)
        {
            double *mean = means + indexes[i] * dim;
            for(j = 0; j < dim; j++) resid[j] -= mean[j] * coef[i];
        }

        memset(w, 0, kNN * sizeof(double));
        for(i = 0; i < kNN; i++)
        {
            double *mean = means + indexes[i] * dim;
            for(j = 0; j < dim; j++) w[i] += mean[j] * resid[j];
        }
        stop = 1;
        for(i = 0; i < kNN; i++) if((!P[i]) || (w[i] > tol)) stop = 0;
    }
    if(P != NULL) delete[] P;
    if(z != NULL) delete[] z;
}



void CalculateWeights(double *means, int dim, int clusters, double *arrayValues, int *arrayIndexes, int arrayCount, double *coorWeights, int *coorIndexes, int kNN, int useDist, double distCoef)
{
    GetKNN(means, dim, clusters, arrayValues, arrayIndexes, arrayCount, kNN, coorIndexes, coorWeights);

    if(useDist)
    {
        if((coorWeights[0] < 1e-9) && (useDist == 1))
        {
            coorWeights[0] = 1;
            memset(coorWeights + 1, 0, (kNN - 1) * sizeof(double));
        }
        else
        {
            int k;
            if(useDist == 1) for(k = 0; k < kNN; k++) coorWeights[k] = 1.0 / coorWeights[k];
            else if(useDist == 2) for(k = 0; k < kNN; k++) coorWeights[k] = exp(-coorWeights[k] * distCoef);

            double sum = 0;
            for(k = 0; k < kNN; k++) sum += coorWeights[k];
            for(k = 0; k < kNN; k++) coorWeights[k] /= sum;
        }
    }
    else NonNegLeastSquares(means, coorIndexes, dim, kNN, arrayValues, arrayIndexes, arrayCount, coorWeights);
}



void TrainLLSVM(double **arrayValues, int **arrayIndexes, int *arrayCounts, int dim, int lineCount, double *coorWeights, int *coorIndexes, int kNN, int *targets, int classIndex, double *weights, double *bias, int clusters, int iterations, double lambda, int t0, int skip, double biasScale)
{
    int t = t0, skipCount = skip;
    int m, l, i, k;

    memset(bias, 0, clusters * sizeof(double));
    memset(weights, 0, dim * clusters * sizeof(double));

    double *norms = new double[clusters];

    for(k = 0; k < iterations; k++)
    {
        for(i = 0; i < lineCount; i++)
        {
            t++;
            double alpha = (1.0 / (lambda * t));

            double *coor = coorWeights + i * kNN;
            int *indexes = coorIndexes + i * kNN;

            int y_k = (targets[i] == classIndex) ? 1 : -1;

            double sum = 0;
            for(m = 0; m < kNN; m++)
            {
                sum += bias[indexes[m]] * coor[m];
                for(l = 0; l < arrayCounts[i]; l++) sum += arrayValues[i][l] * weights[indexes[m] * dim + arrayIndexes[i][l]] * coor[m];
            }

            if(y_k * sum < 1)
            {
                for(m = 0; m < kNN; m++)
                {
                    bias[indexes[m]] += alpha * y_k * biasScale * coor[m];
                    for(l = 0; l < arrayCounts[i]; l++) weights[indexes[m] * dim + arrayIndexes[i][l]] += alpha * y_k * arrayValues[i][l] * coor[m];
                }
            }

            skipCount--;
            if(skipCount < 0)
            {
                double ratio = 1 - alpha * lambda * skip;
                for(m = 0; m < dim * clusters; m++) weights[m] *= ratio;
                skipCount = skip;
            }
        }
    }
    if(norms != NULL) delete[] norms;
}


unsigned int seed = 0;
void MySetRand(int setseed)
{
    seed = setseed;
}


int MyRand()
{
  seed = seed * 214013 + 2531011;
  return (seed>>16) & 0x7fff;
}



void KMeansClustering(double *means, int clusters, double *values, int lineCount, int dim, int iterations)
{
    int clusterCount = 0;
    int *meanIndexes = new int[clusters];
    int i, j, k, l;

    while(clusterCount != clusters)
    {
        int clusterIndex = MyRand() % lineCount;
        int found = 0;
        for(i = 0; (i < clusterCount) && (!found); i++) if(meanIndexes[i] == clusterIndex) found = 1;

        if(!found)
        {
            meanIndexes[clusterCount] = clusterIndex;
            memcpy(means + clusterCount * dim, values + clusterIndex * dim, dim * sizeof(double));
            clusterCount++;
        }
    }
    if(meanIndexes != NULL) delete[] meanIndexes;

    int *clusterIndexes = new int[lineCount];
    int *clusterCounts = new int[clusters];

    for(j = 0; j < iterations; j++)
    {
        memset(clusterCounts, 0, clusters * sizeof(int));
        for(i = 0; i < lineCount; i++)
        {
            int bestIndex = 0;
            double bestDistance = 0;
            double *thisValues = values + i * dim;

            for(l = 0; l < clusters; l++)
            {
                double dist = 0;
                double *thisMeans = means + l * dim;

                for(k = 0; k < dim; k++) dist += (thisMeans[k] - thisValues[k]) * (thisMeans[k] - thisValues[k]);
                if((!l) || (dist < bestDistance)) bestIndex = l, bestDistance = dist;
            }
            clusterIndexes[i] = bestIndex;
            clusterCounts[bestIndex]++;
        }
        memset(means, 0, clusters * dim * sizeof(double));
        for(i = 0; i < lineCount; i++)
        {
            double *thisMean = means + clusterIndexes[i] * dim;
            double *thisValues = values + i * dim;
            for(k = 0; k < dim; k++) thisMean[k] += thisValues[k];
        }
        for(i = 0; i < clusters; i++)
        {
            double *thisMean = means + i * dim;
            if(clusterCounts[i]) for(k = 0; k < dim; k++) thisMean[k] /= clusterCounts[i];
        }
    }
    if(clusterCounts != NULL) delete[] clusterCounts;
    if(clusterIndexes != NULL) delete[] clusterIndexes;
}



int Load(const char *fileName, double **values, int **arrayCounts, int ***arrayIndexes, double ***arrayValues, int **targets, int *labels, int *dimensions, int maxIndex)
{
    FILE *f;
    int max = 10000, maxline = 100000;

    char *buf = new char[maxline];

    f = fopen(fileName, "r");
    if(f == NULL)
    {
        *arrayCounts = NULL, *arrayIndexes = NULL, *arrayValues = NULL, *targets = NULL;
        return(0);
    }
    int lineCount = 0;
    while((fgets(buf, maxline, f) != NULL) && (strlen(buf) > 1)) lineCount++;
    fseek(f, 0, SEEK_SET);

    if(labels != NULL) *labels = 0;
    *arrayCounts = new int [lineCount];
    *arrayIndexes = new int *[lineCount];
    *arrayValues = new double *[lineCount];
    *targets = new int[lineCount];

    int *thisIndexes = new int[max];
    double *thisValues = new double[max];

    int index = 0;
    while((fgets(buf, maxline, f) != NULL) && (strlen(buf) > 1))
    {
        buf[strlen(buf) - 1] = 0;
        const char *data = buf;

        int target = atoi(data);
        if(target == -1) target = 2;
        if((labels != NULL) && (target > *labels)) *labels = target;
        (*targets)[index] = target - 1;

        data = strchr(data, 32);
        if(data != NULL) data++;

        int count = 0;
        while((data != NULL) && (data[0] >= '0') && (data[0] <= '9'))
        {
            thisIndexes[count] = atoi(data) - 1;
            if((dimensions != NULL) && (thisIndexes[count] + 1 > *dimensions)) *dimensions = thisIndexes[count] + 1;

            data = strchr(data, 58);
            if(data != NULL)
            {
                data++;
                thisValues[count] = atof(data);
                if((!maxIndex) || (thisIndexes[count] < maxIndex)) count++;
            }
            data = strchr(data, 32);
            if(data != NULL) data++;
        }
        (*arrayCounts)[index] = count;
        (*arrayIndexes)[index] = new int[count];
        (*arrayValues)[index] = new double [count];
        memcpy((*arrayIndexes)[index], thisIndexes, count * sizeof(int));

        memcpy((*arrayValues)[index], thisValues, count * sizeof(double));
        index++;
    }
    fclose(f);

    int dim = ((dimensions == NULL) ? maxIndex : (*dimensions)), i, j;
    *values = new double[dim * lineCount]; 
    memset(*values, 0, dim * lineCount * sizeof(double));
    for(i = 0; i < lineCount; i++) for(j = 0; j < (*arrayCounts)[i]; j++) (*values)[i * dim + (*arrayIndexes)[i][j]] = (*arrayValues)[i][j];

    if(buf != NULL) delete[] buf;
    if(thisIndexes != NULL) delete[] thisIndexes;
    if(thisValues != NULL) delete[] thisValues;

    return(lineCount);
}

    

struct LLSVMModel *LoadModel(const char *fileName, 
                     int *lineCount)
{
    /*
    FILE *f;
    int max = 10000, maxline = 100000;

    char *buf = new char[maxline];

    f = fopen(fileName, "r");
    if(f == NULL)
    {
        *arrayCounts = NULL, *arrayIndexes = NULL, *arrayValues = NULL, *targets = NULL;
        return(0);
    }
    int lineCount = 0;
    while((fgets(buf, maxline, f) != NULL) && (strlen(buf) > 1)) lineCount++;
    fseek(f, 0, SEEK_SET);

    if(labels != NULL) *labels = 0;
    *arrayCounts = new int [lineCount];
    *arrayIndexes = new int *[lineCount];
    *arrayValues = new double *[lineCount];
    *targets = new int[lineCount];

    int *thisIndexes = new int[max];
    double *thisValues = new double[max];

    int index = 0;
    while((fgets(buf, maxline, f) != NULL) && (strlen(buf) > 1))
    {
        buf[strlen(buf) - 1] = 0;
        const char *data = buf;

        int target = atoi(data);
        if(target == -1) target = 2;
        if((labels != NULL) && (target > *labels)) *labels = target;
        (*targets)[index] = target - 1;

        data = strchr(data, 32);
        if(data != NULL) data++;

        int count = 0;
        while((data != NULL) && (data[0] >= '0') && (data[0] <= '9'))
        {
            thisIndexes[count] = atoi(data) - 1;
            if((dimensions != NULL) && (thisIndexes[count] + 1 > *dimensions)) *dimensions = thisIndexes[count] + 1;

            data = strchr(data, 58);
            if(data != NULL)
            {
                data++;
                thisValues[count] = atof(data);
                if((!maxIndex) || (thisIndexes[count] < maxIndex)) count++;
            }
            data = strchr(data, 32);
            if(data != NULL) data++;
        }
        (*arrayCounts)[index] = count;
        (*arrayIndexes)[index] = new int[count];
        (*arrayValues)[index] = new double [count];
        memcpy((*arrayIndexes)[index], thisIndexes, count * sizeof(int));

        memcpy((*arrayValues)[index], thisValues, count * sizeof(double));
        index++;
    }
    fclose(f);

    int dim = ((dimensions == NULL) ? maxIndex : (*dimensions)), i, j;
    *values = new double[dim * lineCount]; 
    memset(*values, 0, dim * lineCount * sizeof(double));
    for(i = 0; i < lineCount; i++) for(j = 0; j < (*arrayCounts)[i]; j++) (*values)[i * dim + (*arrayIndexes)[i][j]] = (*arrayValues)[i][j];

    if(buf != NULL) delete[] buf;
    if(thisIndexes != NULL) delete[] thisIndexes;
    if(thisValues != NULL) delete[] thisValues;

    return(lineCount);
    */
}




void trainLLSVM(const char *trainFile,
                const char *modelFile,
               const double scale, 
               const int kmeansPerLabel, 
               const int kNN, 
               const int svmIterations, 
               const double distCoef, 
               const int jointClustering)
{
    const double svmLambda = 1e-5;
    const double svmBiasScale = 0;
    const int svmSkip = 200;
    const int svmT0 = 100;
    const int normalize = 0;
    const int useDist = 2;
    const int kmeansIterations = 0;

    int i, j, k, m;
    int labels, dim = 0, lineCount, **arrayIndexes, *arrayCounts, *targets;
    double **arrayValues, *values;

    MySetRand(0);
    printf("loading training file..\n");
    lineCount = Load(trainFile, &values, &arrayCounts, &arrayIndexes, &arrayValues, &targets, &labels, &dim, 0);
    int kmeansClusters = kmeansPerLabel * labels;

    printf("normalising..\n");
    if(normalize) for(i = 0; i < lineCount; i++) Normalize(values + i * dim, dim, arrayValues[i], arrayCounts[i]);
    else for(i = 0; i < lineCount; i++) Normalize(values + i * dim, dim, arrayValues[i], arrayCounts[i], scale);

    double *means = new double[dim * kmeansClusters];
    double *coorWeights = new double[lineCount * kNN];
    int *coorIndexes = new int[lineCount * kNN];
    double **bias = new double *[labels];
    double **weights = new double *[labels];
    double *posValues = new double[dim * lineCount];

    printf("k-means clustering..\n");

    if(jointClustering) KMeansClustering(means, kmeansClusters, values, lineCount, dim, kmeansIterations);
    else for(i = 0; i < labels; i++)
    {
        int posCount = 0;
        for(j = 0; j < lineCount; j++) if(targets[j] == i)
        {
            memcpy(posValues + posCount * dim, values + j * dim, dim * sizeof(double));
            posCount++;
        }
        KMeansClustering(means + i * dim * kmeansPerLabel, kmeansPerLabel, posValues, posCount, dim, kmeansIterations);
    }
    if(posValues != NULL) delete[] posValues;

    printf("calculating coefficients..\n");
    for(j = 0; j < lineCount; j++) CalculateWeights(means, dim, kmeansClusters, arrayValues[j], arrayIndexes[j], arrayCounts[j], coorWeights + j * kNN, coorIndexes + j * kNN, kNN, useDist, distCoef);

    printf("training svm..\n");
    for(i = 0; i < labels; i++)
    {
        bias[i] = new double[kmeansClusters];
        weights[i] = new double[dim * kmeansClusters];

        TrainLLSVM(arrayValues, arrayIndexes, arrayCounts, dim, lineCount, coorWeights, coorIndexes, kNN, targets, i, weights[i], bias[i], kmeansClusters, svmIterations, svmLambda, svmT0, svmSkip, svmBiasScale);
    }

    // save model file
    LLSVMModel llsvm_model;
    llsvm_model.nModels = labels;
    llsvm_model.kNN = kNN;
    llsvm_model.kmeansClusters = kmeansClusters ;
    llsvm_model.means = means;
    llsvm_model.coorIndexes = coorIndexes;
    llsvm_model.coorWeights = coorWeights ;
    llsvm_model.distCoef = distCoef;
    llsvm_model.bias = bias;
    llsvm_model.weights = weights;
    llsvm_model.dim = dim;

    
    if (LLSVMSaveModel(modelFile, &llsvm_model) != 0)
    {
        // throw bad bad error
        printf("Bad error while saving.");
        exit(-1);
    }
    
    if(coorIndexes != NULL) delete[] coorIndexes;
    if(coorWeights != NULL) delete[] coorWeights;
    Free(lineCount, values, arrayCounts, arrayIndexes, arrayValues, targets);
}

void predictLLSVM(const char *testFile, 
             const char *modelFile,
               const double scale, 
               const int kmeansPerLabel, 
               const int svmIterations, 
               const int jointClustering)
{
    const double svmLambda = 1e-5;
    const double svmBiasScale = 0;
    const int svmSkip = 200;
    const int svmT0 = 100;
    const int normalize = 0;
    const int useDist = 2;
    const int kmeansIterations = 0;

    int i, j, k, m;
    int labels, dim = 0, lineCount, **arrayIndexes, *arrayCounts, *targets;
    double **arrayValues, *values;

    MySetRand(0);

    printf("loading model file..\n");
    struct LLSVMModel *llsvm_model;
    llsvm_model = LoadModel(testFile, &lineCount);

    printf("loading test file..\n");
    lineCount = Load(testFile, &values, &arrayCounts, &arrayIndexes, &arrayValues, &targets, NULL, NULL, dim);

    // =-- MODEL
    int kNN = 5;
    int kmeansClusters = kmeansPerLabel * labels;
    double *means = new double[dim * kmeansClusters];
    int *coorIndexes = new int[lineCount * kNN];
    double *coorWeights = new double[lineCount * kNN];
    double distCoef = 0.1;
    double **bias = new double *[labels];
    double **weights = new double *[labels];


    printf("evaluating..\n");
    int correct = 0;
    for(i = 0; i < lineCount; i++)
    {
        if (normalize) Normalize(values + i * dim, dim, arrayValues[i], arrayCounts[i]);
        else Normalize(values + i * dim, dim, arrayValues[i], arrayCounts[i], scale);
        int bestIndex = 0;
        double bestResp = 0;

        CalculateWeights(means, 
                         dim, 
                         kmeansClusters, 
                         arrayValues[i], 
                         arrayIndexes[i], 
                         arrayCounts[i], 
                         coorWeights, 
                         coorIndexes, 
                         kNN, 
                         useDist, 
                         distCoef);

        for(j = 0; j < labels; j++)
        {
            double sum = 0;
            for(m = 0; m < kNN; m++)
            {
                sum += bias[j][coorIndexes[m]] * coorWeights[m];
                for(k = 0; k < arrayCounts[i]; k++) sum += arrayValues[i][k] * weights[j][coorIndexes[m] * dim + arrayIndexes[i][k]] * coorWeights[m];
            }
            if((!j) || (sum > bestResp)) bestIndex = j, bestResp = sum;
        }
        if(targets[i] == bestIndex) correct++;
    }

    if(coorIndexes != NULL) delete[] coorIndexes;
    if(coorWeights != NULL) delete[] coorWeights;

    for(i = 0; i < labels; i++)
    {
        if(weights[i] != NULL) delete[] weights[i];
        if(bias[i] != NULL) delete[] bias[i];
    }
    if(weights != NULL) delete[] weights;
    if(bias != NULL) delete[] bias;
    if(means != NULL) delete[] means;

    printf("score  %f\n", correct /(double)lineCount);
    Free(lineCount, values, arrayCounts, arrayIndexes, arrayValues, targets);
}


int LLSVMSaveModel(const char *model_file_name, const LLSVMModel *model)
{
    FILE *fp = fopen(model_file_name,"w");
    if(fp==NULL) return -1;

    char *old_locale = strdup(setlocale(LC_ALL, NULL));
    setlocale(LC_ALL, "C");

    fprintf(fp,"svm_type llsvm\n");
    fprintf(fp, "labels %d\n", model->nModels);
    fprintf(fp,"kNN %d\n", model->kNN);
    fprintf(fp,"kmeansClusters %d\n", model->kmeansClusters);
    fprintf(fp,"distCoef %f\n", model->distCoef);

    fprintf(fp, "SV\n");
    fprintf(fp, "weights\n");
    for (int i = 0; i < model->nModels; i++)
    {
        for(int m = 0; m < model->dim * model->kmeansClusters; m++)
        {
            fprintf(fp, "%.16g ",model->bias[i][m]);
        }
        fprintf(fp, "\n");
    }    
    fprintf(fp, "bias\n");
    for (int i = 0; i < model->nModels; i++)
    {
        for(int m = 0; m < model->kmeansClusters; m++)
        {
            fprintf(fp, "%.16g ",model->bias[i][m]);
        }
        fprintf(fp, "\n");
    }    
/*
    int kNN;
    int kmeansClusters;
    double *means;
    int *coorIndexes;
    double *coorWeights;
    double **bias;
    double **weights;

    int nr_class = model->nr_class;
    int l = model->l;
    fprintf(fp, "nr_class %d\n", nr_class);
    fprintf(fp, "total_sv %d\n",l);
    
    {
        fprintf(fp, "rho");
        for(int i=0;i<nr_class*(nr_class-1)/2;i++)
            fprintf(fp," %g",model->rho[i]);
        fprintf(fp, "\n");
    }
    
    if(model->label)
    {
        fprintf(fp, "label");
        for(int i=0;i<nr_class;i++)
            fprintf(fp," %d",model->label[i]);
        fprintf(fp, "\n");
    }

    if(model->probA) // regression has probA only
    {
        fprintf(fp, "probA");
        for(int i=0;i<nr_class*(nr_class-1)/2;i++)
            fprintf(fp," %g",model->probA[i]);
        fprintf(fp, "\n");
    }
    if(model->probB)
    {
        fprintf(fp, "probB");
        for(int i=0;i<nr_class*(nr_class-1)/2;i++)
            fprintf(fp," %g",model->probB[i]);
        fprintf(fp, "\n");
    }

    if(model->nSV)
    {
        fprintf(fp, "nr_sv");
        for(int i=0;i<nr_class;i++)
            fprintf(fp," %d",model->nSV[i]);
        fprintf(fp, "\n");
    }

    fprintf(fp, "SV\n");
    const double * const *sv_coef = model->sv_coef;
    const svm_node * const *SV = model->SV;

    for(int i=0;i<l;i++)
    {
        for(int j=0;j<nr_class-1;j++)
            fprintf(fp, "%.16g ",sv_coef[j][i]);

        const svm_node *p = SV[i];

        if(param.kernel_type == PRECOMPUTED)
            fprintf(fp,"0:%d ",(int)(p->value));
        else
            while(p->index != -1)
            {
                fprintf(fp,"%d:%.8g ",p->index,p->value);
                p++;
            }
        fprintf(fp, "\n");
    }
*/
    setlocale(LC_ALL, old_locale);
    free(old_locale);

    if (ferror(fp) != 0 || fclose(fp) != 0) return -1;
    else return 0;
}

