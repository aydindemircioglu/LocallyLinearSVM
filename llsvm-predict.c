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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "llsvm.h"


int print_null(const char *s,...) {return 0;}
static int (*info)(const char *fmt,...) = &printf;


void exit_with_help()
{
        printf(
        "Usage: svm-predict [options] test_file model_file output_file\n"
        "options:\n"
    "-o train_file : if specified some derivates like primal value, dual value etc will be computed\n"
        "-b probability_estimates: whether to predict probability estimates, 0 or 1 (default 0); for one-class SVM only 0 is supported\n"
        "-q : quiet mode (no outputs)\n"
        );
        exit(1);
}



int main(int argc, char **argv)
{
    FILE *input, *output;
    int i;
    
    // parse options
    for(i=1;i<argc;i++)
    {
        if(argv[i][0] != '-') break;
        ++i;
        switch(argv[i-1][1])
        {
            case 'q':
                info = &print_null;
                i--;
                break;
            default:
                fprintf(stderr,"Unknown option: -%c\n", argv[i-1][1]);
                exit_with_help();
        }
    }

    
    if(i>=argc-2)
        exit_with_help();

    input = fopen(argv[i],"r");
    if(input == NULL)
    {
        fprintf(stderr,"can't open input file %s\n",argv[i]);
        exit(1);
    }

    output = fopen(argv[i+2],"w");
    if(output == NULL)
    {
        fprintf(stderr,"can't open output file %s\n",argv[i+2]);
        exit(1);
    }

    
    printf("Predicting..\n");

    predictLLSVM(argv[i], argv[i+1]);

    fclose(input);
    fclose(output);
    return 0;
}



