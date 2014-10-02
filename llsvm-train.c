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
#include <ctype.h>
#include <errno.h>
#include "svm.h"
#include <math.h>
#include <string>
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

#include "llsvm.h"

void print_null(const char *s) {}


void exit_with_help()
{
	printf(
	"Usage: svm-train [options] training_set_file [model_file]\n"
	"options:\n"
    "-s scale (default 1.0)\n"
    "-k kMeans per Label (default 10)\n"
    "-n neighborhood size for kNN (default 20)\n"
    "-i svm iterations (default 500)\n"
    "-d distance coefficient (default 0.1)\n"
    "-j joint clustering on/off (default 0 = off)\n"
	"-q : quiet mode (no outputs)\n"
	);
	exit(1);
}

void exit_input_error(int line_num)
{
	fprintf(stderr,"Wrong input format at line %d\n", line_num);
	exit(1);
}

void parse_command_line(int argc, char **argv, char *input_file_name, char *model_file_name);
void read_problem(const char *filename);
void do_cross_validation();

struct svm_parameter param;		// set by parse_command_line
int cross_validation;
int nr_fold;

static char *line = NULL;



extern double subsamplingAmount;
extern int walltime;
extern int savetime;





void parse_command_line(int argc, char **argv, char *input_file_name, char *model_file_name)
{
	int i;
	void (*print_func)(const char*) = NULL;	// default printing to stdout

	// default values
	param.scale = 1.0;
    param.kMeansperLabel = 10;
    param.kNN = 20;
    param.svmIterations = 500;
    param.distanceCoefficient = 0.1;
    param.jointClustering = 0;

    walltime = -1;
    savetime = -1;
	subsamplingAmount  = -1;

	// parse options
	for(i=1;i<argc;i++)
	{
		if(argv[i][0] != '-') break;
		if(++i>=argc)
			exit_with_help();
		switch(argv[i-1][1])
		{
			case 's':
				param.scale = atof(argv[i]);
				break;
			case 'k':
				param.kMeansperLabel = atoi(argv[i]);
				break;
			case 'n':
				param.kNN = atoi(argv[i]);
				break;
            case 'i':
                param.svmIterations = atoi(argv[i]);
                break;
			case 'd':
				param.distanceCoefficient = atof(argv[i]);
				break;
			case 'j':
				param.jointClustering = atoi(argv[i]);
				break;

            
            case 'a':
                    savetime = atoi(argv[i]);
                    break;
            case 'l':
                    walltime = atoi(argv[i]);
                    break;
			default:
				fprintf(stderr,"Unknown option: -%c\n", argv[i-1][1]);
				exit_with_help();
		}
	}

	svm_set_print_string_function(print_func);

	// determine filenames

	if(i>=argc)
		exit_with_help();

	strcpy(input_file_name, argv[i]);

	if(i<argc-1)
		strcpy(model_file_name,argv[i+1]);
	else
	{
		char *p = strrchr(argv[i],'/');
		if(p==NULL)
			p = argv[i];
		else
			++p;
		sprintf(model_file_name,"%s.model",p);
	}
}




int main(int argc, char **argv)
{
    char input_file_name[1024];
    char model_file_name[1024];

    parse_command_line(argc, argv, input_file_name, model_file_name);
    /*
    // TODO in distant future
    const char *error_msg;
    error_msg = llsvm_check_parameter(&prob,&param);

    if(error_msg)
    {
        fprintf(stderr,"ERROR: %s\n",error_msg);
        exit(1);
    }
*/
    // llsvm will save its model by itself
    trainLLSVM (input_file_name, 
                model_file_name,
                param.scale, 
                param.kMeansperLabel, 
                param.kNN,
                param.svmIterations,
                param.distanceCoefficient,
                param.jointClustering);
    
    svm_destroy_param(&param);
    free(line);

    return 0;
}
