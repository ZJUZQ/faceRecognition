#ifndef FACE_COVAR_EIGEN_HPP
#define FACE_COVAR_EIGEN_HPP

#include "FACE/common_includes.hpp"

namespace FACE{

/**  Flags for cvCalcCovarMatrix
 */

/** flag for cvCalcCovarMatrix, transpose([v1-avg, v2-avg,...]) * [v1-avg,v2-avg,...] 

    The covariance matrix will be nsamples x nsamples. Such an unusual covariance matrix is used for 
    fast PCA of a set of very large vectors (see, for example, the EigenFaces technique for face recognition). 
    Eigenvalues of this "scrambled" matrix match the eigenvalues of the true covariance matrix. The "true" 
    eigenvectors can be easily calculated from the eigenvectors of the "scrambled" covariance matrix. 
*/
#define FACE_COVAR_SCRAMBLED 0

/** flag for cvCalcCovarMatrix, [v1-avg, v2-avg,...] * transpose([v1-avg,v2-avg,...]) 
 
    covar will be a square matrix of the same size as the total number of elements in each input vector. One and 
    only one of COVAR_SCRAMBLED and COVAR_NORMAL must be specified. 
 */
#define FACE_COVAR_NORMAL    1

/** flag for cvCalcCovarMatrix

    If the flag is specified, the function does not calculate mean from the input vectors but, instead, uses 
    the passed mean vector. This is useful if mean has been pre-calculated or known in advance, or if the covariance 
    matrix is calculated by parts. In this case, mean is not a mean vector of the input sub-set of vectors but 
    rather the mean vector of the whole set. 
 */
#define FACE_COVAR_USE_AVG   2

/** flag for cvCalcCovarMatrix

    If the flag is specified, the covariance matrix is scaled. In the "normal" mode, scale is 1./nsamples . 
    In the "scrambled" mode, scale is the reciprocal of the total number of elements in each input vector. 
    By default (if the flag is not specified), the covariance matrix is not scaled ( scale=1 ). 
 */
#define FACE_COVAR_SCALE     4

/** flag for cvCalcCovarMatrix, all the input vectors are stored in a single matrix, as its rows */
#define FACE_COVAR_ROWS      8

/** flag for cvCalcCovarMatrix, all the input vectors are stored in a single matrix, as its columns */
#define FACE_COVAR_COLS     16

//***********************************************************************************************************//

/** Calculates the covariance matrix of a set of vectors.

The function calcCovarMatrix calculates the covariance matrix and, optionally, the mean vector of
the set of input vectors.

    @param samples      :   samples stored as separate matrices
    @param nsamples     :   number of samples
    @param covar        :   output covariance matrix of the type ctype and square size.
    @param mean         :   input or output (depending on the flags) array as the average value of the input vectors.
    @param flags        :   operation flags as a combination of CovarFlags
    @param ctype        :   type of the matrixl; it equals 'CV_64F' by default.
*/
void calcCovarMatrix( const cv::Mat* samples, int nsamples, cv::Mat& covar, cv::Mat& mean, int flags, int ctype = CV_64F );

/** @overload

@note use cv::COVAR_ROWS or cv::COVAR_COLS flag

    @param samples  :   samples stored as rows/columns of a single matrix.
    @param covar    :   output covariance matrix of the type ctype and square size.
    @param mean     :   input or output (depending on the flags) array as the average value of the input vectors.
    @param flags    :   operation flags as a combination of cv::CovarFlags
    @param ctype    :   type of the matrixl; it equals 'CV_64F' by default.
*/
void calcCovarMatrix( cv::Mat samples, cv::Mat covar, cv::Mat& mean, int flags, int ctype = CV_64F );


}; /* namespace FACE */


#endif