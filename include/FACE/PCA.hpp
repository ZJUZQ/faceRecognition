#ifndef FACE_PAC_HPP
#define FACE_PAC_HPP

#include "FACE/common_includes.hpp"

namespace FACE{

/** Principal Component Analysis. 

    The class is used to calculate a special basis for a set of vectors. The basis will consist of eigenvectors 
    of the covariance matrix calculated from the input set of vectors.

    The class PCA can also transform vectors to/from the new coordinate space defined by the basis.

    Usually, in this new coordinate system, each vector from the original set (and any linear combination of such 
    vectors) can be quite accurately approximated by taking its first few components, corresponding to the 
    eigenvectors of the largest eigenvalues of the covariance matrix. Geometrically it means that you calculate a 
    projection of the vector to a subspace formed by a few eigenvectors corresponding to the dominant eigenvalues 
    of the covariance matrix. And usually such a projection is very close to the original vector. So, you can 
    represent the original vector from a high-dimensional space with a much shorter vector consisting of the 
    projected vector's coordinates in the subspace.
 */

class PCA
{
public:
    enum Flags { 
        DATA_AS_ROW = 0, //!< indicates that the input samples are stored as matrix rows
        DATA_AS_COL = 1, //!< indicates that the input samples are stored as matrix columns
        USE_AVG     = 2  //!
    };

    /** @brief default constructor
    */
    PCA();

    /** @overload constructor

        @param data     :       input samples stored as matrix rows or matrix columns.
        @param mean     :       optional mean value; if the matrix is empty (@c noArray()), the mean is computed from the data.
        @param flags    :       operation flags; currently the parameter is only used to specify the data layout (PCA::Flags)
        @param maxComponents :  maximum number of components that %PCA should retain; by default, all the components are retained.
    */
    PCA( cv::InputArray data, cv::InputArray mean, int flags, int maxComponents = 0 );

    /** @overload constructor

        @param data     :       input samples stored as matrix rows or matrix columns.
        @param mean     :       optional mean value; if the matrix is empty (noArray()), the mean is computed from the data.
        @param flags    :       operation flags; currently the parameter is only used to specify the data layout (PCA::Flags)
        @param retainedVariance     :   Percentage of variance that PCA should retain. Using this parameter will let the PCA decided how many components to retain but it will always keep at least 2.
    */
    PCA( cv::InputArray data, cv::InputArray mean, int flags, double retainedVariance );

    
    /** @brief performs PCA

    The operator performs %PCA of the supplied dataset. It is safe to reuse
    the same PCA structure for multiple datasets. That is, if the structure
    has been previously used with another dataset, the existing internal
    data is reclaimed and the new @ref eigenvalues, @ref eigenvectors and @ref
    mean are allocated and computed.

    The computed eigenvalues are sorted from the largest to the smallest and
    the corresponding @ref eigenvectors are stored as eigenvectors rows.

        @param data     :       input samples stored as the matrix rows or as the matrix columns.
        @param mean     :       optional mean value; if the matrix is empty (noArray()), the mean is computed from the data.
        @param flags    :       operation flags; currently the parameter is only used to specify the data layout. (Flags)
        @param maxComponents    :   maximum number of components that PCA should retain; by default, all the components are retained.
     */
    PCA& operator()( cv::InputArray data, cv::InputArray mean, int flags, int maxComponents = 0 );

    /** @overload
        @param data     :       input samples stored as the matrix rows or as the matrix columns.
        @param mean     :       optional mean value; if the matrix is empty (noArray()), the mean is computed from the data.
        @param flags    :       operation flags; currently the parameter is only used to specify the data layout. (PCA::Flags)
        @param retainedVariance     :   Percentage of variance that %PCA should retain. Using this parameter will let the %PCA decided how many components to retain but it will always keep at least 2.
     */
    PCA& operator()( cv::InputArray data, cv::InputArray mean, int flags, double retainedVariance );


    /** Projects vector(s) to the principal component subspace.

    The methods project one or more vectors to the principal component
    subspace, where each vector projection is represented by coefficients in
    the principal component basis. The first form of the method returns the
    matrix that the second form writes to the result. So the first form can
    be used as a part of expression while the second form can be more
    efficient in a processing loop.

        @param vec  :   input vector(s); must have the same dimensionality and the same layout as the input data 
                        used at PCA phase, that is, if DATA_AS_ROW are specified, then `vec.cols==data.cols` (vector dimensionality) 
                        and `vec.rows` is the number of vectors to project, and the same is true for the PCA::DATA_AS_COL case.
    */
    cv::Mat project( cv::InputArray vec ) const;

    /** @overload

        @param vec      :   input vector(s); must have the same dimensionality and the same layout as the input data 
                            used at PCA phase, that is, if DATA_AS_ROW are specified, then `vec.cols==data.cols`
                            (vector dimensionality) and `vec.rows` is the number of vectors to project, and the same 
                            is true for the PCA::DATA_AS_COL case.
        
        @param result   :   output vectors; in case of PCA::DATA_AS_COL, the output matrix has as many columns 
                            as the number of input vectors, this means that `result.cols==vec.cols` and the 
                            number of rows match the number of principal components (for example, `maxComponents` 
                            parameter passed to the constructor).
     */
    void project( cv::InputArray vec, cv::OutputArray result ) const;


    /** Reconstructs vectors from their PC projections.

    The methods are inverse operations to PCA::project. They take PC
    coordinates of projected vectors and reconstruct the original vectors.
    Unless all the principal components have been retained, the
    reconstructed vectors are different from the originals. But typically,
    the difference is small if the number of components is large enough (but
    still much smaller than the original vector dimensionality). As a
    result, PCA is used.

        @param vec      :   coordinates of the vectors in the principal component subspace, the layout and 
                            size are the same as of PCA::project output vectors.
     */
    cv::Mat backProject( cv::InputArray vec ) const;

    /** @overload

        @param vec      :   coordinates of the vectors in the principal component subspace, the layout and size 
                            are the same as of PCA::project output vectors.
        @param result   :   reconstructed vectors; the layout and size are the same as of PCA::project input vectors.
     */
    void backProject( cv::InputArray vec, cv::OutputArray result ) const;

    
    /** write PCA objects

        Writes eigenvalues, eigenvectors and mean to specified FileStorage
     */
    void write( cv::FileStorage& fs ) const;


    /** load PCA objects

        Loads eigenvalues, eigenvectors and mean from specified FileNode
     */
    void read( const cv::FileNode& fn );

public:
    cv::Mat m_eigenvectors; //!< eigenvectors of the covariation matrix
    cv::Mat m_eigenvalues; //!< eigenvalues of the covariation matrix
    cv::Mat m_mean; //!< mean value subtracted before the projection and added after the back projection
};

}; /* namespace FACE */

#endif