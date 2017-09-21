#include "FACE/BasicFaceRecognizer.hpp"

namespace FACE{

template<typename _Tp>
inline std::vector<_Tp> remove_dups( const std::vector<_Tp>& src ){ // remove duplicate elements

    //  typedef创建了存在类型的别名 constSetIterator;
    //  模板类型在实例化之前，编译器并不知道std::set<_Tp>::const_iterator是什么东西,而typename告诉编译器std::set<_Tp>::const_iterator是一个类型而不是一个成员。
    typedef typename std::set<_Tp>::const_iterator constSetIterator;
    typedef typename std::vector<_Tp>::const_iterator constVecIterator;

    std::set<_Tp> setElems;  // Sets are containers that store unique elements following a specific order.
    // set插入重复元素无效，因而这里用来统计不重复元素数目

    for( constVecIterator it = src.begin(); it != src.end(); it++ )
        setElems.insert( *it );

    std::vector<_Tp> elems;
    for( constSetIterator it = setElems.begin(); it != setElems.end(); it++ )
        elems.push_back( *it );

    return elems;
}

/****************************************** Fisherfaces *************************************/

// Belhumeur, P. N., Hespanha, J., and Kriegman, D. "Eigenfaces vs. Fisher-
// faces: Recognition using class specific linear projection.". IEEE
// Transactions on Pattern Analysis and Machine Intelligence 19, 7 (1997),
// 711–720.

// Computes a Fisherfaces model with images in src and corresponding labels in labels.
void Fisherfaces::train( cv::InputArrayOfArrays src, cv::InputArray lbs ){
    if( src.total() == 0 ){
        cv::String error_message = cv::format( "Empty training data was given. You'll need more than one sample to learn a model." );
        CV_Error( cv::Error::StsBadArg, error_message );
    }
    else if( lbs.getMat().type() != CV_32SC1 ){
        cv::String error_message = cv::format("Labels must be given as integer (CV_32SC1). Expected %d, but was %d.", CV_32SC1, lbs.type());
        CV_Error( cv::Error::StsBadArg, error_message );
    }

    // make sure data has correct size
    if( src.total() > 1 ){
        for( int i = 1; i < static_cast<int>( src.total() ); i++ ){
            if( src.getMat( i - 1 ).total() != src.getMat( i ).total() ){
                cv::String error_message = cv::format( "In the Fisherfaces method all input samples (training images) must be of equal size! Expected %d pixels, but was %d pixels.", src.getMat(i-1).total(), src.getMat(i).total() );
                CV_Error( cv::Error::StsUnsupportedFormat, error_message );
            }
        }
    }

    // get data
    cv::Mat labels = lbs.getMat();
    cv::Mat data = asRowMatrix( src, CV_64FC1 ); // 将训练集里的图片像素展开成一个矢量，得到一个矩阵data：每一行表示一张训练图片的所有像素

    int N = data.rows; // number of samples

    // make sure labels are passed in correct shape
    if( labels.total() != ( size_t )N ){
        cv::String error_message = cv::format("The number of samples (src) must equal the number of labels (labels)! len(src)=%d, len(labels)=%d.", N, labels.total());
        CV_Error( cv::Error::StsBadArg, error_message );
    }
    else if( labels.rows != 1 && labels.cols != 1 ){
        cv::String error_message = cv::format("Expected the labels in a matrix with one row or column! Given dimensions are rows=%s, cols=%d.", labels.rows, labels.cols);
        CV_Error( cv::Error::StsBadArg, error_message );
    }

    // clear existing model data
    _labels.release();
    _projections.clear();

    // safely copy from cv::Mat to std::vector
    std::vector<int> ll;
    for( unsigned int i = 0; i < labels.total(); i++ )
        ll.push_back( labels.at<int>( i ) );

    // get the number of unique classes
    int C = (int) remove_dups( ll ).size();

    // clip number of components to be a valid number
    if( _num_components <= 0 || _num_components > ( C - 1 ) )
        _num_components = C - 1;

    // perform a PCA and keep (N - C) components
    cv::PCA pca( data, cv::Mat(), cv::PCA::DATA_AS_ROW, ( N - C ) );

    // project the data and perform a LDA on it
    cv::LDA lda( pca.project( data ), labels, _num_components );

    _mean = pca.mean.reshape( 1, 1 ); // store the total mean vector in row
    _labels = labels.clone();
    lda.eigenvalues().convertTo( _eigenvalues, CV_64FC1 ); // store the eigenvalues fo the discriminants

    // Now calculate the projection matrix as :     
    //                   pca.eigenvectors * lda.eigenvectors
    // Note: OpenCV stores the eigenvectors by row, so we need to transpose it!
    cv::gemm( pca.eigenvectors, lda.eigenvectors(), 1.0, cv::Mat(), 0.0, _eigenvectors, cv::GEMM_1_T );

    // store the projections of the original data
    for( int sampleIdx = 0; sampleIdx < data.rows; sampleIdx++ ){
        cv::Mat p = cv::LDA::subspaceProject( _eigenvectors, _mean, data.row( sampleIdx ) );
        _projections.push_back( p );
    }
}

void Fisherfaces::predict( cv::InputArray _src, cv::Ptr<PredictCollector> collector ) const {
    cv::Mat src = _src.getMat();

    // check data alignment just for clearer exception messages
    if( _projections.empty() ) {
        // throw error if no data (or simply return -1?)
        cv::String error_message = "This Fisherfaces model is not computed yet. Did you call Fisherfaces::train?";
        CV_Error( cv::Error::StsBadArg, error_message );
    } 
    else if( src.total() != (size_t) _eigenvectors.rows ) {
        cv::String error_message = cv::format("Wrong input image size. Reason: Training and Test images must be of equal size! Expected an image with %d elements, but got %d.", _eigenvectors.rows, src.total());
        CV_Error( cv::Error::StsBadArg, error_message );
    }

    // project into LDA subspace
    cv::Mat q = cv::LDA::subspaceProject( _eigenvectors, _mean, src.reshape( 1, 1 ) );

    // find 1-nearest neighbor
    collector->init( (int)_projections.size() );
    for( size_t sampleIdx = 0; sampleIdx < _projections.size(); sampleIdx++ ) {
        double dist = cv::norm( _projections[ sampleIdx ], q, cv::NORM_L2 );
        int label = _labels.at<int>( (int)sampleIdx );
        if ( !collector->collect( label, dist ) )
            return;
    }
}

cv::Ptr<BasicFaceRecognizer> createFisherFaceRecognizer( int num_components, double threshold )
{
    return cv::makePtr<Fisherfaces>( num_components, threshold );
}


}; /* namespace FACE */