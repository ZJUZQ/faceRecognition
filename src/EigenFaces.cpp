#include "FACE/BasicFaceRecognizer.hpp"

namespace FACE{

/************************************ Eigenfaces **********************************/

// Computes an Eigenfaces model with images in src and corresponding labels in labels.
void Eigenfaces::train( cv::InputArrayOfArrays _src, cv::InputArray _local_labels) {

    if( _src.total() == 0 ) {
        cv::String error_message = cv::format( "Empty training data was given. You'll need more than one sample to learn a model." );
        CV_Error( cv::Error::StsBadArg, error_message );
    } else if( _local_labels.getMat().type() != CV_32SC1 ){
        cv::String error_message = cv::format( "Labels must be given as integer (CV_32SC1). Expected %d, but was %d.", CV_32SC1, _local_labels.type() );
        CV_Error( cv::Error::StsBadArg, error_message );
    }

    // make sure data has correct size, that is each image has the same size
    if( _src.total() > 1 ) {
        for( int i = 1; i < static_cast<int>( _src.total() ); i++) {
            if( _src.getMat( i - 1 ).total() != _src.getMat( i ).total() ) {
                cv::String error_message = cv::format("In the Eigenfaces method all input samples (training images) must be of equal size! Expected %d pixels, but was %d pixels.", _src.getMat(i-1).total(), _src.getMat(i).total());
                CV_Error( cv::Error::StsUnsupportedFormat, error_message );
            }
        }
    }

    // get labels
    cv::Mat labels = _local_labels.getMat();

    // observations in row
    cv::Mat data = asRowMatrix( _src, CV_64FC1 ); // 将训练集里的每张图片像素展开成一个矢量，得到一个矩阵data：每一行表示一张训练图片的所有像素

    // number of samples
    int n = data.rows;

    // assert there are as much samples as labels
    if( static_cast<int>( labels.total() ) != n ) {
        cv::String error_message = cv::format("The number of samples (src) must equal the number of labels (labels)! len(src)=%d, len(labels)=%d.", n, labels.total());
        CV_Error( cv::Error::StsBadArg, error_message );
    }

    // clear existing model data
    _labels.release();
    _projections.clear();

    // clip number of components to be valid
    if((_num_components <= 0) || (_num_components > n)) // _num_components: 选择的主成分数目
        _num_components = n;

    // perform the PCA:      cv::PCA (InputArray data, InputArray mean, int flags, int maxComponents=0)
    cv::PCA pca( data, cv::Mat(), cv::PCA::DATA_AS_ROW, _num_components ); // DATA_AS_ROW: indicates that the input samples are stored as matrix rows 

    // copy the PCA results
    _mean = pca.mean.reshape( 1, 1 ); // store the mean vector
    _eigenvalues = pca.eigenvalues.clone(); // eigenvalues by row
    cv::transpose( pca.eigenvectors, _eigenvectors ); // eigenvectors by column

    // store labels for prediction
    _labels = labels.clone();

    // save projections, 将所有训练样本投影到PCA子空间
    for(int sampleIdx = 0; sampleIdx < data.rows; sampleIdx++) {
        /* static Mat subspaceProject (InputArray W, InputArray mean, InputArray src) */
        cv::Mat p = cv::LDA::subspaceProject( _eigenvectors, _mean, data.row( sampleIdx ) ); // Linear Discriminant Analysis
        _projections.push_back(p); 
    }
}

 // Send all predict results to caller side for custom result handling
void Eigenfaces::predict( cv::InputArray _src, cv::Ptr<PredictCollector> collector ) const {
    //  collector   User-defined collector object that accepts all results

    cv::Mat src = _src.getMat(); // get data

    // make sure the user is passing correct data
    if( _projections.empty() ) {
        cv::String error_message = "This Eigenfaces model is not computed yet. Did you call Eigenfaces::train?";
        CV_Error( cv::Error::StsError, error_message );
    } 
    else if( _eigenvectors.rows != static_cast<int>( src.total() ) ) {
        // check data alignment just for clearer exception messages
        cv::String error_message = cv::format("Wrong input image size. Reason: Training and Test images must be of equal size! Expected an image with %d elements, but got %d.", _eigenvectors.rows, src.total());
        CV_Error( cv::Error::StsBadArg, error_message);
    }

    // project into PCA subspace
    cv::Mat q = cv::LDA::subspaceProject( _eigenvectors, _mean, src.reshape(1, 1) ); // 将测试样本投影到PCA子空间

    collector->init( _projections.size() );

    for ( size_t sampleIdx = 0; sampleIdx < _projections.size(); sampleIdx++ ) {
        double dist = cv::norm( _projections[sampleIdx], q, cv::NORM_L2 );
        int label = _labels.at<int>( (int)sampleIdx );
        if ( !collector->collect( label, dist ) ) // dist > threshold ??
            return;
    }
}

cv::Ptr<BasicFaceRecognizer> createEigenFaceRecognizer( int num_components, double threshold ){
    return cv::makePtr<Eigenfaces>( num_components, threshold ); // makePtr<T>(...) is equivalent to Ptr<T>(new T(...)). 
}

};