#include "FACE/LBPHFaceRecognizer.hpp"

namespace FACE{

/************************************ LBPH ***********************************/

void LBPH::load( const cv::FileStorage& fs ) {
/*
    fs["radius"] >> _radius;
    fs["neighbors"] >> _neighbors;
    fs["grid_x"] >> _grid_x;
    fs["grid_y"] >> _grid_y;
    //read matrices
    readFileNodeList(fs["histograms"], _histograms);
    fs["labels"] >> _labels;
    const FileNode& fn = fs["labelsInfo"];
    if (fn.type() == FileNode::SEQ){
        _labelsInfo.clear();
        for (FileNodeIterator it = fn.begin(); it != fn.end();)
        {
            LabelInfo item;
            it >> item;
            _labelsInfo.insert(std::make_pair(item.label, item.value));
        }
    }
*/
}

// See FaceRecognizer::save.
void LBPH::save( cv::FileStorage& fs ) const {
/*
    fs << "radius" << _radius;
    fs << "neighbors" << _neighbors;
    fs << "grid_x" << _grid_x;
    fs << "grid_y" << _grid_y;
    // write matrices
    writeFileNodeList(fs, "histograms", _histograms);
    fs << "labels" << _labels;
    fs << "labelsInfo" << "[";
    for (std::map<int, String>::const_iterator it = _labelsInfo.begin(); it != _labelsInfo.end(); it++)
        fs << LabelInfo(it->first, it->second);
    fs << "]";
*/
}

void LBPH::train( cv::InputArrayOfArrays _in_src, cv::InputArray _in_labels ) {
    this->train( _in_src, _in_labels, false );
}

void LBPH::update( cv::InputArrayOfArrays _in_src, cv::InputArray _in_labels ) {
    // got no data, just return
    if( _in_src.total() == 0 )
        return;

    this->train( _in_src, _in_labels, true );
}

void LBPH::train( cv::InputArrayOfArrays _in_src, cv::InputArray _in_labels, bool preserveData) {

    if(_in_src.kind() != cv::_InputArray::STD_VECTOR_MAT && _in_src.kind() != cv::_InputArray::STD_VECTOR_VECTOR) {
        cv::String error_message = "The images are expected as InputArray::STD_VECTOR_MAT (a std::vector<Mat>) or _InputArray::STD_VECTOR_VECTOR (a std::vector< std::vector<...> >).";
        CV_Error( cv::Error::StsBadArg, error_message );
    }

    if( _in_src.total() == 0 ) {
        cv::String error_message = cv::format("Empty training data was given. You'll need more than one sample to learn a model.");
        CV_Error( cv::Error::StsUnsupportedFormat, error_message );
    } 
    else if( _in_labels.getMat().type() != CV_32SC1 ) {
        cv::String error_message = cv::format("Labels must be given as integer (CV_32SC1). Expected %d, but was %d.", CV_32SC1, _in_labels.type());
        CV_Error( cv::Error::StsUnsupportedFormat, error_message );
    }
    // get the vector of matrices
    std::vector<cv::Mat> src;
    _in_src.getMatVector( src );

    // get the label matrix
    cv::Mat labels = _in_labels.getMat();

    // check if data is well- aligned
    if( labels.total() != src.size() ) {
        cv::String error_message = cv::format("The number of samples (src) must equal the number of labels (labels). Was len(samples)=%d, len(labels)=%d.", src.size(), _labels.total());
        CV_Error( cv::Error::StsBadArg, error_message );
    }

    // if this model should be trained without preserving old data, delete old model data
    if( !preserveData ) {
        _labels.release();
        _histograms.clear();
    }

    // append labels to _labels matrix
    for( size_t labelIdx = 0; labelIdx < labels.total(); labelIdx++ ) {
        _labels.push_back( labels.at<int>( (int)labelIdx ) );
    }

    // store the spatial histograms of the original data
    // 对每张训练图像，将每个网格区域的直方图进行连接(不合并)，获得空间增强的特征向量
    for( size_t sampleIdx = 0; sampleIdx < src.size(); sampleIdx++ ) {
        // calculate lbp image
        cv::Mat lbp_image = elbp( src[sampleIdx], _radius, _neighbors ); // lbp_image的每个像素值是对应src中的一个像素的LBP值：SUM 2^p*s(ip-ic)

        // get spatial histogram from this lbp image
        cv::Mat p = spatial_histogram(
                    lbp_image, /* lbp_image */
                    static_cast<int>( std::pow( 2.0, static_cast<double>(_neighbors) ) ), /* number of possible patterns */
                    _grid_x, /* grid size x */
                    _grid_y, /* grid size y */
                    true);

        // add to templates
        _histograms.push_back( p );
    }
}


void LBPH::predict( cv::InputArray _src, cv::Ptr<PredictCollector> collector ) const {

    if( _histograms.empty() ) {
        // throw error if no data (or simply return -1?)
        cv::String error_message = "This LBPH model is not computed yet. Did you call the train method?";
        CV_Error( cv::Error::StsBadArg, error_message );
    }

    cv::Mat src = _src.getMat();

    // get the spatial histogram from input image
    cv::Mat lbp_image = elbp( src, _radius, _neighbors );

    cv::Mat query = spatial_histogram(
                        lbp_image, /* lbp_image */
                        static_cast<int>(std::pow(2.0, static_cast<double>(_neighbors))), /* number of possible patterns */
                        _grid_x, /* grid size x */
                        _grid_y, /* grid size y */
                        true /* normed histograms */);

    // find 1-nearest neighbor
    collector->init( (int)_histograms.size() );

    for ( size_t sampleIdx = 0; sampleIdx < _histograms.size(); sampleIdx++ ) {
        double dist = cv::compareHist( _histograms[sampleIdx], query, cv::HISTCMP_CHISQR_ALT );
        int label = _labels.at<int>((int)sampleIdx);
        if ( !collector->collect( label, dist ) )
            return;
    }
}


//************************************************************************************//

cv::Ptr<LBPHFaceRecognizer> createLBPHFaceRecognizer(int radius, int neighbors, int grid_x, int grid_y, double threshold )
{
    return cv::makePtr<LBPH>( radius, neighbors, grid_x, grid_y, threshold );
}


}; /* namespace FACE */