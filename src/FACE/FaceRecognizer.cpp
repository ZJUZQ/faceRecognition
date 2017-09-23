#include "FACE/FaceRecognizer.hpp"

namespace FACE{

/********************************* FaceRecognizer *****************************/

std::vector<int> FaceRecognizer::getLabelsByString(const cv::String& str) const{
    std::vector<int> labels;
    for( std::map<int, cv::String>::const_iterator it = _labelsInfo.begin(); it != _labelsInfo.end(); it++ ){
        size_t found = ( it->second ).find( str );
        if ( found != cv::String::npos )
            labels.push_back( it->first );
    }
    return labels;
}

cv::String FaceRecognizer::getLabelInfo( int label ) const {
    std::map<int, cv::String>::const_iterator iter( _labelsInfo.find( label ) );
    return iter != _labelsInfo.end() ? iter->second : "";
}

void FaceRecognizer::setLabelInfo( int label, const cv::String& strInfo ){
    _labelsInfo[ label ] = strInfo;
}

void FaceRecognizer::update( cv::InputArrayOfArrays src, cv::InputArray labels ){
    (void)src;
    (void)labels;
    cv::String error_msg = cv::format( "This FaceRecognizer does not support updating, you have to use FaceRecognizer::train to update it." );
    CV_Error( cv::Error::StsNotImplemented, error_msg );
}

void FaceRecognizer::load( const cv::String& filename ){
    cv::FileStorage fs( filename, cv::FileStorage::READ );
    if ( !fs.isOpened() )
        CV_Error( cv::Error::StsError, "File can't be opened for reading!" );
    this->load( fs );
    fs.release();
}

void FaceRecognizer::load( const cv::FileStorage& fs ){
    if ( !fs.isOpened() )
        CV_Error( cv::Error::StsError, "File can't be opened for reading!" );
    this->load( fs );
}

void FaceRecognizer::save(const cv::String &filename) const{
    cv::FileStorage fs( filename, cv::FileStorage::WRITE );
    if ( !fs.isOpened() )
        CV_Error( cv::Error::StsError, "File can't be opened for writing!" );
    this->save( fs );
    fs.release();
}

void FaceRecognizer::save( cv::FileStorage& fs ) const{
    if ( !fs.isOpened() )
        CV_Error( cv::Error::StsError, "File can't be opened for writing!" );
    this->save( fs );
}

int FaceRecognizer::predict( cv::InputArray src ) const {
    int _label;
    double _dist;
    predict( src, _label, _dist );
    return _label;
}

void FaceRecognizer::predict( cv::InputArray src, int& label, double& confidence) const {
    cv::Ptr<StandardCollector> collector = StandardCollector::create( getThreshold() );
    predict( src, collector );
    label = collector->getMinLabel();
    confidence = collector->getMinDist();
}


}; /* namespace FACE */
