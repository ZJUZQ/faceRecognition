#include "WALDBOOST_CASCADE/featureEvaluator.hpp"

namespace WALDBOOST_CASCADE{

float calcNormFactor( const cv::Mat& sum, const cv::Mat& sqSum ){

    CV_DbgAssert( sum.cols > 3 && sqSum.rows > 3 );
    Rect normrect( 1, 1, sum.cols - 3, sum.rows - 3 );
    size_t p0, p1, p2, p3;
    CV_SUM_OFFSETS( p0, p1, p2, p3, normrect, sum.step1() )
    double area = normrect.width * normrect.height;
    const int *sp = sum.ptr<int>();
    int valSum = sp[p0] - sp[p1] - sp[p2] + sp[p3];
    const double *sqp = sqSum.ptr<double>();
    double valSqSum = sqp[p0] - sqp[p1] - sqp[p2] + sqp[p3];
    return (float) sqrt( (double) (area * valSqSum - (double)valSum * valSum) );
}

/**************************** Params ************************************/
Params::Params() : m_name( "params" ) {}

void Params::printDefaults() const{
    std::cout << "--" << name << "--" << std::endl;
}

void Params::printAttrs() const {}

bool Params::scanAttr( const std::string, const std::string ){
    return false;
}

/********************************* FeatureParams ********************************/
FeatureParams::FeatureParams() : m_maxCatCount( 0 ), m_featSize( 1 ){
    m_name = "featureParams";
}

void FeatureParams::init( const FeatureParams& fp ){
    m_maxCatCount = fp.m_maxCatCount;
    m_featSize = fp.m_featSize;
}

void FeatureParams::write( cv::FileStorage& fs ) const{
    fs << "maxCatCount" << m_maxCatCount;
    fs << "featSize" << m_featSize;
}

bool FeatureParams::read( const cv::FileNode& fn ){
    if( fn.empty() )
        return false;
    m_maxCatCount = fn["maxCatCount"];
    m_featSize = fn["featSize"];
    return ( m_maxCatCount >= 0 && m_featSize >= 1 );
}

/*
cv::Ptr<FeatureParams> FeatureParams::create(){
    return Ptr<CvFeatureParams>(new CvLBPFeatureParams);
}
*/

/********************************* FeatureEvaluator **************************************/
void FeatureEvaluator::init( const FeatureParams* featureParams, int maxSampleCount, cv::Size winSize ){
    CV_Assert( maxSampleCount > 0 );
    m_featureParams = ( FeatureParams* )featureParams;
    m_winSize = winSize;
    m_numFeatures = 0;
    m_cls.create( (int)maxSampleCount, 1, CV_32FC1 );
    generateFeatures();
}

void FeatureEvaluator::setImage( const cv::Mat&, uchar clsLabel, int idx, const std::vector<int>& ){
    //CV_Assert(img.cols == winSize.width);
    //CV_Assert(img.rows == winSize.height);
    CV_Assert( idx < cls.rows );
    cls.ptr<float>( idx )[ 0 ] = clsLabel;
}

/*
cv::Ptr<FeatureEvaluator> FeatureEvaluator::create(){
    return cv::Ptr<CvFeatureEvaluator>( new CvLBPEvaluator );
}
*/


}; /* namespace WALDBOOST_CASCADE */