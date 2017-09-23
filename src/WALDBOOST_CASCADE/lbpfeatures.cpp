#include "WALDBOOST_CASCADE/lbpfeatures.hpp"

namespace WALDBOOST_CASCADE{

/************************************ LBPFeatureParams ************************************/

LBPFeatureParams::LBPFeatureParams(){
    m_maxCatCount = 256;
    m_name = "lbpFeatureParams";
}

cv::Ptr<FeatureParams> LBPFeatureParams::create(){
    return cv::Ptr<FeatureParams>( new LBPFeatureParams );
}


/*************************************** LBPFeatureEvaluator **************************************/

void LBPFeatureEvaluator::init( const FeatureParams* _featureParams, int _maxSampleCount, cv::Size _winSize )
{
    CV_Assert( _maxSampleCount > 0 );
    m_sum.create( (int)_maxSampleCount, ( _winSize.width + 1 ) * ( _winSize.height + 1 ), CV_32SC1 );
    FeatureEvaluator::init( _featureParams, _maxSampleCount, _winSize );
}

void LBPFeatureEvaluator::setImage( const cv::Mat& img, uchar clsLabel, int idx,
                                    const std::vector<int>& feature_ind )
{
    CV_DbgAssert( !m_sum.empty() );
    FeatureEvaluator::setImage( img, clsLabel, idx, feature_ind );
    cv::integral( img, m_sum );
    m_curSum = m_sum;
    m_offset = int( sum.ptr<int>(1) - sum.ptr<int>() ); // step of row

    for( size_t i = 0; i < feature_ind.size(); ++i ) {
        m_features[ feature_ind[i] ].calcPoints( m_offset );
    }
}

void LBPFeatureEvaluator::writeFeatures( cv::FileStorage& fs, const cv::Mat& featureMap ) const{
    _writeFeatures( m_features, fs, featureMap );
}

void LBPFeatureEvaluator::generateFeatures(){

    int offset = m_winSize.width + 1;

    for( int x = 0; x < winSize.width; x++ )
        for( int y = 0; y < winSize.height; y++ ){

            // each lbpFeature's size is Size( 3*w, 3*h )
            for( int w = 1; w <= winSize.width / 3; w++ )
                for( int h = 1; h <= winSize.height / 3; h++ )
                    if ( ( x + 3 * w <= winSize.width ) && ( y + 3 * h <= winSize.height ) )
                        m_features.push_back( Feature( offset, x, y, w, h ) );
        }

    m_numFeatures = (int)m_features.size();
}

LBPFeatureEvaluator::Feature::Feature(){
    m_rect = cv::Rect( 0, 0, 0, 0 );
}

LBPFeatureEvaluator::Feature::Feature( int offset, int x, int y, int _blockWidth, int _blockHeight ){
    m_x = x;
    m_y = y;
    m_blockW = _blockWidth;
    m_blockH = _blockHeight;
    m_offset = offset;
    calcPoints( offset );
}

void LBPFeatureEvaluator::Feature::calcPoints( int offset ){
    /*
        p0  ##  p1  **  p2  ##  p3 
        #       #       #       #
        #       #       #       #
        p4  ##  p5  **  p6  ##  p7  
        *                       *
        *                       *
        p8  ##  p9  **  p10 ##  p11 
        #       #       #       #
        #       #       #       # 
        p12 ##  p13 **  p14 ##  p15   

        m_blockW == (p1 - p0)
        m_blockH == (p4 - p0)
    */
    cv::Rect tr = m_rect = cv::Rect( m_x, m_y, m_blockW, m_blockH);
    CV_SUM_OFFSETS( m_p[0], p[1], p[4], m_p[5], tr, offset ) // set p0, p1, p4, p5

    tr.x += 2 * m_rect.width;
    CV_SUM_OFFSETS( p[2], m_p[3], m_p[6], m_p[7], tr, offset )

    tr.y += 2 * m_rect.height;
    CV_SUM_OFFSETS( m_p[10], m_p[11], m_p[14], m_p[15], tr, offset )

    tr.x -= 2 * m_rect.width;
    CV_SUM_OFFSETS( m_p[8], m_p[9], m_p[12], m_p[13], tr, offset )

    m_offset = offset;
}

void LBPFeatureEvaluator::Feature::write( cv::FileStorage& fs ) const
{
    fs << "rect" << "[:" << m_rect.x << m_rect.y << m_rect.width << m_rect.height << "]";
}



}; /* namespace WALDBOOST_CASCADE */