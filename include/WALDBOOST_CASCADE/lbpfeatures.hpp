#ifndef WALDBOOST_LBPFEATURES_HPP
#define WALDBOOST_LBPFEATURES_HPP

#include "WALDBOOST_CASCADE/common_includes.hpp"
#include "WALDBOOST_CASCADE/featureEvaluator.hpp"

namespace WALDBOOST_CASCADE{

/** Calculate the index fo rect's four corners (p0, p1, p2, p3)

    @param rect
    @param step: step of row
  */
#define CV_SUM_OFFSETS( p0, p1, p2, p3, rect, step )                      \
    /* (x, y) */                                                          \
    (p0) = (rect).x + (step) * (rect).y;                                  \
    /* (x + w, y) */                                                      \
    (p1) = (rect).x + (rect).width + (step) * (rect).y;                   \
    /* (x, y + h) */                                                      \
    (p2) = (rect).x + (step) * ( (rect).y + (rect).height );                \
    /* (x + w, y + h) */                                                  \
    (p3) = (rect).x + (rect).width + (step) * ((rect).y + (rect).height);

/**************************************** LBPFeatureParams **************************************/
class LBPFeatureParams : public FeatureParams{
public:
    LBPFeatureParams();

    virtual cv::Ptr<FeatureParams> create();
}

/*************************************** LBPFeatureEvaluator ***************************************/
class LBPFeatureEvaluator : public FeatureEvaluator{
public:
    virtual ~LBPFeatureEvaluator() {}

    virtual void init( const FeatureParams* _featureParams,
                       int _maxSampleCount, cv::Size _winSize );

    virtual void setImage( const cv::Mat& img, uchar clsLabel, int idx, const std::vector<int>& feature_ind );

    virtual void setWindow(const cv::Point& p){ 
        m_curSum = m_sum.rowRange( p.y, p.y + m_winSize.height ).colRange( p.x, p.x + m_winSize.width ); 
    }

    virtual float operator()(int featureIdx){ 
        return (float)m_features[featureIdx].calc( m_curSum ); 
    }

    virtual void writeFeatures( cv::FileStorage &fs, const cv::Mat& featureMap ) const;

protected:
    virtual void generateFeatures();

    class Feature{
    public:
        Feature();
        Feature( int offset, int x, int y, int _block_w, int _block_h  );
        uchar calc( const cv::Mat& _sum );
        void write( cv::FileStorage &fs ) const;

        cv::Rect m_rect;

        // the number of neighbors is 8, and the radius is controled by m_blockW and m_blockH
        int m_p[16];
        int m_x, m_y, m_blockW, m_blockH; 

        int m_offset; // image's step of row

        void calcPoints( int offset );
    };

    std::vector<Feature> m_features;

    cv::Mat m_sum, m_curSum; // integel image
    int m_offset;
};

inline uchar CvLBPEvaluator::Feature::calc(const cv::Mat &_sum){

    const int* psum = _sum.ptr<int>(); // _sum is integel image

    int cval = psum[ p[5] ] - psum[ m_p[6] ] - psum[ m_p[9] ] + psum[ m_p[10] ]; // pixels' sum [p5 : p9, p5 : p6], calculate by integel image

    return (uchar)( ( psum[ m_p[0] ] - psum[ m_p[1] ] - psum[ m_p[4] ] + psum[ m_p[5] ] >= cval ? 128 : 0 ) |   // 0
                    ( psum[ m_p[1] ] - psum[ m_p[2] ] - psum[ m_p[5] ] + psum[ m_p[6] ] >= cval ? 64 : 0 ) |    // 1
                    ( psum[ m_p[2] ] - psum[ m_p[3] ] - psum[ m_p[6] ] + psum[ m_p[7] ] >= cval ? 32 : 0 ) |    // 2
                    ( psum[ m_p[6] ] - psum[ m_p[7] ] - psum[ m_p[10] ] + psum[ m_p[11] ] >= cval ? 16 : 0 ) |  // 5
                    ( psum[ m_p[10] ] - psum[ m_p[11] ] - psum[ m_p[14] ] + psum[ m_p[15] ] >= cval ? 8 : 0 ) | // 8
                    ( psum[ m_p[9] ] - psum[ m_p[10] ] - psum[ m_p[13] ] + psum[ m_p[14] ] >= cval ? 4 : 0 ) |  // 7
                    ( psum[ m_p[8] ] - psum[ m_p[9] ] - psum[ m_p[12] ] + psum[ m_p[13] ] >= cval ? 2 : 0 ) |   // 6
                    ( psum[ m_p[4] ] - psum[ m_p[5] ] - psum[ m_p[8] ] + psum[ m_p[9] ] >= cval ? 1 : 0 ) );     // 3
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
}


}; /* namespace WALDBOOST_CASCADE */

#endif