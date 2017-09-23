#ifndef WALDBOOST_FEATUREEVALUATOR_HPP
#define WALDBOOST_FEATUREEVALUATOR_HPP

#include "WALDBOOST_CASCADE/common_includes.hpp"

namespace WALDBOOST_CASCADE{

float calcNormFactor( const cv::Mat& sum, const cv::Mat& sqSum );

template<class Feature>
void _writeFeatures( const std::vector<Feature> features, cv::FileStorage& fs, const cv::Mat& featureMap ){

    fs << "features" << "[";
    const cv::Mat_<int>& featureMap_ = ( const cv::Mat_<int>& )featureMap;

    for( int i = 0; i < featureMap.cols; i++ ){
        if( featureMap_( 0, i ) >= 0 ){
            fs << "{";
            features[i].write( fs );
            fs << "}";
        }
    }
    fs << "]";
}

/*********************************** Params ***************************************/
class Params{
public:
    Params();
    virtual ~Params() {}
    // from|to file
    virtual void write( cv::FileStorage &fs ) const = 0;
    virtual bool read( const cv::FileNode &node ) = 0;
    // from|to screen
    virtual void printDefaults() const;
    virtual void printAttrs() const;
    virtual bool scanAttr( const std::string prmName, const std::string val );
    std::string m_name;
};

/************************************ FeatureParams *****************************************/
class FeatureParams : public Params{
public:
    enum { 
        HAAR = 0, 
        LBP = 1, 
        HOG = 2 
    };

    FeatureParams();
    virtual void init( const FeatureParams& fp );
    virtual void write( cv::FileStorage& fs ) const;
    virtual bool read( const cv::FileNode& node );
    virtual cv::Ptr<FeatureParams> create() = 0;

    int m_maxCatCount;
    int m_featSize; 
};

/************************************* FeatureEvaluator ****************************************/
class FeatureEvaluator{
public:
    virtual ~FeatureEvaluator() {}
    virtual void init( const FeatureParams* _featureParams,
                      int _maxSampleCount, cv::Size _winSize );

    virtual void setImage( const cv::Mat& img, uchar clsLabel, int idx, const std::vector<int> &feature_ind );

    virtual void setWindow( const cv::Point& p ) = 0;

    virtual void writeFeatures( cv::FileStorage& fs, const cv::Mat& featureMap ) const = 0;

    virtual float operator()( int featureIdx ) = 0;

    //static cv::Ptr<FeatureEvaluator> create();
    virtual cv::Ptr<FeatureEvaluator> create() = 0;

    int getNumFeatures() const { return m_numFeatures; }
    int getMaxCatCount() const { return featureParams->m_maxCatCount; }
    int getFeatureSize() const { return featureParams->m_featSize; }
    const cv::Mat& getCls() const { return m_cls; }
    float getCls( int si ) const { return m_cls.at<float>( si, 0 ); }

protected:
    virtual void generateFeatures() = 0;

    int m_npos;
    int m_nneg;
    int m_numFeatures;
    cv::Size m_winSize;
    FeatureParams* m_featureParams;
    cv::Mat m_cls;
};


}; /* namespce WALDBOOST_CASCADE */

#endif