#ifndef WALDBOOST_WALDBOOST_HPP
#define WALDBOOST_WALDBOOST_HPP

#include "WALDBOOST_CASCADE/common_includes.hpp"

namespace WALDBOOST_CASCADE{

class WaldBoost{

public:
    WaldBoost( int weak_count );
    WaldBoost();

    std::vector<int> get_feature_indices();

    void detect( cv::Ptr<CvFeatureEvaluator> eval,
                 const cv::Mat& img,
                 const std::vector<float>& scales,
                 std::vector<cv::Rect>& bboxes,
                 cv::Mat1f& confidences );

    void detect( cv::Ptr<CvFeatureEvaluator> eval,
                 const cv::Mat& img,
                 const std::vector<float>& scales,
                 std::vector<cv::Rect>& bboxes,
                 std::vector<double>& confidences );

    void fit( cv::Mat& data_pos, cv::Mat& data_neg );
    int predict( cv::Ptr<CvFeatureEvaluator> eval, float* h ) const;
    void save( const std::string& filename );
    void load( const std::string& filename );

    void read( const cv::FileNode& fn );
    void write( cv::FileStorage& fs ) const;

    void reset( int weak_count );
    ~WaldBoost();

private:
    int weak_count_;
    std::vector<float> thresholds_;
    std::vector<float> alphas_;
    std::vector<int> feature_indices_;
    std::vector<int> polarities_;
    std::vector<float> cascade_thresholds_;    
};


}; /* namespace WALDBOOST_CASCADE */

#endif