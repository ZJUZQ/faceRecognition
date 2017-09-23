#ifndef WALDBOOST_WBDETECTOR_HPP
#define WALDBOOST_WBDETECTOR_HPP

#include "WALDBOOST_CASCADE/common_includes.hpp"
#include "WALDBOOST_CASCADE/waldboost.hpp"

namespace WALDBOOST_CASCADE{

/** Extended object detection
 */
class WBDetector{

public:
    /** @brief Read detector from FileNode.
    */
    virtual void read( const cv::FileNode& node ) = 0;

    /** @brief Write detector to FileStorage.
    */
    virtual void write( cv::FileStorage& fs ) const = 0;


    /** @brief Train WaldBoost detector
            @param pos_samples  :   Path to directory with cropped positive samples
            @param neg_imgs     :   Path to directory with negative (background) images
    */
    virtual void train(
        const std::string& pos_samples,
        const std::string& neg_imgs ) = 0;


    /** @brief Detect objects on image using WaldBoost detector
            @param img      :   Input image for detection
            @param bboxes   :   Bounding boxes coordinates output vector
            @param confidences  :   Confidence values for bounding boxes output vector
    */
    virtual void detect(
        const cv::Mat& img,
        std::vector<cv::Rect> &bboxes,
        std::vector<double> &confidences ) = 0;


    /** @brief Create instance of WBDetector
    */
    static cv::Ptr<WBDetector> create();

    virtual ~WBDetector(){}

};


class WBDetectorImpl : public WBDetector {

public:
    virtual void read( const cv::FileNode& node );
    virtual void write( cv::FileStorage& fs ) const;

    virtual void train( const std::string& pos_samples, const std::string& neg_imgs );

    virtual void detect( const cv::Mat& img, std::vector<cv::Rect>& bboxes, std::vector<double>& confidences );

protected:
    WaldBoost boost_;
};


}; /* namespcace WALDBOOST_CASCADE */

#endif