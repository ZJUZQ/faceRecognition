#ifndef CASCADECLASSIFIER_CASCADECLASSIFIER_HPP
#define CASCADECLASSIFIER_CASCADECLASSIFIER_HPP

#include "CascadeClassifier/common_includes.hpp"
#include "CascadeClassifier/BaseCascadeClassifier.hpp"

namespace CascadeClassifier{

/**************************************** CascadeClassifier ****************************************/

/** @brief Cascade classifier class for object detection.
 */
class CascadeClassifier{

public:
    CascadeClassifier();

    /** @brief Loads a classifier from a file.
            @param filename     :   Name of the file from which the classifier is loaded.
     */
    CascadeClassifier( const cv::String& filename );

    ~CascadeClassifier();

    /** @brief Checks whether the classifier has been loaded.
    */
    bool empty() const;

    /** @brief Loads a classifier from a file.
            @param filename     :   Name of the file from which the classifier is loaded. The file may contain an old
                                    HAAR classifier trained by the haartraining application or a new cascade classifier trained by the
                                    traincascade application.
     */
    bool load( const cv::String& filename );


    /** @brief Reads a classifier from a FileStorage node.
            @note The file may contain a new cascade classifier (trained traincascade application) only.
     */
    bool read( const cv::FileNode& node );


    /** @brief Detects objects of different sizes in the input image. The detected objects are returned as a list
    of rectangles.

        @param image    :   Matrix of the type CV_8U containing an image where objects are detected.
        @param objects  :   Vector of rectangles where each rectangle contains the detected object, the
                            rectangles may be partially outside the original image.
        @param scaleFactor  :   Parameter specifying how much the image size is reduced at each image scale.
        @param minNeighbors :   Parameter specifying how many neighbors each candidate rectangle should have
                                to retain it.
        @param flags    :   Parameter with the same meaning for an old cascade as in the function
                            cvHaarDetectObjects. It is not used for a new cascade.
        @param minSize  :   Minimum possible object size. Objects smaller than that are ignored.
        @param maxSize  :   Maximum possible object size. Objects larger than that are ignored. 
                            If `maxSize == minSize` model is evaluated on single scale.

    The function is parallelized with the TBB library.

    @note
       -   (Python) A face detection example using cascade classifiers can be found at
            opencv_source_code/samples/python/facedetect.py
    */
    void detectMultiScale( cv::InputArray image,
                           std::vector<cv::Rect>& objects,
                           double scaleFactor = 1.1,
                           int minNeighbors = 3, int flags = 0,
                           cv::Size minSize = cv::Size(),
                           cv::Size maxSize = cv::Size() );

    /** @overload

        @param image    :   Matrix of the type CV_8U containing an image where objects are detected.
        @param objects  :   Vector of rectangles where each rectangle contains the detected object, the
                            rectangles may be partially outside the original image.
        @param numDetections    :   Vector of detection numbers for the corresponding objects. An object's number
                                    of detections is the number of neighboring positively classified rectangles that were joined
                                    together to form the object.
        @param scaleFactor  :   Parameter specifying how much the image size is reduced at each image scale.
        @param minNeighbors :   Parameter specifying how many neighbors each candidate rectangle should have
                                to retain it.
        @param flags    :   Parameter with the same meaning for an old cascade as in the function
                            cvHaarDetectObjects. It is not used for a new cascade.
        @param minSize  :   Minimum possible object size. Objects smaller than that are ignored.
        @param maxSize  :   Maximum possible object size. Objects larger than that are ignored. If `maxSize == minSize` model is evaluated on single scale.
    */
    void detectMultiScale( cv::InputArray image,
                           std::vector<cv::Rect>& objects,
                           std::vector<int>& numDetections,
                           double scaleFactor=1.1,
                           int minNeighbors=3, int flags=0,
                           cv::Size minSize = cv::Size(),
                           cv::Size maxSize = cv::Size() );


    /** @overload
    This function allows you to retrieve the final stage decision certainty of classification.
    For this, one needs to set `outputRejectLevels` on true and provide the `rejectLevels` and `levelWeights` parameter.
    For each resulting detection, `levelWeights` will then contain the certainty of classification at the final stage.
    This value can then be used to separate strong from weaker classifications.

    If outputRejectLevels is true returns rejectLevels and levelWeights

    A code sample on how to use it efficiently can be found below:
        @code
        Mat img;
        vector<double> weights;
        vector<int> levels;
        vector<Rect> detections;
        CascadeClassifier model("/path/to/your/model.xml");
        model.detectMultiScale(img, detections, levels, weights, 1.1, 3, 0, Size(), Size(), true);
        cerr << "Detection " << detections[0] << " with weight " << weights[0] << endl;
        @endcode
    */
    void detectMultiScale( cv::InputArray image,
                           std::vector<cv::Rect>& objects,
                           std::vector<int>& rejectLevels,
                           std::vector<double>& levelWeights,
                           double scaleFactor = 1.1,
                           int minNeighbors = 3, int flags = 0,
                           cv::Size minSize = cv::Size(),
                           cv::Size maxSize = cv::Size(),
                           bool outputRejectLevels = false );

    bool isOldFormatCascade() const;

    cv::Size getOriginalWindowSize() const;

    int getFeatureType() const;

    void* getOldCascade();

    static bool convert( const cv::String& oldcascade, const cv::String& newcascade );

    void setMaskGenerator( const cv::Ptr<BaseCascadeClassifier::MaskGenerator>& maskGenerator );

    cv::Ptr<BaseCascadeClassifier::MaskGenerator> getMaskGenerator();

    cv::Ptr<BaseCascadeClassifier> cc;

};


};

#endif