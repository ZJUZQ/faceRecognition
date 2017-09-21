#ifndef FACE_FACERECOGNIZER_HPP
#define FACE_FACERECOGNIZER_HPP

#include "FACE/FaceRecognizer.hpp"
#include "FACE/predict_collector.hpp"

namespace FACE{

/**
    Abstract base class for all face recognition models. 

    Every FaceRecognizer supports the:

    * Training of a FaceRecognizer with FaceRecognizer::train on a given set of images (your face database!).
    * Prediction of a given sample image, that means a face. The image is given as a Mat.
    * Loading/Saving the model state from/to a given XML or YAML.
    * Setting/Getting labels info, that is stored as a string. String labels info is useful for keeping names of the recognized people.

 */
class FaceRecognizer : public cv::Algorithm {
public:

    /** Trains a FaceRecognizer with given data and associated labels.

        @param src : The training images, that means the faces you want to learn. The data has to be given as a vector\<Mat\>.
        @param labels : The labels corresponding to the images have to be given either as a vector\<int\> or a
     */
    virtual void train( cv::InputArrayOfArrays src, cv::InputArray labels ) = 0;


    /** Updates a FaceRecognizer with given data and associated labels.
     */
    virtual void update( cv::InputArrayOfArrays src, cv::InputArray labels );


    /** Predicts a label for a given input image. 
     */
    int predict( cv::InputArray src ) const;

    // Predicts a label and associated confidence (e.g. distance) for a given input image.
    void predict( cv::InputArray src, int& label, double& confidence ) const;

    // if implemented - send all result of prediction to collector that can be used for somehow custom result handling 
    virtual void predict( cv::InputArray src, cv::Ptr<PredictCollector> collector ) const = 0;


    /** Saves a FaceRecognizer and its model state.
        Saves this model to a given filename, either as XML or YAML. 
     */
    virtual void save(const cv::String& filename) const;

    // This is an overloaded member function, provided for convenience. It differs from the above function only in what argument(s) it accepts. Saves this model to a given FileStorage. 
    virtual void save( cv::FileStorage& fs ) const;


    /** Loads a FaceRecognizer and its model state.
        Loads a persisted model and state from a given XML or YAML file . 
     */
    virtual void load(const cv::String& filename);

    virtual void load( const cv::FileStorage& fs );

    
    /** Sets string info for the specified model's label.
        The string info is replaced by the provided value if it was set before for the specified label.
     */
    virtual void setLabelInfo( int label, const cv::String& strInfo );


    /** Gets string information by label.
        If an unknown label id is provided or there is no label information associated with the specified
        label id the method returns an empty string.
     */
    virtual cv::String getLabelInfo( int label ) const;


    /** Gets vector of labels by string.
        The function searches for the labels containing the specified sub-string in the associated string
        info.
     */
    virtual std::vector<int> getLabelsByString( const cv::String& str ) const;


    /** threshold parameter accessor - required for default BestMinDist collector */
    virtual double getThreshold() const = 0;


    /** @brief Sets threshold of model */
    virtual void setThreshold( double val ) = 0;

protected:
    // Stored pairs "label id - string info"
    std::map<int, cv::String> _labelsInfo;

};


inline cv::Mat asRowMatrix( cv::InputArrayOfArrays src, int rtype, double alpha = 1, double beta = 0 ) {

    // make sure the input data is a vector of matrices or vector of vector
    if( src.kind() != cv::_InputArray::STD_VECTOR_MAT && src.kind() != cv::_InputArray::STD_VECTOR_VECTOR ) {
        cv::String error_message = "The data is expected as InputArray::STD_VECTOR_MAT (a std::vector<Mat>) or _InputArray::STD_VECTOR_VECTOR (a std::vector< std::vector<...> >).";
        CV_Error( cv::Error::StsBadArg, error_message );
    }

    // number of samples
    size_t n = src.total(); // 训练集中所有图片的数目 

    // return empty matrix if no matrices given
    if( n == 0 )
        return cv::Mat();

    /* size_t cv::Mat::total()  const
            Returns the total number of array elements.
            The method returns the number of array elements (a number of pixels if the array represents an image).
    */

    // dimensionality of (reshaped) samples
    size_t d = src.getMat(0).total(); // 第一张训练图片的像素数目

    // create data matrix
    cv::Mat data( (int)n, (int)d, rtype ); // 将训练集里的图片像素展开成一个矢量，得到一个矩阵：每一行表示一张训练图片的所有像素

    // now copy data
    for(unsigned int i = 0; i < n; i++) {
        // make sure data can be reshaped, throw exception if not!
        if(src.getMat(i).total() != d) {
            cv::String error_message = cv::format("Wrong number of elements in matrix #%d! Expected %d was %d.", i, d, src.getMat(i).total());
            CV_Error( cv::Error::StsBadArg, error_message );
        }
        // get a hold of the current row
        cv::Mat xi = data.row(i);
      
        /*  Mat cv::Mat::reshape ( int channels, int  rows = 0 ) const;
        */
        if( src.getMat( i ).isContinuous() ) {
            src.getMat( i ).reshape( 1, 1 ).convertTo( xi, rtype, alpha, beta );
        } 
        else {
            src.getMat( i ).clone().reshape( 1, 1 ).convertTo( xi, rtype, alpha, beta ); // make reshape happy by cloning for non-continuous matrices
        }
    }

    return data;
}

}; /* namespace FACE */

#endif