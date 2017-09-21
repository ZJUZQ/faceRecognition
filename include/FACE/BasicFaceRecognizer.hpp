#ifndef FACE_BASICFACERECOGNIZER_HPP
#define FACE_BASICFACERECOGNIZER_HPP

#include "FACE/common_includes.hpp"
#include "FACE/FaceRecognizer.hpp"

namespace FACE{

// base for two classes: EigenFace and FisherFace
class BasicFaceRecognizer : public FaceRecognizer {
public:

    virtual int getNumComponents() const = 0;
    virtual void setNumComponents( int val ) = 0;

    // The threshold applied in the prediction.
    virtual double getThreshold() const = 0;
    virtual void setThreshold( double val ) = 0;

    virtual std::vector<cv::Mat> getProjections() const = 0;

    virtual cv::Mat getLabels() const = 0;

    virtual cv::Mat getEigenValues() const = 0;

    virtual cv::Mat getEigenVectors() const = 0;

    virtual cv::Mat getMean() const = 0; 
};


/**
    @param num_components : The number of components (read: Eigenfaces) kept for this Principal
                            Component Analysis. As a hint: There's no rule how many components (read: Eigenfaces) should be
                            kept for good reconstruction capabilities. It is based on your input data, so experiment with the
                            number. Keeping 80 components should almost always be sufficient.

    @param threshold : The threshold applied in the prediction.

### Notes:

-   Training and prediction must be done on grayscale images, use cvtColor to convert between the
    color spaces.
-   **THE EIGENFACES METHOD MAKES THE ASSUMPTION, THAT THE TRAINING AND TEST IMAGES ARE OF EQUAL
    SIZE.** (caps-lock, because I got so many mails asking for this). You have to make sure your
    input data has the correct shape, else a meaningful exception is thrown. Use resize to resize
    the images.
-   This model does not support updating.

### Model internal data:

-   num_components      :   see createEigenFaceRecognizer.
-   threshold           :   see createEigenFaceRecognizer.
-   eigenvalues         :   The eigenvalues for this Principal Component Analysis (ordered descending).
-   eigenvectors        :   The eigenvectors for this Principal Component Analysis (ordered by their eigenvalue).
-   mean                :   The sample mean calculated from the training data.
-   projections         :   The projections of the training data.
-   labels              :   The threshold applied in the prediction. If the distance to the nearest neighbor is larger than the threshold, this method returns -1.
*/
cv::Ptr<BasicFaceRecognizer> createEigenFaceRecognizer(int num_components = 0, double threshold = DBL_MAX);




/**
    @param num_components : The number of components (read: Fisherfaces) kept for this Linear
                            Discriminant Analysis with the Fisherfaces criterion. It's useful to keep all components, that
                            means the number of your classes c (read: subjects, persons you want to recognize). If you leave
                            this at the default (0) or set it to a value less-equal 0 or greater (c-1), it will be set to the
                            correct number (c-1) automatically.

    @param threshold :  The threshold applied in the prediction. If the distance to the nearest neighbor
                        is larger than the threshold, this method returns -1.

### Notes:

-   Training and prediction must be done on grayscale images, use cvtColor to convert between the
    color spaces.

-   **THE FISHERFACES METHOD MAKES THE ASSUMPTION, THAT THE TRAINING AND TEST IMAGES ARE OF EQUAL
    SIZE.** (caps-lock, because I got so many mails asking for this). You have to make sure your
    input data has the correct shape, else a meaningful exception is thrown. Use resize to resize
    the images.

-   This model does not support updating.

### Model internal data:

-   num_components  :   see createFisherFaceRecognizer.
-   threshold       :   see createFisherFaceRecognizer.
-   eigenvalues     :   The eigenvalues for this Linear Discriminant Analysis (ordered descending).
-   eigenvectors    :   The eigenvectors for this Linear Discriminant Analysis (ordered by their eigenvalue).
-   mean            :   The sample mean calculated from the training data.
-   projections     :   The projections of the training data.
-   labels          :   The labels corresponding to the projections.
 */
cv::Ptr<BasicFaceRecognizer> createFisherFaceRecognizer(int num_components = 0, double threshold = DBL_MAX);


/*********************************** BasicFaceRecognizerImpl **********************************/

class BasicFaceRecognizerImpl : public BasicFaceRecognizer{

public:
    BasicFaceRecognizerImpl( int num_components = 0, double threshold = DBL_MAX )
        : _num_components( num_components ), _threshold( threshold ) {}

    void load( const cv::FileStorage& fs ){
        // TODO
    }

    void save( cv::FileStorage& fs ) const {
        // TODO
    }

    int getNumComponents() const { return _num_components; }

    void setNumComponents( int val ) { _num_components = val; }

    // The threshold applied in the prediction.
    double getThreshold() const { return _threshold; }

    void setThreshold( double val ) { _threshold = val; }

    std::vector<cv::Mat> getProjections() const { return _projections; }

    cv::Mat getLabels() const { return _labels; }

    cv::Mat getEigenValues() const { return _eigenvalues; }

    cv::Mat getEigenVectors() const { return _eigenvectors; }

    cv::Mat getMean() const { return _mean; }

protected:
    int _num_components;
    double _threshold;
    std::vector<cv::Mat> _projections;
    cv::Mat _labels;
    cv::Mat _eigenvectors;
    cv::Mat _eigenvalues;
    cv::Mat _mean;
};

/************************************** Eigenfaces *********************************/

// Turk, M., and Pentland, A. "Eigenfaces for recognition.". Journal of
// Cognitive Neuroscience 3 (1991), 71–86.
class Eigenfaces : public BasicFaceRecognizerImpl{

public:
    // Initializes an empty Eigenfaces model.
    Eigenfaces( int num_components = 0, double threshold = DBL_MAX )
        : BasicFaceRecognizerImpl( num_components, threshold )
    {}

    // Computes an Eigenfaces model with images in src and corresponding labels in labels.
    void train( cv::InputArrayOfArrays src, cv::InputArray labels );

    // Send all predict results to caller side for custom result handling
    void predict( cv::InputArray src, cv::Ptr<PredictCollector> collector ) const;
};

/******************************* Fisherfaces ************************************/

// Belhumeur, P. N., Hespanha, J., and Kriegman, D. "Eigenfaces vs. Fisher-
// faces: Recognition using class specific linear projection.". IEEE
// Transactions on Pattern Analysis and Machine Intelligence 19, 7 (1997),
// 711–720.
class Fisherfaces: public BasicFaceRecognizerImpl
{
public:
    // Initializes an empty Fisherfaces model.
    Fisherfaces( int num_components = 0, double threshold = DBL_MAX )
        : BasicFaceRecognizerImpl( num_components, threshold )
    { }

    // Computes a Fisherfaces model with images in src and corresponding labels in labels.
    void train( cv::InputArrayOfArrays src, cv::InputArray labels );

    // Send all predict results to caller side for custom result handling
    void predict( cv::InputArray src, cv::Ptr<PredictCollector> collector ) const;
};


}; /* namespace FACE */


#endif