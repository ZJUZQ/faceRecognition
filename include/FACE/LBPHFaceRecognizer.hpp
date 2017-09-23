#ifndef FACE_LBPHFACERECOGNIZER_HPP
#define FACE_LBPHFACERECOGNIZER_HPP

#include "FACE/common_includes.hpp"
#include "FACE/FaceRecognizer.hpp"
#include "FACE/elbp.hpp"

namespace FACE{

/********************************** LBPHFaceRecognizer ************************************/

class LBPHFaceRecognizer : public FaceRecognizer{

public:
        /** @see setGridX */
    virtual int getGridX() const = 0;

    /** @copybrief getGridX @see getGridX */
    virtual void setGridX( int val ) = 0;

    /** @see setGridY */
    virtual int getGridY() const = 0;

    /** @copybrief getGridY @see getGridY */
    virtual void setGridY( int val ) = 0;

    /** @see setRadius */
    virtual int getRadius() const = 0;

    /** @copybrief getRadius @see getRadius */
    virtual void setRadius( int val ) = 0;

    /** @see setNeighbors */
    virtual int getNeighbors() const = 0;

    /** @copybrief getNeighbors @see getNeighbors */
    virtual void setNeighbors( int val ) = 0;

    /** @see setThreshold */
    virtual double getThreshold() const = 0;

    /** @copybrief getThreshold @see getThreshold */
    virtual void setThreshold( double val ) = 0;

    virtual std::vector<cv::Mat> getHistograms() const = 0;

    virtual cv::Mat getLabels() const = 0;

};

/**************************************** LBPH *****************************************/
// Face Recognition based on Local Binary Patterns.
//
//  Ahonen T, Hadid A. and Pietikäinen M. "Face description with local binary
//  patterns: Application to face recognition." IEEE Transactions on Pattern
//  Analysis and Machine Intelligence, 28(12):2037-2041.
class LBPH : public LBPHFaceRecognizer{

private:
    int _grid_x;
    int _grid_y;
    int _radius;
    int _neighbors;
    double _threshold;

    std::vector<cv::Mat> _histograms; 
    // Local Binary Patterns Histograms calculated from the given training data (empty if none was given).

    cv::Mat _labels; // Labels corresponding to the calculated Local Binary Patterns Histograms.

    void train( cv::InputArrayOfArrays src, cv::InputArray labels, bool preserveData );

public:
    //using FaceRecognizer::save;
    //using FaceRecognizer::load;

    // radius, neighbors are used in the local binary patterns creation.
    // grid_x, grid_y control the grid size of the spatial histograms.
    LBPH( int radius = 1, int neighbors = 8, int gridx = 8, int gridy = 8, double threshold = DBL_MAX )
        : _grid_x( gridx ), _grid_y( gridy ), _radius( radius ), _neighbors( neighbors ), _threshold( threshold ) {}

    LBPH( cv::InputArrayOfArrays src, cv::InputArray labels, int radius = 1, int neighbors = 8, int gridx = 8, int gridy = 8, double threshold = DBL_MAX ) 
        : _grid_x( gridx ), _grid_y( gridy ), _radius( radius ), _neighbors( neighbors ), _threshold( threshold ) 
    {
        train(src, labels);
    }

    ~LBPH() {}

    // Computes a LBPH model with images in src and
    // corresponding labels in labels.
    void train( cv::InputArrayOfArrays src, cv::InputArray labels );

    // Updates this LBPH model with images in src and
    // corresponding labels in labels.
    void update( cv::InputArrayOfArrays src, cv::InputArray labels );

    // Send all predict results to caller side for custom result handling
    void predict( cv::InputArray src, cv::Ptr<PredictCollector> collector ) const;

    // See FaceRecognizer::load.
    void load( const cv::FileStorage& fs );

    // See FaceRecognizer::save.
    void save( cv::FileStorage& fs) const;

            /** @see setGridX */
    int getGridX() const { return _grid_x; }

    /** @copybrief getGridX @see getGridX */
    void setGridX( int val ) { _grid_x = val; }

    /** @see setGridY */
    int getGridY() const { return _grid_y; }

    /** @copybrief getGridY @see getGridY */
    void setGridY( int val ) { _grid_y = val; }

    /** @see setRadius */
    int getRadius() const { return _radius; }

    /** @copybrief getRadius @see getRadius */
    void setRadius( int val ) { _radius = val; }

    /** @see setNeighbors */
    int getNeighbors() const { return _neighbors; }

    /** @copybrief getNeighbors @see getNeighbors */
    void setNeighbors( int val ) { _neighbors = val; }

    /** @see setThreshold */
    double getThreshold() const { return _threshold; }

    /** @copybrief getThreshold @see getThreshold */
    void setThreshold( double val ) { _threshold = val; }

    std::vector<cv::Mat> getHistograms() const { return _histograms; }

    cv::Mat getLabels() const { return _labels; }

};


/**
    @param radius       :       The radius used for building the Circular Local Binary Pattern. The greater the
                                radius, the

    @param neighbors    :       The number of sample points to build a Circular Local Binary Pattern from. An
                                appropriate value is to use `8` sample points. Keep in mind: the more sample points you include,
                                the higher the computational cost.

    @param grid_x       :       The number of cells in the horizontal direction, 8 is a common value used in
                                publications. The more cells, the finer the grid, the higher the dimensionality of the resulting
                                feature vector.

    @param grid_y       :       The number of cells in the vertical direction, 8 is a common value used in
                                publications. The more cells, the finer the grid, the higher the dimensionality of the resulting
                                feature vector.

    @param threshold    :       The threshold applied in the prediction. If the distance to the nearest neighbor
                                is larger than the threshold, this method returns -1.

### Notes:

-   The Circular Local Binary Patterns (used in training and prediction) expect the data given as
    grayscale images, use cvtColor to convert between the color spaces.
-   This model supports updating.
 */
cv::Ptr<LBPHFaceRecognizer> createLBPHFaceRecognizer( int radius = 1, int neighbors = 8, int grid_x = 8, int grid_y = 8, double threshold = DBL_MAX );


}; /* namespace FACE */

#endif