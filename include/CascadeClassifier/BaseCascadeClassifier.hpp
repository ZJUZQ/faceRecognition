#ifndef CASCADEC_BASECASCADECLASSIFIER_HPP
#define CASCADEC_BASECASCADECLASSIFIER_HPP

#include "CascadeClassifier/common_includes.hpp"

namespace CascadeClassifier{

/************************************ BaseCascadeClassifier *****************************************/
class BaseCascadeClassifier : public Algorithm{

public:
    virtual ~BaseCascadeClassifier();

    // Returns true if the Algorithm is empty (e.g. in the very beginning or after unsuccessful read. 
    virtual bool empty() const = 0;

    virtual bool load( const cv::String& filename ) = 0;

    virtual void detectMultiScale( cv::InputArray image,
                                   std::vector<cv::Rect>& objects,
                                   double scaleFactor,
                                   int minNeighbors, int flags,
                                   cv::Size minSize, cv::Size maxSize ) = 0;

    virtual void detectMultiScale( cv::InputArray image,
                                   std::vector<cv::Rect>& objects,
                                   std::vector<int>& numDetections,
                                   double scaleFactor,
                                   int minNeighbors, int flags,
                                   cv::Size minSize, cv::Size maxSize ) = 0;

    virtual void detectMultiScale( cv::InputArray image,
                                   std::vector<cv::Rect>& objects,
                                   std::vector<int>& rejectLevels,
                                   std::vector<double>& levelWeights,
                                   double scaleFactor,
                                   int minNeighbors, int flags,
                                   cv::Size minSize, cv::Size maxSize,
                                   bool outputRejectLevels ) = 0;

    virtual bool isOldFormatCascade() const = 0;

    virtual cv::Size getOriginalWindowSize() const = 0;

    virtual int getFeatureType() const = 0;

    virtual void* getOldCascade() = 0;

    class MaskGenerator
    {
    public:
        virtual ~MaskGenerator() {}

        virtual cv::Mat generateMask( const cv::Mat& src ) = 0;
        virtual void initializeMask( const cv::Mat& /*src*/) { }
    };

    virtual void setMaskGenerator( const cv::Ptr<MaskGenerator>& maskGenerator ) = 0;

    virtual cv::Ptr<MaskGenerator> getMaskGenerator() = 0;
};


/************************************* CascadeClassifierImpl *****************************************/
class CascadeClassifierImpl : public BaseCascadeClassifier
{
public:
    CascadeClassifierImpl();
    virtual ~CascadeClassifierImpl();

    bool empty() const;
    bool load( const String& filename );
    void read( const FileNode& node );
    bool read_( const FileNode& node );
    void detectMultiScale( InputArray image,
                          CV_OUT std::vector<Rect>& objects,
                          double scaleFactor = 1.1,
                          int minNeighbors = 3, int flags = 0,
                          Size minSize = Size(),
                          Size maxSize = Size() );

    void detectMultiScale( InputArray image,
                          CV_OUT std::vector<Rect>& objects,
                          CV_OUT std::vector<int>& numDetections,
                          double scaleFactor=1.1,
                          int minNeighbors=3, int flags=0,
                          Size minSize=Size(),
                          Size maxSize=Size() );

    void detectMultiScale( InputArray image,
                          CV_OUT std::vector<Rect>& objects,
                          CV_OUT std::vector<int>& rejectLevels,
                          CV_OUT std::vector<double>& levelWeights,
                          double scaleFactor = 1.1,
                          int minNeighbors = 3, int flags = 0,
                          Size minSize = Size(),
                          Size maxSize = Size(),
                          bool outputRejectLevels = false );


    bool isOldFormatCascade() const;
    Size getOriginalWindowSize() const;
    int getFeatureType() const;
    void* getOldCascade();

    void setMaskGenerator(const Ptr<MaskGenerator>& maskGenerator);
    Ptr<MaskGenerator> getMaskGenerator();

protected:
    enum { SUM_ALIGN = 64 };

    bool detectSingleScale( InputArray image, Size processingRectSize,
                            int yStep, double factor, std::vector<Rect>& candidates,
                            std::vector<int>& rejectLevels, std::vector<double>& levelWeights,
                            Size sumSize0, bool outputRejectLevels = false );
#ifdef HAVE_OPENCL
    bool ocl_detectMultiScaleNoGrouping( const std::vector<float>& scales,
                                         std::vector<Rect>& candidates );
#endif
    void detectMultiScaleNoGrouping( InputArray image, std::vector<Rect>& candidates,
                                    std::vector<int>& rejectLevels, std::vector<double>& levelWeights,
                                    double scaleFactor, Size minObjectSize, Size maxObjectSize,
                                    bool outputRejectLevels = false );

    enum { MAX_FACES = 10000 };
    enum { BOOST = 0 };
    enum { DO_CANNY_PRUNING    = CASCADE_DO_CANNY_PRUNING,
        SCALE_IMAGE         = CASCADE_SCALE_IMAGE,
        FIND_BIGGEST_OBJECT = CASCADE_FIND_BIGGEST_OBJECT,
        DO_ROUGH_SEARCH     = CASCADE_DO_ROUGH_SEARCH
    };

    friend class CascadeClassifierInvoker;
    friend class SparseCascadeClassifierInvoker;

    template<class FEval>
    friend int predictOrdered( CascadeClassifierImpl& cascade, Ptr<FeatureEvaluator> &featureEvaluator, double& weight);

    template<class FEval>
    friend int predictCategorical( CascadeClassifierImpl& cascade, Ptr<FeatureEvaluator> &featureEvaluator, double& weight);

    template<class FEval>
    friend int predictOrderedStump( CascadeClassifierImpl& cascade, Ptr<FeatureEvaluator> &featureEvaluator, double& weight);

    template<class FEval>
    friend int predictCategoricalStump( CascadeClassifierImpl& cascade, Ptr<FeatureEvaluator> &featureEvaluator, double& weight);

    int runAt( Ptr<FeatureEvaluator>& feval, Point pt, int scaleIdx, double& weight );

    class Data
    {
    public:
        struct DTreeNode
        {
            int featureIdx;
            float threshold; // for ordered features only
            int left;
            int right;
        };

        struct DTree
        {
            int nodeCount;
        };

        struct Stage
        {
            int first;
            int ntrees;
            float threshold;
        };

        struct Stump
        {
            Stump() { }
            Stump(int _featureIdx, float _threshold, float _left, float _right)
            : featureIdx(_featureIdx), threshold(_threshold), left(_left), right(_right) {}

            int featureIdx;
            float threshold;
            float left;
            float right;
        };

        Data();

        bool read(const FileNode &node);

        int stageType;
        int featureType;
        int ncategories;
        int minNodesPerTree, maxNodesPerTree;
        Size origWinSize;

        std::vector<Stage> stages;
        std::vector<DTree> classifiers;
        std::vector<DTreeNode> nodes;
        std::vector<float> leaves;
        std::vector<int> subsets;
        std::vector<Stump> stumps;
    };

    Data data;
    Ptr<FeatureEvaluator> featureEvaluator;
    Ptr<CvHaarClassifierCascade> oldCascade;

    Ptr<MaskGenerator> maskGenerator;
    UMat ugrayImage;
    UMat ufacepos, ustages, unodes, uleaves, usubsets;
#ifdef HAVE_OPENCL
    ocl::Kernel haarKernel, lbpKernel;
    bool tryOpenCL;
#endif

    Mutex mtx;
};


}; /* namespace CascadeClassifier */

#endif