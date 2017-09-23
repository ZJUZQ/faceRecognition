#include "CascadeClassifier/BaseCascadeClassifier.hpp"

namespace CascadeClassifier{

/***************************************** CascadeClassifierImpl *****************************************/

CascadeClassifierImpl::CascadeClassifierImpl() {}

CascadeClassifierImpl::~CascadeClassifierImpl() {}

bool CascadeClassifierImpl::empty() const
{
    return !oldCascade && data.stages.empty(); // cv::Ptr<CvHaarClassifierCascade> oldCascade;
}

bool CascadeClassifierImpl::load(const String& filename)
{
    oldCascade.release();
    data = Data();
    featureEvaluator.release();

    FileStorage fs(filename, FileStorage::READ);
    if( !fs.isOpened() )
        return false;

    if( read_(fs.getFirstTopLevelNode()) )
        return true;

    fs.release();

    oldCascade.reset((CvHaarClassifierCascade*)cvLoad(filename.c_str(), 0, 0, 0));
    return !oldCascade.empty();
}

void CascadeClassifierImpl::read(const FileNode& node)
{
    read_(node);
}

int CascadeClassifierImpl::runAt( Ptr<FeatureEvaluator>& evaluator, Point pt, int scaleIdx, double& weight )
{
    CV_INSTRUMENT_REGION()

    assert( !oldCascade &&
           (data.featureType == FeatureEvaluator::HAAR ||
            data.featureType == FeatureEvaluator::LBP ||
            data.featureType == FeatureEvaluator::HOG) );

    if( !evaluator->setWindow(pt, scaleIdx) )
        return -1;
    if( data.maxNodesPerTree == 1 )
    {
        if( data.featureType == FeatureEvaluator::HAAR )
            return predictOrderedStump<HaarEvaluator>( *this, evaluator, weight );
        else if( data.featureType == FeatureEvaluator::LBP )
            return predictCategoricalStump<LBPEvaluator>( *this, evaluator, weight );
        else
            return -2;
    }
    else
    {
        if( data.featureType == FeatureEvaluator::HAAR )
            return predictOrdered<HaarEvaluator>( *this, evaluator, weight );
        else if( data.featureType == FeatureEvaluator::LBP )
            return predictCategorical<LBPEvaluator>( *this, evaluator, weight );
        else
            return -2;
    }
}

void CascadeClassifierImpl::setMaskGenerator(const Ptr<MaskGenerator>& _maskGenerator)
{
    maskGenerator=_maskGenerator;
}
Ptr<CascadeClassifierImpl::MaskGenerator> CascadeClassifierImpl::getMaskGenerator()
{
    return maskGenerator;
}

Ptr<BaseCascadeClassifier::MaskGenerator> createFaceDetectionMaskGenerator()
{
#ifdef HAVE_TEGRA_OPTIMIZATION
    if (tegra::useTegra())
        return tegra::getCascadeClassifierMaskGenerator();
#endif
    return Ptr<BaseCascadeClassifier::MaskGenerator>();
}

class CascadeClassifierInvoker : public ParallelLoopBody
{
public:
    CascadeClassifierInvoker( CascadeClassifierImpl& _cc, int _nscales, int _nstripes,
                              const FeatureEvaluator::ScaleData* _scaleData,
                              const int* _stripeSizes, std::vector<Rect>& _vec,
                              std::vector<int>& _levels, std::vector<double>& _weights,
                              bool outputLevels, const Mat& _mask, Mutex* _mtx)
    {
        classifier = &_cc;
        nscales = _nscales;
        nstripes = _nstripes;
        scaleData = _scaleData;
        stripeSizes = _stripeSizes;
        rectangles = &_vec;
        rejectLevels = outputLevels ? &_levels : 0;
        levelWeights = outputLevels ? &_weights : 0;
        mask = _mask;
        mtx = _mtx;
    }

    void operator()(const Range& range) const
    {
        CV_INSTRUMENT_REGION()

        Ptr<FeatureEvaluator> evaluator = classifier->featureEvaluator->clone();
        double gypWeight = 0.;
        Size origWinSize = classifier->data.origWinSize;

        for( int scaleIdx = 0; scaleIdx < nscales; scaleIdx++ )
        {
            const FeatureEvaluator::ScaleData& s = scaleData[scaleIdx];
            float scalingFactor = s.scale;
            int yStep = s.ystep;
            int stripeSize = stripeSizes[scaleIdx];
            int y0 = range.start*stripeSize;
            Size szw = s.getWorkingSize(origWinSize);
            int y1 = std::min(range.end*stripeSize, szw.height);
            Size winSize(cvRound(origWinSize.width * scalingFactor),
                         cvRound(origWinSize.height * scalingFactor));

            for( int y = y0; y < y1; y += yStep )
            {
                for( int x = 0; x < szw.width; x += yStep )
                {
                    int result = classifier->runAt(evaluator, Point(x, y), scaleIdx, gypWeight);
                    if( rejectLevels )
                    {
                        if( result == 1 )
                            result = -(int)classifier->data.stages.size();
                        if( classifier->data.stages.size() + result == 0 )
                        {
                            mtx->lock();
                            rectangles->push_back(Rect(cvRound(x*scalingFactor),
                                                       cvRound(y*scalingFactor),
                                                       winSize.width, winSize.height));
                            rejectLevels->push_back(-result);
                            levelWeights->push_back(gypWeight);
                            mtx->unlock();
                        }
                    }
                    else if( result > 0 )
                    {
                        mtx->lock();
                        rectangles->push_back(Rect(cvRound(x*scalingFactor),
                                                   cvRound(y*scalingFactor),
                                                   winSize.width, winSize.height));
                        mtx->unlock();
                    }
                    if( result == 0 )
                        x += yStep;
                }
            }
        }
    }

    CascadeClassifierImpl* classifier;
    std::vector<Rect>* rectangles;
    int nscales, nstripes;
    const FeatureEvaluator::ScaleData* scaleData;
    const int* stripeSizes;
    std::vector<int> *rejectLevels;
    std::vector<double> *levelWeights;
    std::vector<float> scales;
    Mat mask;
    Mutex* mtx;
};


struct getRect { Rect operator ()(const CvAvgComp& e) const { return e.rect; } };
struct getNeighbors { int operator ()(const CvAvgComp& e) const { return e.neighbors; } };

#ifdef HAVE_OPENCL
bool CascadeClassifierImpl::ocl_detectMultiScaleNoGrouping( const std::vector<float>& scales,
                                                            std::vector<Rect>& candidates )
{
    int featureType = getFeatureType();
    std::vector<UMat> bufs;
    featureEvaluator->getUMats(bufs);
    Size localsz = featureEvaluator->getLocalSize();
    if( localsz.area() == 0 )
        return false;
    Size lbufSize = featureEvaluator->getLocalBufSize();
    size_t localsize[] = { (size_t)localsz.width, (size_t)localsz.height };
    const int grp_per_CU = 12;
    size_t globalsize[] = { grp_per_CU*ocl::Device::getDefault().maxComputeUnits()*localsize[0], localsize[1] };
    bool ok = false;

    ufacepos.create(1, MAX_FACES*3+1, CV_32S);
    UMat ufacepos_count(ufacepos, Rect(0, 0, 1, 1));
    ufacepos_count.setTo(Scalar::all(0));

    if( ustages.empty() )
    {
        copyVectorToUMat(data.stages, ustages);
        if (!data.stumps.empty())
            copyVectorToUMat(data.stumps, unodes);
        else
            copyVectorToUMat(data.nodes, unodes);
        copyVectorToUMat(data.leaves, uleaves);
        if( !data.subsets.empty() )
            copyVectorToUMat(data.subsets, usubsets);
    }

    int nstages = (int)data.stages.size();
    int splitstage_ocl = 1;

    if( featureType == FeatureEvaluator::HAAR )
    {
        Ptr<HaarEvaluator> haar = featureEvaluator.dynamicCast<HaarEvaluator>();
        if( haar.empty() )
            return false;

        if( haarKernel.empty() )
        {
            String opts;
            if (lbufSize.area())
                opts = format("-D LOCAL_SIZE_X=%d -D LOCAL_SIZE_Y=%d -D SUM_BUF_SIZE=%d -D SUM_BUF_STEP=%d -D NODE_COUNT=%d -D SPLIT_STAGE=%d -D N_STAGES=%d -D MAX_FACES=%d -D HAAR",
                              localsz.width, localsz.height, lbufSize.area(), lbufSize.width, data.maxNodesPerTree, splitstage_ocl, nstages, MAX_FACES);
            else
                opts = format("-D LOCAL_SIZE_X=%d -D LOCAL_SIZE_Y=%d -D NODE_COUNT=%d -D SPLIT_STAGE=%d -D N_STAGES=%d -D MAX_FACES=%d -D HAAR",
                              localsz.width, localsz.height, data.maxNodesPerTree, splitstage_ocl, nstages, MAX_FACES);
            haarKernel.create("runHaarClassifier", ocl::objdetect::cascadedetect_oclsrc, opts);
            if( haarKernel.empty() )
                return false;
        }

        Rect normrect = haar->getNormRect();
        int sqofs = haar->getSquaresOffset();

        haarKernel.args((int)scales.size(),
                        ocl::KernelArg::PtrReadOnly(bufs[0]), // scaleData
                        ocl::KernelArg::ReadOnlyNoSize(bufs[1]), // sum
                        ocl::KernelArg::PtrReadOnly(bufs[2]), // optfeatures

                        // cascade classifier
                        ocl::KernelArg::PtrReadOnly(ustages),
                        ocl::KernelArg::PtrReadOnly(unodes),
                        ocl::KernelArg::PtrReadOnly(uleaves),

                        ocl::KernelArg::PtrWriteOnly(ufacepos), // positions
                        normrect, sqofs, data.origWinSize);
        ok = haarKernel.run(2, globalsize, localsize, true);
    }
    else if( featureType == FeatureEvaluator::LBP )
    {
        if (data.maxNodesPerTree > 1)
            return false;

        Ptr<LBPEvaluator> lbp = featureEvaluator.dynamicCast<LBPEvaluator>();
        if( lbp.empty() )
            return false;

        if( lbpKernel.empty() )
        {
            String opts;
            if (lbufSize.area())
                opts = format("-D LOCAL_SIZE_X=%d -D LOCAL_SIZE_Y=%d -D SUM_BUF_SIZE=%d -D SUM_BUF_STEP=%d -D SPLIT_STAGE=%d -D N_STAGES=%d -D MAX_FACES=%d -D LBP",
                              localsz.width, localsz.height, lbufSize.area(), lbufSize.width, splitstage_ocl, nstages, MAX_FACES);
            else
                opts = format("-D LOCAL_SIZE_X=%d -D LOCAL_SIZE_Y=%d -D SPLIT_STAGE=%d -D N_STAGES=%d -D MAX_FACES=%d -D LBP",
                              localsz.width, localsz.height, splitstage_ocl, nstages, MAX_FACES);
            lbpKernel.create("runLBPClassifierStumpSimple", ocl::objdetect::cascadedetect_oclsrc, opts);
            if( lbpKernel.empty() )
                return false;
        }

        int subsetSize = (data.ncategories + 31)/32;
        lbpKernel.args((int)scales.size(),
                       ocl::KernelArg::PtrReadOnly(bufs[0]), // scaleData
                       ocl::KernelArg::ReadOnlyNoSize(bufs[1]), // sum
                       ocl::KernelArg::PtrReadOnly(bufs[2]), // optfeatures

                       // cascade classifier
                       ocl::KernelArg::PtrReadOnly(ustages),
                       ocl::KernelArg::PtrReadOnly(unodes),
                       ocl::KernelArg::PtrReadOnly(usubsets),
                       subsetSize,

                       ocl::KernelArg::PtrWriteOnly(ufacepos), // positions
                       data.origWinSize);

        ok = lbpKernel.run(2, globalsize, localsize, true);
    }

    if( ok )
    {
        Mat facepos = ufacepos.getMat(ACCESS_READ);
        const int* fptr = facepos.ptr<int>();
        int nfaces = fptr[0];
        nfaces = std::min(nfaces, (int)MAX_FACES);

        for( int i = 0; i < nfaces; i++ )
        {
            const FeatureEvaluator::ScaleData& s = featureEvaluator->getScaleData(fptr[i*3 + 1]);
            candidates.push_back(Rect(cvRound(fptr[i*3 + 2]*s.scale),
                                      cvRound(fptr[i*3 + 3]*s.scale),
                                      cvRound(data.origWinSize.width*s.scale),
                                      cvRound(data.origWinSize.height*s.scale)));
        }
    }
    return ok;
}
#endif

bool CascadeClassifierImpl::isOldFormatCascade() const
{
    return !oldCascade.empty();
}

int CascadeClassifierImpl::getFeatureType() const
{
    return featureEvaluator->getFeatureType();
}

Size CascadeClassifierImpl::getOriginalWindowSize() const
{
    return data.origWinSize;
}

void* CascadeClassifierImpl::getOldCascade()
{
    return oldCascade;
}

static void detectMultiScaleOldFormat( const Mat& image, Ptr<CvHaarClassifierCascade> oldCascade,
                                       std::vector<Rect>& objects,
                                       std::vector<int>& rejectLevels,
                                       std::vector<double>& levelWeights,
                                       std::vector<CvAvgComp>& vecAvgComp,
                                       double scaleFactor, int minNeighbors,
                                       int flags, Size minObjectSize, Size maxObjectSize,
                                       bool outputRejectLevels = false )
{
    MemStorage storage(cvCreateMemStorage(0));
    CvMat _image = image;
    CvSeq* _objects = cvHaarDetectObjectsForROC( &_image, oldCascade, storage, rejectLevels, levelWeights, scaleFactor,
                                                 minNeighbors, flags, minObjectSize, maxObjectSize, outputRejectLevels );
    Seq<CvAvgComp>(_objects).copyTo(vecAvgComp);
    objects.resize(vecAvgComp.size());
    std::transform(vecAvgComp.begin(), vecAvgComp.end(), objects.begin(), getRect());
}

void CascadeClassifierImpl::detectMultiScaleNoGrouping( InputArray _image, std::vector<Rect>& candidates,
                                                    std::vector<int>& rejectLevels, std::vector<double>& levelWeights,
                                                    double scaleFactor, Size minObjectSize, Size maxObjectSize,
                                                    bool outputRejectLevels )
{
    CV_INSTRUMENT_REGION()

    Size imgsz = _image.size();
    Size originalWindowSize = getOriginalWindowSize();

    if( maxObjectSize.height == 0 || maxObjectSize.width == 0 )
        maxObjectSize = imgsz;

    // If a too small image patch is entering the function, break early before any processing
    if( (imgsz.height < originalWindowSize.height) || (imgsz.width < originalWindowSize.width) )
        return;

    std::vector<float> all_scales, scales;
    all_scales.reserve(1024);
    scales.reserve(1024);

    // First calculate all possible scales for the given image and model, then remove undesired scales
    // This allows us to cope with single scale detections (minSize == maxSize) that do not fall on precalculated scale
    for( double factor = 1; ; factor *= scaleFactor )
    {
        Size windowSize( cvRound(originalWindowSize.width*factor), cvRound(originalWindowSize.height*factor) );
        if( windowSize.width > imgsz.width || windowSize.height > imgsz.height )
            break;
        all_scales.push_back((float)factor);
    }

    // This will capture allowed scales and a minSize==maxSize scale, if it is in the precalculated scales
    for( size_t index = 0; index < all_scales.size(); index++){
        Size windowSize( cvRound(originalWindowSize.width*all_scales[index]), cvRound(originalWindowSize.height*all_scales[index]) );
        if( windowSize.width > maxObjectSize.width || windowSize.height > maxObjectSize.height)
            break;
        if( windowSize.width < minObjectSize.width || windowSize.height < minObjectSize.height )
            continue;
        scales.push_back(all_scales[index]);
    }

    // If minSize and maxSize parameter are equal and scales is not filled yet, then the scale was not available in the precalculated scales
    // In that case we want to return the most fitting scale (closest corresponding scale using L2 distance)
    if( scales.empty() && !all_scales.empty() ){
        std::vector<double> distances;
        // Calculate distances
        for(size_t v = 0; v < all_scales.size(); v++){
            Size windowSize( cvRound(originalWindowSize.width*all_scales[v]), cvRound(originalWindowSize.height*all_scales[v]) );
            double d = (minObjectSize.width - windowSize.width) * (minObjectSize.width - windowSize.width)
                       + (minObjectSize.height - windowSize.height) * (minObjectSize.height - windowSize.height);
            distances.push_back(d);
        }
        // Take the index of lowest value
        // Use that index to push the correct scale parameter
        size_t iMin=0;
        for(size_t i = 0; i < distances.size(); ++i){
            if(distances[iMin] > distances[i])
                    iMin=i;
        }
        scales.push_back(all_scales[iMin]);
    }

    candidates.clear();
    rejectLevels.clear();
    levelWeights.clear();

#ifdef HAVE_OPENCL
    bool use_ocl = tryOpenCL && ocl::useOpenCL() &&
         OCL_FORCE_CHECK(_image.isUMat()) &&
         featureEvaluator->getLocalSize().area() > 0 &&
         (data.minNodesPerTree == data.maxNodesPerTree) &&
         !isOldFormatCascade() &&
         maskGenerator.empty() &&
         !outputRejectLevels;
#endif

    Mat grayImage;
    _InputArray gray;

    if (_image.channels() > 1)
        cvtColor(_image, grayImage, COLOR_BGR2GRAY);
    else if (_image.isMat())
        grayImage = _image.getMat();
    else
        _image.copyTo(grayImage);
    gray = grayImage;

    if( !featureEvaluator->setImage(gray, scales) )
        return;

#ifdef HAVE_OPENCL
    // OpenCL code
    CV_OCL_RUN(use_ocl, ocl_detectMultiScaleNoGrouping( scales, candidates ))

    if (use_ocl)
        tryOpenCL = false;
#endif

    // CPU code
    featureEvaluator->getMats();
    {
        Mat currentMask;
        if (maskGenerator)
            currentMask = maskGenerator->generateMask(gray.getMat());

        size_t i, nscales = scales.size();
        cv::AutoBuffer<int> stripeSizeBuf(nscales);
        int* stripeSizes = stripeSizeBuf;
        const FeatureEvaluator::ScaleData* s = &featureEvaluator->getScaleData(0);
        Size szw = s->getWorkingSize(data.origWinSize);
        int nstripes = cvCeil(szw.width/32.);
        for( i = 0; i < nscales; i++ )
        {
            szw = s[i].getWorkingSize(data.origWinSize);
            stripeSizes[i] = std::max((szw.height/s[i].ystep + nstripes-1)/nstripes, 1)*s[i].ystep;
        }

        CascadeClassifierInvoker invoker(*this, (int)nscales, nstripes, s, stripeSizes,
                                         candidates, rejectLevels, levelWeights,
                                         outputRejectLevels, currentMask, &mtx);
        parallel_for_(Range(0, nstripes), invoker);
    }
}


void CascadeClassifierImpl::detectMultiScale( InputArray _image, std::vector<Rect>& objects,
                                          std::vector<int>& rejectLevels,
                                          std::vector<double>& levelWeights,
                                          double scaleFactor, int minNeighbors,
                                          int flags, Size minObjectSize, Size maxObjectSize,
                                          bool outputRejectLevels )
{
    CV_INSTRUMENT_REGION()

    CV_Assert( scaleFactor > 1 && _image.depth() == CV_8U );

    if( empty() )
        return;

    if( isOldFormatCascade() )
    {
        Mat image = _image.getMat();
        std::vector<CvAvgComp> fakeVecAvgComp;
        detectMultiScaleOldFormat( image, oldCascade, objects, rejectLevels, levelWeights, fakeVecAvgComp, scaleFactor,
                                   minNeighbors, flags, minObjectSize, maxObjectSize, outputRejectLevels );
    }
    else
    {
        detectMultiScaleNoGrouping( _image, objects, rejectLevels, levelWeights, scaleFactor, minObjectSize, maxObjectSize,
                                    outputRejectLevels );
        const double GROUP_EPS = 0.2;
        if( outputRejectLevels )
        {
            groupRectangles( objects, rejectLevels, levelWeights, minNeighbors, GROUP_EPS );
        }
        else
        {
            groupRectangles( objects, minNeighbors, GROUP_EPS );
        }
    }
}

void CascadeClassifierImpl::detectMultiScale( InputArray _image, std::vector<Rect>& objects,
                                          double scaleFactor, int minNeighbors,
                                          int flags, Size minObjectSize, Size maxObjectSize)
{
    CV_INSTRUMENT_REGION()

    std::vector<int> fakeLevels;
    std::vector<double> fakeWeights;
    detectMultiScale( _image, objects, fakeLevels, fakeWeights, scaleFactor,
        minNeighbors, flags, minObjectSize, maxObjectSize );
}

void CascadeClassifierImpl::detectMultiScale( InputArray _image, std::vector<Rect>& objects,
                                          std::vector<int>& numDetections, double scaleFactor,
                                          int minNeighbors, int flags, Size minObjectSize,
                                          Size maxObjectSize )
{
    CV_INSTRUMENT_REGION()

    Mat image = _image.getMat();
    CV_Assert( scaleFactor > 1 && image.depth() == CV_8U );

    if( empty() )
        return;

    std::vector<int> fakeLevels;
    std::vector<double> fakeWeights;
    if( isOldFormatCascade() )
    {
        std::vector<CvAvgComp> vecAvgComp;
        detectMultiScaleOldFormat( image, oldCascade, objects, fakeLevels, fakeWeights, vecAvgComp, scaleFactor,
                                   minNeighbors, flags, minObjectSize, maxObjectSize );
        numDetections.resize(vecAvgComp.size());
        std::transform(vecAvgComp.begin(), vecAvgComp.end(), numDetections.begin(), getNeighbors());
    }
    else
    {
        detectMultiScaleNoGrouping( image, objects, fakeLevels, fakeWeights, scaleFactor, minObjectSize, maxObjectSize );
        const double GROUP_EPS = 0.2;
        groupRectangles( objects, numDetections, minNeighbors, GROUP_EPS );
    }
}


CascadeClassifierImpl::Data::Data()
{
    stageType = featureType = ncategories = maxNodesPerTree = 0;
}

bool CascadeClassifierImpl::Data::read(const FileNode &root)
{
    static const float THRESHOLD_EPS = 1e-5f;

    // load stage params
    String stageTypeStr = (String)root[CC_STAGE_TYPE];
    if( stageTypeStr == CC_BOOST )
        stageType = BOOST;
    else
        return false;

    String featureTypeStr = (String)root[CC_FEATURE_TYPE];
    if( featureTypeStr == CC_HAAR )
        featureType = FeatureEvaluator::HAAR;
    else if( featureTypeStr == CC_LBP )
        featureType = FeatureEvaluator::LBP;
    else if( featureTypeStr == CC_HOG )
    {
        featureType = FeatureEvaluator::HOG;
        CV_Error(Error::StsNotImplemented, "HOG cascade is not supported in 3.0");
    }
    else
        return false;

    origWinSize.width = (int)root[CC_WIDTH];
    origWinSize.height = (int)root[CC_HEIGHT];
    CV_Assert( origWinSize.height > 0 && origWinSize.width > 0 );

    // load feature params
    FileNode fn = root[CC_FEATURE_PARAMS];
    if( fn.empty() )
        return false;

    ncategories = fn[CC_MAX_CAT_COUNT];
    int subsetSize = (ncategories + 31)/32,
        nodeStep = 3 + ( ncategories>0 ? subsetSize : 1 );

    // load stages
    fn = root[CC_STAGES];
    if( fn.empty() )
        return false;

    stages.reserve(fn.size());
    classifiers.clear();
    nodes.clear();
    stumps.clear();

    FileNodeIterator it = fn.begin(), it_end = fn.end();
    minNodesPerTree = INT_MAX;
    maxNodesPerTree = 0;

    for( int si = 0; it != it_end; si++, ++it )
    {
        FileNode fns = *it;
        Stage stage;
        stage.threshold = (float)fns[CC_STAGE_THRESHOLD] - THRESHOLD_EPS;
        fns = fns[CC_WEAK_CLASSIFIERS];
        if(fns.empty())
            return false;
        stage.ntrees = (int)fns.size();
        stage.first = (int)classifiers.size();
        stages.push_back(stage);
        classifiers.reserve(stages[si].first + stages[si].ntrees);

        FileNodeIterator it1 = fns.begin(), it1_end = fns.end();
        for( ; it1 != it1_end; ++it1 ) // weak trees
        {
            FileNode fnw = *it1;
            FileNode internalNodes = fnw[CC_INTERNAL_NODES];
            FileNode leafValues = fnw[CC_LEAF_VALUES];
            if( internalNodes.empty() || leafValues.empty() )
                return false;

            DTree tree;
            tree.nodeCount = (int)internalNodes.size()/nodeStep;
            minNodesPerTree = std::min(minNodesPerTree, tree.nodeCount);
            maxNodesPerTree = std::max(maxNodesPerTree, tree.nodeCount);

            classifiers.push_back(tree);

            nodes.reserve(nodes.size() + tree.nodeCount);
            leaves.reserve(leaves.size() + leafValues.size());
            if( subsetSize > 0 )
                subsets.reserve(subsets.size() + tree.nodeCount*subsetSize);

            FileNodeIterator internalNodesIter = internalNodes.begin(), internalNodesEnd = internalNodes.end();

            for( ; internalNodesIter != internalNodesEnd; ) // nodes
            {
                DTreeNode node;
                node.left = (int)*internalNodesIter; ++internalNodesIter;
                node.right = (int)*internalNodesIter; ++internalNodesIter;
                node.featureIdx = (int)*internalNodesIter; ++internalNodesIter;
                if( subsetSize > 0 )
                {
                    for( int j = 0; j < subsetSize; j++, ++internalNodesIter )
                        subsets.push_back((int)*internalNodesIter);
                    node.threshold = 0.f;
                }
                else
                {
                    node.threshold = (float)*internalNodesIter; ++internalNodesIter;
                }
                nodes.push_back(node);
            }

            internalNodesIter = leafValues.begin(), internalNodesEnd = leafValues.end();

            for( ; internalNodesIter != internalNodesEnd; ++internalNodesIter ) // leaves
                leaves.push_back((float)*internalNodesIter);
        }
    }

    if( maxNodesPerTree == 1 )
    {
        int nodeOfs = 0, leafOfs = 0;
        size_t nstages = stages.size();
        for( size_t stageIdx = 0; stageIdx < nstages; stageIdx++ )
        {
            const Stage& stage = stages[stageIdx];

            int ntrees = stage.ntrees;
            for( int i = 0; i < ntrees; i++, nodeOfs++, leafOfs+= 2 )
            {
                const DTreeNode& node = nodes[nodeOfs];
                stumps.push_back(Stump(node.featureIdx, node.threshold,
                                       leaves[leafOfs], leaves[leafOfs+1]));
            }
        }
    }

    return true;
}


bool CascadeClassifierImpl::read_(const FileNode& root)
{
#ifdef HAVE_OPENCL
    tryOpenCL = true;
    haarKernel = ocl::Kernel();
    lbpKernel = ocl::Kernel();
#endif
    ustages.release();
    unodes.release();
    uleaves.release();
    if( !data.read(root) )
        return false;

    // load features
    featureEvaluator = FeatureEvaluator::create(data.featureType);
    FileNode fn = root[CC_FEATURES];
    if( fn.empty() )
        return false;

    return featureEvaluator->read(fn, data.origWinSize);
}

}; /* namespace CascadeClassifier */