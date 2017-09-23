#include "CascadeClassifier/CascadeClassifier.hpp"

namespace CascadeClassifier{

CascadeClassifier::CascadeClassifier() {}

CascadeClassifier::CascadeClassifier( const cv::String& filename ){
    load( filename );
}

CascadeClassifier::~CascadeClassifier() {}

bool CascadeClassifier::empty() const {
    return cc.empty() || cc->empty(); // Ptr<BaseCascadeClassifier> cc;
}

bool CascadeClassifier::load( const cv::String& filename )
{
    cc = cv::makePtr<CascadeClassifierImpl>();
    if( !cc->load( filename ) )
        cc.release();
    return !empty();
}

bool CascadeClassifier::read( const cv::FileNode& root )
{
    cv::Ptr<CascadeClassifierImpl> ccimpl = cv::makePtr<CascadeClassifierImpl>();
    bool ok = ccimpl->read_( root );
    if( ok )
        cc = ccimpl.staticCast<BaseCascadeClassifier>();
    else
        cc.release();
    return ok;
}

void clipObjects( cv::Size sz, std::vector<cv::Rect>& objects,
                  std::vector<int>* a, std::vector<double>* b )
{
    size_t i, j = 0, n = objects.size();
    Rect win0 = Rect(0, 0, sz.width, sz.height);

    if(a){
        CV_Assert(a->size() == n);
    }

    if(b){
        CV_Assert(b->size() == n);
    }

    for( i = 0; i < n; i++ ){
        Rect r = win0 & objects[i];

        if( r.area() > 0 ){
            objects[j] = r;

            if( i > j ){
                if(a) a->at(j) = a->at(i);
                if(b) b->at(j) = b->at(i);
            }
            j++;
        }
    }

    if( j < n ){
        objects.resize(j);
        if(a) a->resize(j);
        if(b) b->resize(j);
    }
}

void CascadeClassifier::detectMultiScale( cv::InputArray image,
                      std::vector<cv::Rect>& objects,
                      double scaleFactor,
                      int minNeighbors, int flags,
                      Size minSize,
                      Size maxSize )
{
    //CV_INSTRUMENT_REGION()

    CV_Assert(!empty());
    cc->detectMultiScale(image, objects, scaleFactor, minNeighbors, flags, minSize, maxSize);
    clipObjects(image.size(), objects, 0, 0);
}

void CascadeClassifier::detectMultiScale( cv::InputArray image,
                      std::vector<cv::Rect>& objects,
                      std::vector<int>& numDetections,
                      double scaleFactor,
                      int minNeighbors, int flags,
                      cv::Size minSize, cv::Size maxSize )
{
    //CV_INSTRUMENT_REGION()

    CV_Assert(!empty());
    cc->detectMultiScale(image, objects, numDetections,
                         scaleFactor, minNeighbors, flags, minSize, maxSize);
    clipObjects(image.size(), objects, &numDetections, 0);
}

void CascadeClassifier::detectMultiScale( cv::InputArray image,
                      std::vector<cv::Rect>& objects,
                      std::vector<int>& rejectLevels,
                      std::vector<double>& levelWeights,
                      double scaleFactor,
                      int minNeighbors, int flags,
                      cv::Size minSize, cv::Size maxSize,
                      bool outputRejectLevels )
{
    //CV_INSTRUMENT_REGION()

    CV_Assert(!empty());
    cc->detectMultiScale(image, objects, rejectLevels, levelWeights,
                         scaleFactor, minNeighbors, flags,
                         minSize, maxSize, outputRejectLevels);
    clipObjects(image.size(), objects, &rejectLevels, &levelWeights);
}

bool CascadeClassifier::isOldFormatCascade() const {
    CV_Assert( !empty() );
    return cc->isOldFormatCascade();
}

cv::Size CascadeClassifier::getOriginalWindowSize() const{
    CV_Assert( !empty() );
    return cc->getOriginalWindowSize();
}

int CascadeClassifier::getFeatureType() const{
    CV_Assert( !empty() );
    return cc->getFeatureType();
}

void* CascadeClassifier::getOldCascade(){
    CV_Assert( !empty() );
    return cc->getOldCascade();
}

void CascadeClassifier::setMaskGenerator(const cv::Ptr<BaseCascadeClassifier::MaskGenerator>& maskGenerator){
    CV_Assert( !empty() );
    cc->setMaskGenerator( maskGenerator );
}

cv::Ptr<BaseCascadeClassifier::MaskGenerator> CascadeClassifier::getMaskGenerator(){
    CV_Assert( !empty() );
    return cc->getMaskGenerator();
}


}; /* namespace CascadeClassifier */