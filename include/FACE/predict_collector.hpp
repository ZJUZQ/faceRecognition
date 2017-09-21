#ifndef FACE_PREDICT_COLLECTOR_HPP
#define FACE_PREDICT_COLLECTOR_HPP

#include "FACE/common_includes.hpp"

namespace FACE{

/** @brief Abstract base class for all strategies of prediction result handling
*/
class PredictCollector{
public:
    virtual ~PredictCollector() {}

    /** @brief Interface method called by face recognizer before results processing
    @param size total size of prediction evaluation that recognizer could perform
    */
    virtual void init(size_t size) {
        // (void)size; 
    }


    /** @brief Interface method called by face recognizer for each result
    @param label current prediction label
    @param dist current prediction distance (confidence)
    */
    virtual bool collect( int label, double dist ) = 0;

};


/** @brief Default predict collector

Trace minimal distance with treshhold checking (that is default behavior for most predict logic)
*/
class StandardCollector : public PredictCollector{
public:
    struct PredictResult{
        int label;
        double distance;
        PredictResult( int label_ = -1, double distance_ = DBL_MAX ) : label( label_ ), distance( distance_ ) {}
    };

protected:
    double threshold; // threshold for distance ?
    PredictResult minRes;
    std::vector<PredictResult> data;

public:
    StandardCollector( double threshold_ = DBL_MAX );

    /** @brief overloaded interface method */
    void init(size_t size);

    /** @brief overloaded interface method */
    bool collect(int label, double dist);

    /** @brief Returns label with minimal distance */
    int getMinLabel() const;

    /** @brief Returns minimal distance value */
    double getMinDist() const;

    /** @brief Return results as vector
    @param sorted If set, results will be sorted by distance
    Each values is a pair of label and distance.
    */
    std::vector< std::pair<int, double> > getResultsVector( bool sorted = false ) const;

    /** @brief Return results as map
    Labels are keys, values are minimal distances
    */
    std::map<int, double> getResultsMap() const;


    /*  静态成员函数与静态数据成员一样，都是在类的内部实现，属于类定义的一部分, 不存在与程序中其他全局名字冲突的可能性; 
        静态成员函数为类的全部服务，而不是为某一个类的具体对象服务;
        静态成员函数由于不是与任何的对象相联系，因此它不具有this指针，从这个意义上讲，它无法访问属于类对象的非静态数据成员，也无法访问非静态成员函数，它只能调用其余的静态成员函数和静态数据成员。
    */
    static cv::Ptr<StandardCollector> create( double threshold_ = DBL_MAX );

};


}; /* namespace FACE */

#endif