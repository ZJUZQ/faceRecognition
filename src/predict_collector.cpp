#include "FACE/predict_collector.hpp"

namespace FACE{

static std::pair<int, double> toPair( const StandardCollector::PredictResult& val ){
    return std::make_pair( val.label, val.distance );
}

static bool pairLess( const std::pair<int, double>& lhs, const std::pair<int, double>& rhs ){
    return lhs.second < rhs.second;
}

/************************** StandardCollector ****************************/

StandardCollector::StandardCollector( double threshold_ ) : threshold( threshold_ ) {
    init( 0 );
}

void StandardCollector::init( size_t size ){
    minRes = PredictResult();
    data.clear();
    data.reserve( size ); // Requests that the vector capacity be at least enough to contain size elements.
}

bool StandardCollector::collect( int label, double dist ){
    if( dist < threshold ){
        PredictResult res( label, dist );
        if( res.distance < minRes.distance )
            minRes = res;
        data.push_back( res );
    }
    return true;
}

int StandardCollector::getMinLabel() const{
    return minRes.label;
}

double StandardCollector::getMinDist() const{
    return minRes.distance;
}

std::vector< std::pair<int, double> > StandardCollector::getResultsVector( bool sorted ) const{
    std::vector< std::pair<int, double> > res( data.size() );
    std::transform( data.begin(), data.end(), res.begin(), &toPair ); // Applies an operation( toPair ) sequentially to the elements of one (1) or two (2) ranges and stores the result in the range that begins at result.
    if( sorted ){
        std::sort( res.begin(), res.end(), &pairLess );
    }
    return res;
}

std::map<int, double> StandardCollector::getResultsMap() const{
    std::map<int, double> res;
    for( std::vector<PredictResult>::const_iterator i = data.begin(); i != data.end(); i++ ){
        std::map<int, double>::iterator j = res.find( i->label );
        if( j == res.end() )
            res.insert( toPair( *i ) );
        else if( i->distance < j->second )
            j->second = i->distance;
    }
    return res;
}

cv::Ptr<StandardCollector> StandardCollector::create( double threshold_ ){
    return cv::makePtr<StandardCollector>( threshold_ );
}


}; /* namespace FACE */