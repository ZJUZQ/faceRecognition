#include "FACE/Covar_Eigen.hpp"

namespace FACE{

void calcCovarMatrix( const cv::Mat* data, int nsamples, cv::Mat& covar, cv::Mat& _mean, int flags, int ctype )
{
    //CV_INSTRUMENT_REGION()

    CV_Assert( data && nsamples > 0 );
    cv::Size size = data[0].size();

    int sz = size.width * size.height;
    int esz = (int)data[0].elemSize(); // The method returns the matrix element size in bytes. For example, if the matrix type is CV_16SC3 , the method returns 3*sizeof(short) or 6. 
    int type = data[0].type();
    
    cv::Mat mean;
    ctype = std::max( std::max( CV_MAT_DEPTH(ctype >= 0 ? ctype : type), _mean.depth() ), CV_32F );

    if( (flags & FACE_COVAR_USE_AVG) != 0 )
    {
        CV_Assert( _mean.size() == size );
        if( _mean.isContinuous() && _mean.type() == ctype ) // The method returns true if the matrix elements are stored continuously without gaps at the end of each row.
            mean = _mean.reshape(1, 1);
        else
        {
            _mean.convertTo(mean, ctype); // Converts an array to another data type with optional scaling. 
            mean = mean.reshape(1, 1);
        }
    }

    cv::Mat _data(nsamples, sz, type);

    for( int i = 0; i < nsamples; i++ )
    {
        CV_Assert( data[i].size() == size && data[i].type() == type );
        if( data[i].isContinuous() )
            std::memcpy( _data.ptr(i), data[i].ptr(), sz*esz );
        else
        {
            cv::Mat dataRow( size.height, size.width, type, _data.ptr(i) );
            data[i].copyTo( dataRow );
        }
    }

    calcCovarMatrix( _data, covar, mean, ( flags & ~( FACE_COVAR_ROWS | FACE_COVAR_COLS ) ) | FACE_COVAR_ROWS, ctype ); // _data as row

    if( (flags & FACE_COVAR_USE_AVG) == 0 )
        _mean = mean.reshape(1, size.height);
}


void calcCovarMatrix( cv::Mat _src, cv::Mat _covar, cv::Mat& _mean, int flags, int ctype )
{
    //CV_INSTRUMENT_REGION()

    cv::Mat data = _src.clone();
    cv::Mat mean;
    CV_Assert( ((flags & FACE_COVAR_ROWS) != 0) ^ ((flags & FACE_COVAR_COLS) != 0) );
    bool takeRows = (flags & FACE_COVAR_ROWS) != 0;

    int type = data.type();
    int nsamples = takeRows ? data.rows : data.cols;
    CV_Assert( nsamples > 0 );
    cv::Size size = takeRows ? cv::Size(data.cols, 1) : cv::Size(1, data.rows);

    if( (flags & FACE_COVAR_USE_AVG) != 0 ){ // given mean value

        mean = _mean.clone();
        ctype = std::max(std::max(CV_MAT_DEPTH(ctype >= 0 ? ctype : type), mean.depth()), CV_32F);
        CV_Assert( mean.size() == size );
        if( mean.type() != ctype )
        {
            cv::Mat tmp( mean.size(), ctype );
            mean.convertTo(tmp, ctype);
            mean = tmp;
        }
    }
    else
    {
        ctype = std::max(CV_MAT_DEPTH(ctype >= 0 ? ctype : type), CV_32F);

        cv::reduce( _src, _mean, takeRows ? 0 : 1, CV_REDUCE_AVG, ctype ); // Reduces a matrix to a vector, calculate the _mean

        mean = _mean.clone();
    }

    // Calculates the product of a matrix and its transposition. 
    cv::mulTransposed( data, _covar, ( (flags & FACE_COVAR_NORMAL) == 0 ) ^ takeRows, mean, (flags & FACE_COVAR_SCALE) != 0 ? 1./nsamples : 1, ctype );
}


}; /* namespace FACE */