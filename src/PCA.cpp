#include "FACE/PCA.hpp"

namespace FACE{

PCA::PCA(){

}

PCA::PCA( cv::InputArray data, cv::InputArray mean, int flags, int maxComponents ){
    operator()( data, mean, flags, maxComponents );
}

PCA::PCA( cv::InputArray data, cv::InputArray mean, int flags, double retainedVariance ){
    operator()( data, mean, flags, retainedVariance );
}

PCA& PCA::operator()( cv::InputArray _data, cv::InputArray mean, int flags, int maxComponents ){
    // The operator performs PCA of the supplied dataset. 对数据进行主成分分析

    cv::Mat data = _data.getMat();
    cv::Mat _mean = mean.getMat();
    int covar_flags = CV_COVAR_SCALE;
    // flag for cvCalcCovarMatrix, scale the covariance matrix coefficients by number of the vectors

    int len, in_count;
    cv::Size mean_sz;

    CV_Assert( data.channels() == 1 ); // 确保是灰度图

    /*
        enum cv::PCA::Flags

        Enumerator
            DATA_AS_ROW    : indicates that the input samples are stored as matrix rows
            DATA_AS_COL    : indicates that the input samples are stored as matrix columns 
    */
    if( flags & PCA::DATA_AS_COL ){
        len = data.rows;            // 数据xi的维度
        in_count = data.cols;       // 数据集中数据xi的数目
        covar_flags |= CV_COVAR_COLS;
        // CV_COVAR_COLS : flag for cvCalcCovarMatrix, all the input vectors are stored in a single matrix, as its columns
        
        mean_sz = cv::Size(1, len); // 均值表示成列矢量, Size_ (_Tp _width, _Tp _height)
    }
    else{
        len = data.cols;
        in_count = data.rows;
        covar_flags |= CV_COVAR_ROWS;
        mean_sz = Size(len, 1); // 均值表示成行矢量
    }

    int count = std::min(len, in_count), out_count = count;

    if( maxComponents > 0 )
        out_count = std::min(count, maxComponents);
    
    if( len <= in_count ) // 数据集中数据的数目 >= 数据的维度
        covar_flags |= CV_COVAR_NORMAL;

    int ctype = std::max( CV_32F, data.depth() );
    m_mean.create( mean_sz, ctype );

    cv::Mat covar( count, count, ctype );

    if( !_mean.empty() )
    {
        CV_Assert( _mean.size() == mean_sz );
        _mean.convertTo( m_mean, ctype );
        covar_flags |= CV_COVAR_USE_AVG;
    }

    cv::calcCovarMatrix( data, covar, m_mean, covar_flags, ctype ); //Calculates the covariance matrix of a set of vectors. 
    cv::eigen( covar, m_eigenvalues, m_eigenvectors );

    /*
        flag for cvCalcCovarMatrix, transpose([v1-avg, v2-avg,...]) * [v1-avg,v2-avg,...]
        #define CV_COVAR_SCRAMBLED 0

        flag for cvCalcCovarMatrix, [v1-avg, v2-avg,...] * transpose([v1-avg,v2-avg,...]) 
        #define CV_COVAR_NORMAL    1
    */

    // "scrambled" way to compute PCA (when cols(A)>rows(A)):
    // B = A'A; B*x=b*x; 待计算的特征值b和特征矢量x
    // 但是，如果A的cols(A) > rows(A)，那么B的尺度会比较大， 转而计算C=AA'的特征值c特征矢量y，然后将c,y变换为b,x : b = c, x=A'*y
    // C = AA'; C*y=c*y -> AA'*y=c*y -> A'A*(A'*y)=c*(A'*y) -> b = c, x=A'*y

    if( !(covar_flags & CV_COVAR_NORMAL) )
    {
        // CV_PCA_DATA_AS_ROW: cols(A)>rows(A). x=A'*y -> x'=y'*A
        // CV_PCA_DATA_AS_COL: rows(A)>cols(A). x=A''*y -> x'=y'*A'
        cv::Mat tmp_data, tmp_mean = cv::repeat( m_mean, data.rows / m_mean.rows, data.cols / m_mean.cols );
        if( data.type() != ctype || tmp_mean.data == m_mean.data )
        {
            data.convertTo( tmp_data, ctype );
            cv::subtract( tmp_data, tmp_mean, tmp_data );
        }
        else
        {
            cv::subtract( data, tmp_mean, tmp_mean );
            tmp_data = tmp_mean;
        }

        cv::Mat evects1(count, len, ctype);

        // The function cv::gemm performs generalized matrix multiplication similar to the gemm functions in BLAS level 3
        cv::gemm( m_eigenvectors, tmp_data, 1, Mat(), 0, evects1,
            (flags & CV_PCA_DATA_AS_COL) ? CV_GEMM_B_T : 0);

        m_eigenvectors = evects1;

        // normalize eigenvectors
        int i;
        for( i = 0; i < out_count; i++ )
        {
            cv::Mat vec = m_eigenvectors.row(i);
            cv::normalize(vec, vec); // 归一化为单位长度
        }
    }

    if( count > out_count )
    {
        // use clone() to physically copy the data and thus deallocate the original matrices
        m_eigenvalues = m_eigenvalues.rowRange(0,out_count).clone();
        m_eigenvectors = m_eigenvectors.rowRange(0,out_count).clone();
    }
    return *this;
}

void PCA::write( cv::FileStorage& fs ) const
{
    CV_Assert( fs.isOpened() );

    fs << "name" << "PCA";
    fs << "vectors" << m_eigenvectors;
    fs << "values" << m_eigenvalues;
    fs << "mean" << m_mean;
}

void PCA::read(const cv::FileNode& fn )
{
    CV_Assert( !fn.empty() );
    CV_Assert( (cv::String)fn["name"] == "PCA" );

    cv::read(fn["vectors"], m_eigenvectors);
    cv::read(fn["values"], m_eigenvalues);
    cv::read(fn["mean"], m_mean);
}

template <typename T>
int computeCumulativeEnergy(const Mat& eigenvalues, double retainedVariance)
{
    CV_DbgAssert( eigenvalues.type() == DataType<T>::type );

    Mat g(eigenvalues.size(), DataType<T>::type);

    for(int ig = 0; ig < g.rows; ig++)
    {
        g.at<T>(ig, 0) = 0;
        for(int im = 0; im <= ig; im++)
        {
            g.at<T>(ig,0) += eigenvalues.at<T>(im,0);
        }
    }

    int L;

    for(L = 0; L < eigenvalues.rows; L++)
    {
        double energy = g.at<T>(L, 0) / g.at<T>(g.rows - 1, 0);
        if(energy > retainedVariance)
            break;
    }

    L = std::max(2, L);

    return L;
}

PCA& PCA::operator()(InputArray _data, InputArray __mean, int flags, double retainedVariance)
{
    Mat data = _data.getMat(), _mean = __mean.getMat();
    int covar_flags = CV_COVAR_SCALE;
    int len, in_count;
    Size mean_sz;

    CV_Assert( data.channels() == 1 );
    if( flags & CV_PCA_DATA_AS_COL )
    {
        len = data.rows;
        in_count = data.cols;
        covar_flags |= CV_COVAR_COLS;
        mean_sz = Size(1, len);
    }
    else
    {
        len = data.cols;
        in_count = data.rows;
        covar_flags |= CV_COVAR_ROWS;
        mean_sz = Size(len, 1);
    }

    CV_Assert( retainedVariance > 0 && retainedVariance <= 1 );

    int count = std::min(len, in_count);

    // "scrambled" way to compute PCA (when cols(A)>rows(A)):
    // B = A'A; B*x=b*x; C = AA'; C*y=c*y -> AA'*y=c*y -> A'A*(A'*y)=c*(A'*y) -> c = b, x=A'*y
    if( len <= in_count )
        covar_flags |= CV_COVAR_NORMAL;

    int ctype = std::max(CV_32F, data.depth());
    mean.create( mean_sz, ctype );

    Mat covar( count, count, ctype );

    if( !_mean.empty() )
    {
        CV_Assert( _mean.size() == mean_sz );
        _mean.convertTo(mean, ctype);
    }

    calcCovarMatrix( data, covar, mean, covar_flags, ctype );
    eigen( covar, eigenvalues, eigenvectors );

    if( !(covar_flags & CV_COVAR_NORMAL) )
    {
        // CV_PCA_DATA_AS_ROW: cols(A)>rows(A). x=A'*y -> x'=y'*A
        // CV_PCA_DATA_AS_COL: rows(A)>cols(A). x=A''*y -> x'=y'*A'
        Mat tmp_data, tmp_mean = repeat(mean, data.rows/mean.rows, data.cols/mean.cols);
        if( data.type() != ctype || tmp_mean.data == mean.data )
        {
            data.convertTo( tmp_data, ctype );
            subtract( tmp_data, tmp_mean, tmp_data );
        }
        else
        {
            subtract( data, tmp_mean, tmp_mean );
            tmp_data = tmp_mean;
        }

        Mat evects1(count, len, ctype);
        gemm( eigenvectors, tmp_data, 1, Mat(), 0, evects1,
            (flags & CV_PCA_DATA_AS_COL) ? CV_GEMM_B_T : 0);
        eigenvectors = evects1;

        // normalize all eigenvectors
        int i;
        for( i = 0; i < eigenvectors.rows; i++ )
        {
            Mat vec = eigenvectors.row(i);
            normalize(vec, vec);
        }
    }

    // compute the cumulative energy content for each eigenvector
    int L;
    if (ctype == CV_32F)
        L = computeCumulativeEnergy<float>(eigenvalues, retainedVariance);
    else
        L = computeCumulativeEnergy<double>(eigenvalues, retainedVariance);

    // use clone() to physically copy the data and thus deallocate the original matrices
    eigenvalues = eigenvalues.rowRange(0,L).clone();
    eigenvectors = eigenvectors.rowRange(0,L).clone();

    return *this;
}

void PCA::project(InputArray _data, OutputArray result) const
{
    // 计算观测矢量x的k个主成分y: y = W^T(x-u), W=(v1, v2, ..., vk)
    Mat data = _data.getMat();
    CV_Assert( !mean.empty() && !eigenvectors.empty() &&
        ((mean.rows == 1 && mean.cols == data.cols) || (mean.cols == 1 && mean.rows == data.rows)));
    
    // The function cv::repeat duplicates the input array one or more times along each of the two axes: 
    Mat tmp_data, tmp_mean = repeat(mean, data.rows/mean.rows, data.cols/mean.cols);
    int ctype = mean.type();
    if( data.type() != ctype || tmp_mean.data == mean.data )
    {
        data.convertTo( tmp_data, ctype );
        subtract( tmp_data, tmp_mean, tmp_data ); // 计算x-u
    }
    else
    {
        subtract( data, tmp_mean, tmp_mean );
        tmp_data = tmp_mean;
    }
    if( mean.rows == 1 )
        gemm( tmp_data, eigenvectors, 1, Mat(), 0, result, GEMM_2_T );
    /*
        Enumerator
            GEMM_1_T    :       transposes src1
            GEMM_2_T    :       transposes src2
            GEMM_3_T    :       transposes src3 
    */
        // result = 1*tmp_data*eigenvectors^T + 0 * Mat()
    else
        gemm( eigenvectors, tmp_data, 1, Mat(), 0, result, 0 );
        // result = 1*eigenvectors^T*tmp_data + 0 * Mat()
}

Mat PCA::project(InputArray data) const
{
    Mat result;
    project(data, result);
    return result;
}

void PCA::backProject(InputArray _data, OutputArray result) const
{
    // 重构: x=Wy+u
    
    Mat data = _data.getMat();
    CV_Assert( !mean.empty() && !eigenvectors.empty() &&
        ((mean.rows == 1 && eigenvectors.rows == data.cols) ||
         (mean.cols == 1 && eigenvectors.rows == data.rows)));

    Mat tmp_data, tmp_mean;
    data.convertTo(tmp_data, mean.type());
    if( mean.rows == 1 )
    {
        tmp_mean = repeat(mean, data.rows, 1);
        gemm( tmp_data, eigenvectors, 1, tmp_mean, 1, result, 0 );
        // result = 1*tem_data^T*eigenvectors + 1*tmp_mean
    }
    else
    {
        tmp_mean = repeat(mean, 1, data.cols);
        gemm( eigenvectors, tmp_data, 1, tmp_mean, 1, result, GEMM_1_T );
    }
}

Mat PCA::backProject(InputArray data) const
{
    Mat result;
    backProject(data, result);
    return result;
}

}

void cv::PCACompute(InputArray data, InputOutputArray mean,
                    OutputArray eigenvectors, int maxComponents)
{
    CV_INSTRUMENT_REGION()

    PCA pca;
    pca(data, mean, 0, maxComponents);
    pca.mean.copyTo(mean);
    pca.eigenvectors.copyTo(eigenvectors);
}

void cv::PCACompute(InputArray data, InputOutputArray mean,
                    OutputArray eigenvectors, double retainedVariance)
{
    CV_INSTRUMENT_REGION()

    PCA pca;
    pca(data, mean, 0, retainedVariance);
    pca.mean.copyTo(mean);
    pca.eigenvectors.copyTo(eigenvectors);
}

void cv::PCAProject( cv::InputArray data, cv::InputArray mean, cv::InputArray eigenvectors, cv::OutputArray result ){
    CV_INSTRUMENT_REGION()

    PCA pca;
    pca.m_mean = mean.getMat();
    pca.m_eigenvectors = eigenvectors.getMat();
    pca.project(data, result);
}

void cv::PCABackProject( cv::InputArray data, cv::InputArray mean, cv::InputArray eigenvectors, cv::OutputArray result ){
    CV_INSTRUMENT_REGION()

    PCA pca;
    pca.m_mean = mean.getMat();
    pca.m_eigenvectors = eigenvectors.getMat();
    pca.backProject(data, result);
}



}; /* namespace FACE */