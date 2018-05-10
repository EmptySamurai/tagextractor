
#include <iostream>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <queue>
#include <vector>
#include <algorithm>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

using Eigen::Dynamic;
using Eigen::Matrix;
using Eigen::RowMajor;
using Eigen::SparseMatrix;
using Eigen::Triplet;

typedef float datatype;
using namespace std;
namespace py = pybind11;


template <class T>
struct IndicesCompare{
    IndicesCompare( const vector<T>& v ) : _v(v) {}
    bool operator ()(int a, int b) { return _v[a] > _v[b]; }
    const vector<T>& _v;
};

void getTagsIndicesWithHighestProbability(vector<datatype> &featureVector, vector<int> &indices, int nTags, vector<int> &out)
{
    partial_sort(indices.begin(), indices.begin()+nTags, indices.end(), IndicesCompare<datatype>(featureVector));
    
    out.reserve(nTags);

    for (auto i=0; i<nTags; i++) {
        auto ind = indices[i];
        if (featureVector[ind] != 0) {
            out.push_back(ind);
        } else {
            break;
        }
    }
}

void fillFeatureVector(SparseMatrix<datatype, RowMajor> &tfidfMatrix, Matrix<datatype, Dynamic, Dynamic> &similarityMatrix, int row, vector<datatype> &featureVector)
{
    fill(featureVector.begin(), featureVector.end(), 0.0);
    for (int i = 0; i < tfidfMatrix.rows(); i++)
    {
        datatype similarityValue = similarityMatrix(row, i);
        for (SparseMatrix<datatype, RowMajor>::InnerIterator it(tfidfMatrix, i); it; ++it)
        {
            featureVector[it.col()] += similarityValue * it.value();
        }
    }
}

void getTagsIndices(SparseMatrix<datatype, RowMajor> &tfidfMatrix, int nTags, vector<vector<int>> &result, int batchsize)
{
    int nDocs = tfidfMatrix.rows();
    int nTerms = tfidfMatrix.cols();

    int nPrecomputedRows = min(nDocs, batchsize);
    int nBatches = nDocs / nPrecomputedRows;
    if (nDocs % nPrecomputedRows != 0)
    {
        nBatches++;
    }

    vector<datatype> featureVector(nTerms);
    vector<int> indices(nTerms);
    iota(indices.begin(), indices.end(), 0);

    for (int b = 0; b < nBatches; b++)
    {

        int start = b * nPrecomputedRows;
        int nRows = min(nPrecomputedRows, nDocs - start);

        cout << "Start processing batch " << b << endl;
        SparseMatrix<datatype, RowMajor> similaritySparse = tfidfMatrix.innerVectors(start, nRows) * tfidfMatrix.transpose();

        Matrix<datatype, Dynamic, Dynamic> similarityMatrix(similaritySparse);
        similaritySparse.resize(0, 0);

        #pragma omp parallel for firstprivate(featureVector, indices)
        for (int r = 0; r < nRows; r++)
        {
            int rowIndex = r + start;
            fillFeatureVector(tfidfMatrix, similarityMatrix, r, featureVector);
            getTagsIndicesWithHighestProbability(featureVector, indices, nTags, result[rowIndex]);
        }
    }
}

typedef py::array_t<int, py::array::c_style | py::array::forcecast> indices_arr;
typedef py::array_t<datatype, py::array::c_style | py::array::forcecast> values_arr;

vector<vector<int>> getTagsIndicesInterface(indices_arr rows, indices_arr columns, values_arr values,
                                     int nDocs, int nTerms, int nTags,
                                     int batchsize)
{
    nTags = min(nTags, nTerms);

    auto rowsInfo = rows.request();
    auto columnsInfo = columns.request();
    auto valuesInfo = values.request();
    if (rowsInfo.ndim != 1 || columnsInfo.ndim != 1 || valuesInfo.ndim != 1)
        throw runtime_error("Number of vectors dimentions must be one");

    auto nValues = valuesInfo.size;

    if (rowsInfo.size != nValues || columnsInfo.size != nValues)
        throw runtime_error("Vectors sizes must be equal");

    auto rowsPtr = rows.data();
    auto columnsPtr = columns.data();
    auto valuesPtr = values.data();

    vector<Triplet<datatype>> triplets;
    triplets.reserve(nValues);
    for (auto i = 0; i < nValues; i++)
    {
        triplets.push_back(Triplet<datatype>(rowsPtr[i], columnsPtr[i], valuesPtr[i]));
    }

    SparseMatrix<datatype, RowMajor> tfidfMatrix(nDocs, nTerms);
    tfidfMatrix.setFromTriplets(triplets.begin(), triplets.end());

    vector<vector<int>> allTags(nDocs);
    getTagsIndices(tfidfMatrix, nTags, allTags, batchsize);
    return allTags;
}

PYBIND11_MODULE(native, m)
{
    m.doc() = "Tagextractor native code"; // optional module docstring

    m.def("get_tags_indices", &getTagsIndicesInterface, "A function to get indices of tags");
}