
#include <iostream>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <queue>
#include <vector>
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

typedef struct Tag
{
    datatype probability;
    int index;
} Tag;

bool operator<(const Tag &lhs, const Tag &rhs)
{
    return lhs.probability < rhs.probability;
}

void getTagsIndicesWithHighestProbability(vector<datatype> &featureVector, priority_queue<Tag> &q, int nTags, vector<int> &out)
{

    for (int i = 0; i < featureVector.size(); i++)
    {
        if (featureVector[i] > 0)
        {
            if (q.size() < nTags)
                q.push({-featureVector[i], i});
            else if (q.top().probability > -featureVector[i])
            {
                q.pop();
                q.push({-featureVector[i], i});
            }
        }
    }

    for (int i = 0, k = q.size(); i < k; i++)
    {
        out.push_back(q.top().index);
        q.pop();
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
    priority_queue<Tag> q;

    for (int b = 0; b < nBatches; b++)
    {

        int start = b * nPrecomputedRows;
        int nRows = min(nPrecomputedRows, nDocs - start);

        cout << "Start processing batch " << b << endl;
        SparseMatrix<datatype, RowMajor> similaritySparse = tfidfMatrix.innerVectors(start, nRows) * tfidfMatrix.transpose();

        Matrix<datatype, Dynamic, Dynamic> similarityMatrix(similaritySparse);
        similaritySparse.resize(0, 0);

        for (int r = 0; r < nRows; r++)
        {
            int rowIndex = r + start;
            fillFeatureVector(tfidfMatrix, similarityMatrix, r, featureVector);
            getTagsIndicesWithHighestProbability(featureVector, q, nTags, result[rowIndex]);
        }
    }
}

typedef py::array_t<int, py::array::c_style | py::array::forcecast> indices_arr;
typedef py::array_t<datatype, py::array::c_style | py::array::forcecast> values_arr;

vector<vector<int>> get_tags_indices(indices_arr rows, indices_arr columns, values_arr values,
                                     int nDocs, int nTerms, int nTags,
                                     int batchsize)
{

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

    m.def("get_tags_indices", &get_tags_indices, "A function to get indices of tags");
}