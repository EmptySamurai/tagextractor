
#include <iostream>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <queue>
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>


using Eigen::Matrix;
using Eigen::Dynamic;
using Eigen::SparseMatrix;
using Eigen::Triplet;
using Eigen::RowMajor;


typedef float datatype;
using namespace std;
namespace py = pybind11;


typedef struct Tag {
    datatype probability;
    int index;
} Tag;

bool operator<(const Tag& lhs, const Tag& rhs)
{
  return lhs.probability < rhs.probability;
}


void getTagsIndices(SparseMatrix<datatype, RowMajor>& tfidfMatrix, int nTags, vector<vector<int>>& result, int batchsize) {

    int nDocs = tfidfMatrix.rows();
    int nTerms = tfidfMatrix.cols();

    
    int nPrecomputedRows = min(nDocs, batchsize);

    int nBatches = nDocs/nPrecomputedRows;
    
    if (nDocs % nPrecomputedRows != 0) {
        nBatches++;
    }

    priority_queue<Tag> q;
    Matrix<datatype,1,Dynamic> featureVector(nTerms);

    for (int b = 0; b < nBatches; b++) {
            
        int start = b*nPrecomputedRows;
        int nRows=min(nPrecomputedRows, nDocs-start);
        
        cout<<"Start processing batch "<<b<<endl;
        SparseMatrix<datatype, RowMajor> similaritySparse =tfidfMatrix.innerVectors(start, nRows) * tfidfMatrix.transpose();

        Matrix<datatype, Dynamic, Dynamic> similarityMatrix(similaritySparse);
        similaritySparse.resize(0, 0);
        
        for (int r = 0; r < nRows; r++) {
            int rowIndex = r+start;

            featureVector.fill(0);

            for (int j = 0; j < nDocs; j++) {
                datatype similarityValue = similarityMatrix(r, j);
                featureVector += tfidfMatrix.row(j)*similarityValue;
            }


            for (int j = 0; j < nTerms; j++) {
                if (featureVector[j]>0) {
                    if (q.size() < nTags)
                        q.push({.probability =-featureVector[j], .index=j});
                    else if (q.top().probability>-featureVector[j]) {
                        q.pop();
                        q.push({.probability =-featureVector[j], .index=j});
                    }
                }
            }


            for (int j = 0, k = q.size(); j < k; j++) {
                result[rowIndex].push_back(q.top().index);
                q.pop();
            }
        }
    }
}

typedef py::array_t<int, py::array::c_style | py::array::forcecast> indices_arr;
typedef py::array_t<datatype, py::array::c_style | py::array::forcecast> values_arr;

vector<vector<int>> get_tags_indices(indices_arr rows, indices_arr columns, values_arr values, 
int nDocs, int nTerms, int nTags,
 int batchsize) {

    auto rowsInfo = rows.request();
    auto columnsInfo = columns.request();
    auto valuesInfo = values.request();
    if (rowsInfo.ndim != 1 || columnsInfo.ndim != 1 || valuesInfo.ndim !=1)
        throw runtime_error("Number of vectors dimentions must be one");

    auto nValues = valuesInfo.size;

    if (rowsInfo.size != nValues || columnsInfo.size != nValues)
        throw runtime_error("Vectors sizes must be equal");

    auto rowsPtr = rows.data();
    auto columnsPtr = columns.data();
    auto valuesPtr = values.data();

    vector<Triplet < datatype >> triplets;
    triplets.reserve(nValues);
    for (auto i = 0; i < nValues; i++) {
        triplets.push_back(Triplet<datatype>(rowsPtr[i], columnsPtr[i], valuesPtr[i]));
    }

    SparseMatrix<datatype, RowMajor> tfidfMatrix(nDocs, nTerms);
    tfidfMatrix.setFromTriplets(triplets.begin(), triplets.end());

    vector<vector<int>> allTags(nDocs);
    getTagsIndices(tfidfMatrix, nTags, allTags, batchsize);
    return allTags;
}

PYBIND11_MODULE(native, m) {
    m.doc() = "Tagextractor native code"; // optional module docstring

    m.def("get_tags_indices", &get_tags_indices, "A function to get indices of tags");
}