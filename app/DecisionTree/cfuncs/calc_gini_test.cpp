#include <iostream>
#include <vector>
#include <string>
#include <cassert>
#include <numeric>
#include <algorithm>

void hello() {
    std::cout << "This is a test." << std::endl;
}

struct Split {
    int col_idx;
    float threshold;
    float left_gini;
    float right_gini;
};

float calc_gini(std::vector<int> &y_subset, std::vector<int> &labels) {
    assert(y_subset.size() >= 1);
    assert(labels.size() >= 1);
    // label_counts ... labelsと同じサイズのvector. labelsと同じ位置に, そのlabelの個数が入る.
    // つまり, y_subsetに出てくるlabels[i]の個数が, label_counts[i]に入る.
    std::vector<int> label_counts(labels.size(), 0);

    for (int i=0; i<y_subset.size(); i++) {
        for (int label_idx=0; label_idx<labels.size(); label_idx++) {
            if (labels.at(label_idx) == y_subset.at(i)) {
                label_counts.at(label_idx) = label_counts.at(label_idx) + 1;
                break;
            } else if (label_idx == labels.size() - 1) {
                throw "labelsに登録されていない値がy_subsetに入っています: " + std::to_string(labels.at(label_idx));
            }
        }
    }

    // label_countsからginiを計算
    float p_sq_sum = 0;
    for (int i=0; i<label_counts.size(); i++) {
        float p = static_cast<float>(label_counts.at(i)) / y_subset.size();
        float p_sq = p * p;
        p_sq_sum = p_sq_sum + p_sq;
    }

    return 1 - p_sq_sum;
}

Split find_min_gini_threshold (std::vector<std::vector<float>> &X_subset_T, std::vector<int> &y_subset, int bin_num) {
    int col_n = X_subset_T.size();
    int data_n = X_subset_T.at(0).size();

    for (int col_idx=0; col_idx<col_n; col_idx++) {
        std::vector<float>& column = X_subset_T.at(col_idx);

        // columnを基準に, インデックスのリストを昇順にソートする.
        std::vector<int> indices(data_n); 
        std::iota(indices.begin(), indices.end(), 0); // indices = {0, 1, ... , datan-1}
        std::sort(
            indices.begin(), 
            indices.end(), 
            [&column] (int i, int j) {
            return column.at(i) < column.at(j);
            }
        );
        
        // 出力
        for (int i=0; i<indices.size(); i++) {
            std::cout << indices.at(i) << " ";
        }
        std::cout << "\n" << std::endl;
    }
    return Split();
}

int main() {
    std::vector<std::vector<float>> X_T = 
    {
        {1, 3.5, 2},
        {3, 0, -8}
    };
    std::vector<int> y = {0, 0, 1, 1, 0};

    try {
        find_min_gini_threshold(X_T, y, 256);
        return 0;
    } catch(char *str) {
        std::cout << *str;
        return -1;
    }
}