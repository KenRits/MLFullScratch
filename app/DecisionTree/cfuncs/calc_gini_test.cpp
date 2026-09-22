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
    bool updated;
    int col_idx;
    float threshold;
    float left_gini;
    float right_gini;

    public:
    Split () : updated(false), col_idx(0), threshold(0), left_gini(0), right_gini(0) {}

    void update (int _col_idx, float _threshold, float _left_gini, float _right_gini) {
        updated = true;
        col_idx = _col_idx;
        threshold = _threshold;
        left_gini = _left_gini;
        right_gini = _right_gini;
    }
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

Split find_min_gini_threshold (std::vector<std::vector<float>> &X_subset_T, std::vector<int> &y_subset, std::vector<int> &labels, int bin_num) {
    int col_n = X_subset_T.size();
    int data_n = X_subset_T.at(0).size();

    int min_gini_col_idx = 0;
    float min_gini_threshold = 0;
    float split_left_gini = 0;
    float split_right_gini = 0;

    // 返り値の用意
    Split split;

    float min_gini = 2; //giniは[0, 1]の範囲であるから, 2で初期化すれば必ず更新される.

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
        
        // columnとy_subsetをindicesの通り並べ替える.

        std::vector<float> column_sorted(data_n);
        std::vector<int> y_sorted(data_n);

        for (int i=0; i<data_n; i++) {
            column_sorted.at(i) = column.at(indices.at(i));
            y_sorted.at(i) = y_subset.at(indices.at(i));
        }
        
        // ヒストグラムを作成するために, 最小値とbinの幅を取得. bin_width
        float x_min = column_sorted.at(0); // columnの最小値
        float x_max = column_sorted.at(data_n-1); // columnの最大値
        float bin_width = (x_max - x_min) / bin_num; // binの幅

        // x_minからbin_widthずつ足していくことで閾値を得て, その閾値によりy_sortedを左右に分ける. 
        int split_idx = 1; // column_sorted[:split_idx]とcolumn_sorted[split_idx:]で二つに分ける.
        
        // ビンとビンの間を走査していくイメージ(閾値ごとに1loop). 植木算より, ビンとビンの間の数はbin_num - 1 
        for (int i=0; i<bin_num-1; i++) {
            float threshold = x_min + (i+1) * bin_width; // threshold更新
            
            // column_sortedがthresholdを超えるまで大きい側へ添え字を移動.
            while (column_sorted.at(split_idx) < threshold) {
                split_idx++;
            } 
            std::vector<int> y_sorted_left(y_sorted.begin(), y_sorted.begin()+split_idx);
            std::vector<int> y_sorted_right(y_sorted.begin()+split_idx, y_sorted.end());
            
            // gini不純度の計算
            float left_gini = calc_gini(y_sorted_left, labels);
            float right_gini = calc_gini(y_sorted_right, labels);

            float gini = (left_gini * y_sorted_left.size() + right_gini * y_sorted_right.size()) / data_n;
            
            // gini不純度が[0, 1]に収まっていることを確認.
            assert(0 <= left_gini && left_gini <= 1);
            assert(0 <= right_gini && right_gini <= 1);
            assert(0 <= gini && gini <= 1);
            
            // giniが暫定の最小値より小さい場合, 更新
            if (gini < min_gini) {
                min_gini = gini;
                min_gini_threshold = (column_sorted.at(split_idx) + column_sorted.at(split_idx+1)) / 2;

                split.update(col_idx, min_gini_threshold, left_gini, right_gini);
            }
        }
    }
    assert(split.updated);
    return split;
}

int main() {
    std::vector<std::vector<float>> X_T = 
    {
        {1, 2, 3},
        {3, 0, -8}
    };
    std::vector<int> y = {0, 0, 1};
    std::vector<int> labels = {0, 1};
    int bin_num = 256;

    try {
        Split split = find_min_gini_threshold(X_T, y, labels, bin_num);
        std::cout << split.col_idx << "\n";
        std::cout << split.threshold << "\n";

        return 0;
    } catch(char *str) {
        std::cout << *str;
        return -1;
    }
}