#include <iostream>
#include <vector>
#include <string>
#include <cassert>
#include <numeric>
#include <algorithm>
#include <unordered_set>

void hello() {
    std::cout << "This is a test." << std::endl;
}

// Python側に公開する, 返り値用の構造体.
extern "C" struct SplitForC {
    bool updated;
    int col_idx;
    float threshold;
    float left_gini;
    float right_gini;
};

class Split {
    public:
    bool updated;
    int col_idx;
    float threshold;
    float left_gini;
    float right_gini;

    Split () : updated(false), col_idx(0), threshold(0), left_gini(0), right_gini(0) {}

    void update (int _col_idx, float _threshold, float _left_gini, float _right_gini) {
        updated = true;
        col_idx = _col_idx;
        threshold = _threshold;
        left_gini = _left_gini;
        right_gini = _right_gini;
    }
    
    // Python側に返すために定義.
    SplitForC create_obj_for_C() {
        SplitForC s;
        s.updated = updated;
        s.col_idx = col_idx;
        s.threshold = threshold;
        s.left_gini = left_gini;
        s.right_gini = right_gini;

        return s;
    }
};



float calc_gini(const std::vector<int> &y_subset, const std::unordered_set<int> &labels) {

    assert(y_subset.size() >= 1);
    assert(labels.size() >= 1);
    // label_counts ... labelsと同じサイズのvector. labelsと同じ位置に, そのlabelの個数が入る.
    // つまり, y_subsetに出てくるlabels[i]の個数が, label_counts[i]に入る.
    std::vector<int> label_counts(labels.size(), 0);

    for (const int label : y_subset) {
        for (int label_idx=0; label_idx<labels.size(); ++label_idx) {
            if (labels.contains(label)) {
                ++label_counts.at(label_idx);
                break;
            } else if (label_idx == labels.size() - 1) {
                throw std::invalid_argument("labelsに登録されていない値がy_subsetに入っています: " + std::to_string(label));
            }
        }
    }

    // label_countsからginiを計算
    float p_sq_sum = 0;
    for (int label_count : label_counts) {
        float p = static_cast<float>(label_count) / y_subset.size();
        float p_sq = p * p;
        p_sq_sum = p_sq_sum + p_sq;
    }
    float gini = 1 - p_sq_sum;

    assert(gini >= 0 && gini < 1);

    return gini;
}

Split find_min_gini_threshold(const std::vector<std::vector<float>>& X_subset_T, const std::vector<int> &y_subset, const int bin_num) {

    int col_n = X_subset_T.size();
    int data_n = X_subset_T.at(0).size();

    if (data_n != y_subset.size()) {
        throw std::invalid_argument("配列のサイズがX_subset_Tとy_subsetで一致しません.");
    }

    const std::unordered_set<int> labels(y_subset.begin(), y_subset.end());

    int min_gini_col_idx = 0; // 選ばれたカラムのインデックス. 
    float min_gini_threshold = 0; // 選ばれたカラムにおける, ジニ不純度を最小化する閾値
    float split_left_gini = 0; // 分割後の, 左子のジニ不純度
    float split_right_gini = 0; // 分割後の, 右子のジニ不純度

    // 返り値の用意
    Split split;

    float min_gini = 2; //giniは[0, 1]の範囲であるから, 2で初期化すれば必ず更新される.

    for (int col_idx=0; col_idx<col_n; col_idx++) {
        const std::vector<float>& column = X_subset_T.at(col_idx);

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
        int split_idx = 0; // column_sorted[:split_idx]とcolumn_sorted[split_idx:]で二つに分ける.
        
        // ビンとビンの間を走査していくイメージ(閾値ごとに1loop). 植木算より, ビンとビンの間の数はbin_num - 1 
        for (int i=0; i<bin_num-1; i++) {
            float threshold = x_min + (i+1) * bin_width; // threshold更新

            // column_sortedがthresholdを超えるまで大きい側へ添え字を移動.
            bool split_idx_updated = false;
            while (column_sorted.at(split_idx) < threshold) {
                split_idx_updated = true;
                ++split_idx;
            }
            
            // threshldを更新したにもかかわらず, split_idxが変わらない場合 = ビンが空である場合は, 
            // そのビンの処理をスキップする.
            if (!split_idx_updated) {
                continue;
            }

            const std::vector<int> y_sorted_left(y_sorted.begin(), y_sorted.begin()+split_idx);
            const std::vector<int> y_sorted_right(y_sorted.begin()+split_idx, y_sorted.end());
            
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
                min_gini_threshold = (column_sorted.at(split_idx-1) + column_sorted.at(split_idx)) / 2;

                split.update(col_idx, min_gini_threshold, left_gini, right_gini);
            }
        }
    }
    
    return split;
    
}

extern "C" __declspec(dllexport) SplitForC find_min_gini_threshold_shared(float* X_subset_flatten_ptr, int* y_subset_ptr, int data_n, int col_n, int bin_num) {
    // X_subset_T, y_subsetを作る. 0で初期化.
    std::vector<std::vector<float>> X_subset_T; 
    std::vector<int> y_subset;

    X_subset_T.assign(col_n, std::vector<float>(data_n, 0));
    y_subset.assign(data_n, 0);
    
    for (int i = 0; i < data_n; ++i) {
        y_subset.at(i) = y_subset_ptr[i];
        for (int j = 0; j < col_n; ++j) {
            X_subset_T.at(j).at(i) = X_subset_flatten_ptr[i*col_n + j]; // 転値するため, X_subset_T.at(j).at(i)の順でアクセス.
        }
    }

    // X_subset_T確認用

    for (int i=0; i < X_subset_T.size(); ++i) {
        for (int j = 0; j < X_subset_T.at(0).size(); ++j) {
            std::cout << X_subset_T.at(i).at(j);
        }
        std::cout << "\n" << std::endl;
    }

    Split split = find_min_gini_threshold(X_subset_T, y_subset, bin_num);
    SplitForC split_for_c = split.create_obj_for_C();

    return split_for_c;
}

int main() {
    
    std::vector<std::vector<float>> X_T = 
    {
        {3, 3, -3, -2},
        {3, 2, 1, 4},
        {3, 3, 1, 0}
    };
    std::vector<int> y = {1, 1, 1, 0};

    std::vector<std::vector<float>> X = 
    {
        {3, 3, 3},
        {3, 2, 3}, 
        {-3, 1, 1},
        {-2, 4, 0}
    };

    float X_C[12] = {3, 3, 3, 3, 2, 3, -3, 1, 1, -2, 4, 0};
    int y_C[4] = {1, 1, 1, 0};
    float* X_C_ptr = X_C;

    SplitForC split_for_C = find_min_gini_threshold_shared(X_C_ptr, y_C, 4, 3, 256);

    std::cout << "col_idx: " << split_for_C.col_idx << "\nthreshold: " << split_for_C.threshold << "\nleft_gini: " << split_for_C.left_gini << "\nrigt_gini: " << split_for_C.right_gini << std::endl;

    Split split = find_min_gini_threshold(X_T, y, 256);

    std::cout << "col_idx: " << split.col_idx << "\nthreshold: " << split.threshold << "\nleft_gini: " << split.left_gini << "\nrigt_gini: " << split.right_gini << std::endl;
}