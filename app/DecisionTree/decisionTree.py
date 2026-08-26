from model import Model
import exception

import numpy as np

class Node:
    def __init__(self, path, label, gini, sample_n):
        self.path = path
        self._is_leaf = False
        self._left = None
        self._right = None
        self._label = label
        self._gini = gini
        self._sample_n = sample_n
        self._parent = None

    def get_label(self):
        return self._label
    
    def get_gini(self):
        return self._gini
    
    def get_sample_n(self):
        return self._sample_n

    def get_left(self):
        return self._left

    def get_right(self):
        return self._right

    def get_parent(self):
        return self._parent

    def set_parent(self, parent):
        self._parent = parent

class LeafNode(Node):
    def __init__(self, path, label, gini, sample_n):
        super().__init__(path, label, gini, sample_n)
        self._is_leaf = True

    def set_left(self, left):
        raise exception.InvalidNodeSettingException("葉ノードに対して子ノードを設定することはできません.")
    
    def set_right(self, right):
        raise exception.InvalidNodeSettingException("葉ノードに対して子ノードを設定することはできません.")

    def is_leaf(self):
        return self._is_leaf

    def get_tree_dict(self):
        return {self.path: self}
    
    def predict(self, x):
        if self._label is not None:
            return self._label
        else:
            raise exception.InvalidNodeSettingException("葉ノードにラベルが存在しません. ノードを作成する際は必ずNode.set_paramsを実行し, Node.predictを実行する前にラベルをはじめとした各種パラメータを設定してください.")

class MiddleNode(Node):
    def __init__(self, path, label, gini, sample_n, col_idx, threshold):
        super().__init__(path, label, gini, sample_n)
        self._col_idx = col_idx
        self._threshold = threshold
    
    def set_left(self, left):
        self._left = left
        left.set_parent(self)

    def set_right(self, right):
        self._right = right
        right.set_parent(self)

    def is_leaf(self):
        return self._is_leaf

    def predict(self, x):
        if x[self._col_idx] < self._threshold:
            prdicted_label = self._left.predict(x)
        else:
            prdicted_label = self._right.predict(x)
        return prdicted_label

    def get_tree_dict(self):
        dleft = self._left.get_tree_dict()
        dright = self._right.get_tree_dict()
        dself = {self.path: self}
        return dict(**dself, **dleft, **dright)

class DecisionTreeClassifer(Model):
    def __init__(self):
        super().__init__()
        self.root = None
        self.ROOT_CHAR = "."
        self.LEFT = "l"
        self.RIGHT = "r"
        self.DELIMITER = "/"

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, max_depth=5, bin_num=256, ccp_alpha=None):

        self.labels = np.unique(y_train)
        self.d = X_train.shape[1]

        root_gini = self._gini(y_train)
        self.root = self._build_tree(self.ROOT_CHAR, X_train, y_train, depth=0, max_depth=max_depth, gini=root_gini, bin_num=bin_num)
        if ccp_alpha is not None:
            pruned_node_count = self._prune(ccp_alpha)
            print(f"{pruned_node_count} leaves have pruned.")

    def predict(self, X: np.ndarray):
        result = np.zeros(len(X))
        if self.root is not None:
            for i, x in enumerate(X):
                result[i] = self.root.predict(x)
            return result
        else:
            raise exception.FunctionUndefinedException("まずDecisionTree.fitを実行してください.")
        
    def _find_min_gini_threshold(self, X_subset, y_subset, bin_num=256):
        max_a = np.max(X_subset, axis=0)
        min_a = np.min(X_subset, axis=0)
        min_gini = np.inf
        min_gini_col_idx = None
        min_gini_threshold = None
        splited_left_gini = None
        splited_right_gini = None

        for col_idx in range(self.d):

            column = (X_subset.T[col_idx] - min_a[col_idx]) / (max_a[col_idx] - min_a[col_idx] + 1e-5)
            bins = np.linspace(0, 1, bin_num)

            for bin_idx in range(0, bin_num-1):
                threshold = (bins[bin_idx] + bins[bin_idx+1]) / 2
                left = y_subset[column < threshold]
                right = y_subset[column >= threshold]

                if len(left) == 0 or len(right) == 0:
                    continue

                left_gini = self._gini(left)
                right_gini = self._gini(right)
                gini = (len(left) * left_gini + len(right) * right_gini) / len(y_subset)

                if gini < min_gini:
                    min_gini = gini
                    min_gini_col_idx = col_idx
                    min_gini_threshold = threshold
                    splited_left_gini = left_gini
                    splited_right_gini = right_gini

        min_gini_threshold = min_gini_threshold * (max_a[min_gini_col_idx] - min_a[min_gini_col_idx] + 1e-5) + min_a[min_gini_col_idx]
        return min_gini_col_idx, min_gini_threshold, splited_left_gini, splited_right_gini
    
    def _build_tree(self, path, X_subset, y_subset, depth, max_depth, gini, bin_num=256, ):
        
        if len(y_subset) == 0 or len(X_subset) == 0:
            raise exception.NonDataException()

        y_unique, counts = np.unique(y_subset, return_counts=True)
        label = y_unique[np.argmax(counts)]
        X_unique = np.unique(X_subset, axis=0)

        if len(y_unique) == 1 or len(X_unique) == 1:
            node = LeafNode(path, label, gini=gini, sample_n=len(y_subset))
            return node

        if depth >= max_depth:
            node = LeafNode(path, label, gini=gini, sample_n=len(y_subset))
            return node

        min_gini_col_idx, min_gini_threshold, left_gini, right_gini = self._find_min_gini_threshold(X_subset, y_subset, bin_num=bin_num)
        node = MiddleNode(path, label, gini, len(y_subset), min_gini_col_idx, min_gini_threshold)
        left_mask = X_subset.T[min_gini_col_idx] < min_gini_threshold
        left_node = self._build_tree(path + self.DELIMITER + self.LEFT, X_subset[left_mask], y_subset[left_mask], depth=depth+1, max_depth=max_depth, gini=left_gini, bin_num=bin_num)
        right_node = self._build_tree(path + self.DELIMITER + self.RIGHT , X_subset[~left_mask], y_subset[~left_mask], depth=depth+1, max_depth=max_depth, gini=right_gini, bin_num=bin_num)
        
        node.set_left(left_node)
        node.set_right(right_node)

        return node
    
    def _gini(self, y_subset: np.ndarray):
        n = len(y_subset)
        p_sum_square = 0
        for label in self.labels:
            p_sum_square += (sum(y_subset == label) / n)**2
        gini = 1 - p_sum_square
        return gini

    def _prune(self, ccp_alpha: float):
        pruned_leaf_n = 0
        while True:
            tree_dict = self.root.get_tree_dict()
            reversed_paths = sorted(tree_dict.keys(), key=lambda x: len(x), reverse=True)
            node_impurity_leaf_n_dict = {}
            min_alpha = np.inf
            min_alpha_path = None

            for path in reversed_paths:

                node = tree_dict[path]

                if node.is_leaf():
                    node_impurity_leaf_n_dict[path] = {"si": node.get_sample_n() * node.get_gini(), "leaf_n": 1}

                else:
                    alpha = (node.get_sample_n() * node.get_gini() - node_impurity_leaf_n_dict[path]["si"]) / (node_impurity_leaf_n_dict[path]["leaf_n"] - 1)

                    if alpha < min_alpha:
                        min_alpha = alpha
                        min_alpha_path = path

                if path == self.ROOT_CHAR:
                    break

                parent = node.get_parent()

                if parent.path not in node_impurity_leaf_n_dict.keys():
                    node_impurity_leaf_n_dict[parent.path] = {"si": 0, "leaf_n": 0}
                
                node_impurity_leaf_n_dict[parent.path]["si"] += node_impurity_leaf_n_dict[node.path]["si"]
                node_impurity_leaf_n_dict[parent.path]["leaf_n"] += node_impurity_leaf_n_dict[node.path]["leaf_n"]

            if min_alpha >= ccp_alpha:
                break

            pruned_leaf_n += node_impurity_leaf_n_dict[min_alpha_path]["leaf_n"] - 1
            target_node = tree_dict[min_alpha_path]

            label = target_node.get_label()
            gini = target_node.get_gini()
            sample_n = target_node.get_sample_n()

            leaf = LeafNode(min_alpha_path, label, gini, sample_n)

            if min_alpha_path == self.ROOT_CHAR:
                self.root = leaf
                return pruned_leaf_n
            
            parent = target_node.get_parent()

            if min_alpha_path[-1] == self.LEFT:
                parent.set_left(leaf)

            elif min_alpha_path[-1] == self.RIGHT:
                parent.set_right(leaf)

            else:
                raise exception.InvalidNodeSettingException(f"Nodeのパスが正しくありません. パスは'.', 'l', 'r', '/'からのみ構成されます. path: {min_alpha_path}")

        return pruned_leaf_n
            
        


if __name__ == "__main__":

    # 分類
    rng = np.random.default_rng(0)
    model = DecisionTreeClassifer(unit_nums=[2, 16, 16, 2], task="c")

    # 訓練データ作成(2値分類)
    data_n = 1000 # データの数
    r = 3 # 原点からの距離が3未満と3以上の点でクラスを分ける
    lim = (-5, 5) # データの範囲

    X_train = (lim[1] - lim[0]) * (rng.random((data_n, 2)) - 0.5) # x_1, x_2 \in [-5, 5]
    mask = np.sum(X_train ** 2, axis=1) < r**2
    y_train = mask.astype(int) # One-hotベクトル

    
    model = DecisionTreeClassifer(X_train, y_train)

    model.fit(bin_num=100, max_depth=100, ccp_alpha=1)

    x1 = np.linspace(-5, 5, 100)
    x2 = np.linspace(-5, 5, 100)
    xx1, xx2 = np.meshgrid(x1, x2)

    X_test = np.array([x1, x2]).T

    prediction = model.predict(X_test)
