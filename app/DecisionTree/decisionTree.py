from model import Model

import numpy as np
class Node:
    def __init__(self, path: str, label: any, gini: float, sample_n: int):
        self.path = path # ルートノードからのパス. ルートの左子の右子ならば, pathは"./l/r"(".", "l", "r")といった文字列は, DecisionTreeクラスで変更できる.
        self._is_leaf = False
        self._left = None # Nodeインスタンス. 左子.
        self._right = None # 右子
        self._label = label # 葉ノードの場合のみ. 葉ノードの表すラベル(このノードに割り当てられた訓練データのラベルの多い方.)
        self._gini = gini # このノードに割り当てられた訓練データのジニ不純度.
        self._sample_n = sample_n # このノードに割り当てられた訓練データの数.
        self._parent = None # Nodeインスタンス. 親ノード.

    # getter
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

    # setter
    def set_parent(self, parent):
        self._parent = parent

class LeafNode(Node):
    def __init__(self, path: str, label: any, gini: float, sample_n: int):
        super().__init__(path, label, gini, sample_n)
        self._is_leaf = True

    def set_left(self, left: Node):
        raise ValueError("葉ノードに対して子ノードを設定することはできません.")
    
    def set_right(self, right: Node):
        raise ValueError("葉ノードに対して子ノードを設定することはできません.")

    def is_leaf(self):
        return self._is_leaf

    def get_tree_dict(self):
        return {self.path: self}
    
    def predict(self, x: np.ndarray) -> any:
        """特徴ベクトルxのラベルを予測する. 
        
        訓練時に登録されたラベルを返す.

        Args:
            x (np.ndarray): 特徴量. 1次元のnumpy配列.

        Raises:
            ValueError: ラベルが設定されていない場合に発火.

        Returns:
            any: 予測されたラベル.
        """
        if self._label is not None:
            return self._label
        else:
            raise ValueError("葉ノードにラベルが存在しません. ノードを作成する際は必ずNode.set_paramsを実行し, Node.predictを実行する前にラベルをはじめとした各種パラメータを設定してください.")

class MiddleNode(Node):
    def __init__(self, path: str, label: any, gini: float, sample_n: int, col_idx: int, threshold: float):
        super().__init__(path, label, gini, sample_n)
        self._col_idx = col_idx
        self._threshold = threshold
    
    def set_left(self, left: Node) -> None:
        self._left = left
        left.set_parent(self)

    def set_right(self, right: Node) -> None:
        self._right = right
        right.set_parent(self)

    def is_leaf(self) -> bool:
        return self._is_leaf

    def predict(self, x: np.ndarray) -> any:
        """あるxのラベルを予測する. 再帰関数となっており, 親ノードは分岐後の子ノードのpredict関数を使用する. 

        Args:
            x (np.ndarray): 特徴量. 1次元のnumpy配列.

        Returns:
            any: 予測されたラベル.
        """
        if x[self._col_idx] < self._threshold:
            prdicted_label = self._left.predict(x)
        else:
            prdicted_label = self._right.predict(x)
        
        return prdicted_label

    def get_tree_dict(self):
        """
        キーがpath(str), 値がNode(Node)の辞書を作成する.

        このノード以下の, {path: Nodeインスタンス ... } という辞書を作成する再帰関数.
        ルートでget_tree_dictを呼ぶと, 木全体の辞書を得ることができる.

        Returns:
            dict: パス(str)がキー, ノードインスタンス(Node)がvalueの辞書.
        """

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

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, max_depth: int =5, bin_num: int =256, ccp_alpha: float=None):
        """決定木を訓練する.

        Args:
            X_train (np.ndarray): 説明変数. すべての要素はint, floatなど大小比較ができるものを想定している.
            y_train (np.ndarray): 正解ラベル. len(X) == len(y)を満たす.
            max_depth (int, optional): 木の最大深さ. Defaults to 5.
            bin_num (int, optional): 閾値を決定する際に, 特徴を分けるビンの数. Defaults to 256.
            ccp_alpha (float, optional): 枝刈りのための閾値. 大きいほど多くの枝刈りが行われる. Defaults to None.
        """
        self.labels = np.unique(y_train)
        self.d = X_train.shape[1]

        root_gini = self._gini(y_train)
        self.root = self._build_tree(self.ROOT_CHAR, X_train, y_train, depth=0, max_depth=max_depth, gini=root_gini, bin_num=bin_num)

        if ccp_alpha is not None:
            pruned_node_count = self._prune(ccp_alpha)
            print(f"{pruned_node_count} leaves have pruned.")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """_summary_

        Args:
            X (np.ndarray): X.shape == (データ数, 特徴量の次元)の形のデータ.

        Raises:
            ValueError: 訓練する前に予測を行うと発生.

        Returns:
            np.ndarray: 予測されたラベル.
        """
        result = np.zeros(len(X))
        if self.root is not None:
            for i, x in enumerate(X):
                result[i] = self.root.predict(x)
            return result
        else:
            raise ValueError("まずDecisionTree.fitを実行してください.")
        
    def _find_min_gini_threshold(self, X_subset: np.ndarray, y_subset: np.ndarray, bin_num=256) -> tuple[int, np.float64, np.float64, np.float64]:
        """特徴量から, yを最もよく分ける特徴と, その閾値を算出する.

        Args:
            X_subset (np.ndarray): X_subset.shape == (データ数, 特徴量の次元) の行列. ただし「データ数」はミニバッチのデータ数を意味する.
            y_subset (np.ndarray): X_subsetに対応する, 正解ラベルの配列. len(X_subset) == len(y_subset) を要求.
            bin_num (int, optional): 特徴量を分ける際に作成されるヒストグラムのビンの数. 多いほど精度が上がるが, 計算量も大きくなる. Defaults to 256.

        Returns:
            tuple[int, np.float64, np.float64, np.float64]: 列のインデックス, 閾値, 分割後の左子のジニ不純度, 分割後の右子のジニ不純度
        """
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
    
    def _build_tree(self, path: str, X_subset: np.ndarray, y_subset: np.ndarray, depth: int, max_depth: int, gini: np.float64, bin_num: int =256, ) -> Node:
        """Nodeオブジェクトを作成する. 再帰関数.

        新たに作成されるNodeオブジェクトは, その子ノードを引っ提げて親ノードに返される.
        子ノードはこの関数の再帰呼び出しにより作成されて, それを新たに作成したノードの子ノードとして設定する.

        Args:
            path (str): ルートから, 次に分割する(=子ノードを作る)ノードまでのパス.
            X_subset (np.ndarray): 分割対象のノードに割り当てられた特徴量.
            y_subset (np.ndarray): 分割対象のノードに割り当てられた特徴量に対応する正解ラベル.
            depth (int): 分割対象のノードの, ルートからの距離.
            max_depth (int): 決定木の最大深さ.
            gini (np.float64): y_subsetのジニ不純度.
            bin_num (int, optional): 特徴量を分ける際に作成されるヒストグラムのビンの数. 多いほど精度が上がるが, 計算量も大きくなる. Defaults to 256.

        Raises:
            ValueError: X_subsetやy_subsetが空集合であるときに発生.

        Returns:
            Node: pathの位置に相当するノードオブジェクト.
        """
        if len(y_subset) == 0 or len(X_subset) == 0:
            raise ValueError(f"与えられたミニバッチの特徴量Xと対応する正解ラベルyのデータ数がそろっていません.\n len(X_subset): {len(X_subset)}, len(y_subset): {len(y_subset)}")

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
    
    def _gini(self, y_subset: np.ndarray) -> np.float64:
        """正解ラベルの集合からジニ不純度を計算する.

        Args:
            y_subset (np.ndarray): 正解ラベル(の部分集合)

        Returns:
            np.float64: ジニ不純度
        """
        n = len(y_subset)
        p_sum_square = 0
        for label in self.labels:
            p_sum_square += (np.sum(y_subset == label) / n)**2
        gini = 1 - p_sum_square
        return gini

    def _prune(self, ccp_alpha: float):
        """枝刈りを行う.

        Args:
            ccp_alpha (float): 枝刈りの程度を決定する. 大きいほど多くの枝が刈られる.

        Raises:
            ValueError: 
            self.DELIMITER, self.LEFT, self.RIGHT, self.ROOT_CHAR以外のパスに出会った場合に発生する.

        Returns:
            int: 刈られたノードの数.
        """
        pruned_leaf_n = 0

        while True:
            # キーがpath, 値がNodオブジェクトの辞書を作成.
            tree_dict = self.root.get_tree_dict()
            # pathの長さは, ノードの深さを表す. 葉ノード側から順にalphaを計算し, 親ノードに足していく.
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
                raise ValueError(f"Nodeのパスが正しくありません. パスは'{self.ROOT_CHAR}', '{self.LEFT}', '{self.RIGHT}', '{self.DELIMITER}'からのみ構成されます. path: {min_alpha_path}")

        return pruned_leaf_n
