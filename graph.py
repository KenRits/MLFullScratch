import networkx as nx
from matplotlib import pyplot as plt

class Graph(nx.DiGraph):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._indices_dict = {}
        self.node_generations = []

        # node_data
        # オブジェクトidと, オブジェクトのグラフ表示用の情報を対応付ける. 
        # key: object_id(int)
        # value: {"cls_name": Variable, "generation": 1, "idx": 0}
        self.node_data = {}

    def add_node(self, obj, **attr):
        # 既に追加されていれば無視.
        if self.node_data.get(id(obj)) is not None:
            return 
        # self.indices_dict ... {0: {"Variable": 0, "Square": 2, ... } 1: ...}
        gen_indices_dict = self._indices_dict.get(obj.generation)

        # 対応するgenerationのキーがself.indices_dictに登録されていなければ, 空の辞書を登録する
        if gen_indices_dict is None:
            gen_indices_dict = self._indices_dict[obj.generation] = {}

        # generationの, objの所属クラスで使用された最後のインデックスを取得し, 1を足したものを本objのインデックスとする.
        final_idx = gen_indices_dict.get(obj.__class__.__name__)

        if final_idx is None:
            node_idx = gen_indices_dict[obj.__class__.__name__] = 0
        else:
            node_idx = final_idx + 1

        # 最後に使われたidxを, 自分のidxに変更する.
        gen_indices_dict[obj.__class__.__name__] = node_idx

        self.node_data[id(obj)] = {"cls_name": obj.__class__.__name__, "generation": obj.generation, "idx": node_idx} 

        # グラフに表示する用のノード名を取得. ノード名はユニークである.
        disp_name = self._get_obj_disp_name(obj_id=id(obj))

        super().add_node(disp_name, **attr)

    def add_edge(self, obj_fr, obj_to, **attr):
        obj_fr_name = self._get_obj_disp_name(id(obj_fr))
        obj_to_name = self._get_obj_disp_name(id(obj_to))
        super().add_edge(obj_fr_name, obj_to_name, **attr)

    def _get_obj_disp_name(self, obj_id: int):
        """グラフに表示する用のノード名を取得. ノード名はユニークである.

        Args:
            obj_id (int): オブジェクトのid

        Returns:
            str: 表示用のユニークな名前.
        """
        assert self.node_data.get(obj_id) is not None, f"obj_id: {obj_id}はGraph.node_dataに登録されていません."
        obj_data = self.node_data[obj_id]
        return f"{obj_data["cls_name"]}{obj_data["idx"]}({obj_data["generation"]})"

    def show(self):

        pos = nx.spring_layout(self, seed=0)
        plt.figure(figsize=(10, 10))

        nx.draw_networkx(self, pos, node_color=[d["generation"] for d in self.node_data.values()], cmap=plt.cm.coolwarm, node_size=1000)
        plt.show()