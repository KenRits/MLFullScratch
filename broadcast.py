def _get_broadcast_shape(shape1: tuple, shape2: tuple) -> tuple:
    """ブロードキャスト後の行列の形を返す. 

    Args:
        shape1 (tuple): 行列1の形状. 2次元のみ.
        shape2 (tuple): 行列2の形状. 2次元のみ.

    Returns:
        tuple: (行列1のブロードキャスト後のshape, 行列2のブロードキャスト後のshape)
        None: ブロードキャストできない場合はNoneが返る.
    """
    # shapeのi番目(i = 0, 1)が等しいか, そうでなければshape1[i]かshape2[i]のどちらかが1.
    # という条件がi=0, 1のどちらでも成り立つ場合, ブロードキャスト可能.

    out_shape1 = list(shape1)
    out_shape2 = list(shape2)
    for i in [0, 1]:
        if shape1[i] != shape2[i]:
            if shape1[i] != 1 and shape2[i] != 1:
                # ブロードキャスト不可能
                return None
            elif shape1[i] == 1 and shape2[i] != 1:
                # shape1[i]を引き延ばす.
                out_shape1[i] = shape2[i]
            
            elif shape1[i] != 1 and shape2[i] == 1:
                # shape2[i]を引き延ばす.
                out_shape2[i] = shape1[i]

            else:
                # shape1[i] = shape2[i] = 1
                pass
    
    return tuple(out_shape1), tuple(out_shape2)
