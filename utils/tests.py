
def test_update_beam_table():
    import utils
    import numpy as np
    B = 4
    L = 8

    beam_idx = np.zeros([B, L], dtype=np.int32)
    beam_tab = np.zeros([B, L], dtype=np.int32)

    beam_idx=np.array([[0, 0, 0, 0, 0, 0, 0, 0],
                       [1, 1, 1, 3, 1, 0, 0, 0],
                       [2, 2, 2, 2, 0, 0, 0, 0],
                       [3, 3, 3, 2, 2, 0, 0, 0]], dtype=np.int32)
    utils.update_beam_table(beam_idx, beam_tab, 5)

    print(beam_idx)
    print(beam_tab)
