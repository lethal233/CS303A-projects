import numpy as np
from reversi import proj1 as reversi
import time

black = reversi.AI(8, -1, 5)
white = reversi.AI(8, 1, 5)

chessboard = np.zeros((8, 8))
chessboard[3, 3] = 1
chessboard[3, 4] = -1
chessboard[4, 3] = -1
chessboard[4, 4] = 1

# left = 24

chessboard = np.array(
    [[0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 1, 0, 0, 0, 0],
     [-1, -1, 1, 1, 1, -1, 0, 0],
     [0, 1, 1, 1, -1, 1, 1, 0],
     [0, 0, 1, -1, 1, 1, 0, 0],
     [-1, -1, -1, 1, -1, -1, 0, 0],
     [0, 0, 0, 0, 1, 0, 0, 0],
     [0, 0, 0, 0, -1, 1, 0, 0]])
t = time.time()
# print()
# print(white.go(chessboard), white.candidate_list)
print(black.go(chessboard), black.candidate_list)
# print(black.make_next_move(chessboard, 0, 7, -1))

# for x in range(0, 8):
#     for y in range(0, 8):
#         if chessboard[x, y] != 0:
#             print(black.stability_modify(chessboard, x, y, chessboard[x, y]), end=" ")
#         else:
#             print("NONE", end=" ")
#     print()
print(time.time() - t)

# while True:
#     new_c = chessboard.copy()
#     t = time.time()
#     black.go(new_c)
#     print(time.time() - t)
#     v_list = black.candidate_list.copy()
#     if len(v_list) == 0:
#         white.go(new_c)
#         v_list = white.candidate_list.copy()
#         if len(v_list) == 0:
#             break
#         else:
#             choose = v_list[-1]
#             reversi.AI.do_step(chessboard, white.color, choose)
#     else:
#         choose = v_list[-1]
#         reversi.AI.do_step(chessboard, black.color, choose)
#
#     left -= 1
#     print('left:', left)
#
#     new_c = chessboard.copy()
#     t = time.time()
#     white.go(new_c)
#     print(time.time() - t)
#     v_list = white.candidate_list.copy()
#     if len(v_list) == 0:
#         black.go(new_c)
#         v_list = black.candidate_list.copy()
#         if len(v_list) == 0:
#             break
#         else:
#             choose = v_list[-1]
#             reversi.AI.do_step(chessboard, black.color, choose)
#     else:
#         choose = v_list[-1]
#         reversi.AI.do_step(chessboard, white.color, choose)
#
#     left -= 1
#     print('left:', left)
#
# print(white.eval_final(chessboard))
