## 第二阶段 74 名 第三阶段 28名

import numpy as np
import random
import time

COLOR_BLACK = -1
COLOR_WHITE = 1
COLOR_NONE = 0
random.seed(0)


## add in reverse number
class AI(object):
    def __init__(self, chessboard_size, color, time_out):
        self.chessboard_size = chessboard_size
        self.color = color
        self.time_out = time_out
        self.candidate_list = []
        self.BEST_POINT = -1e7
        self.WORST_POINT = 1e7
        self.weight_point = np.array([
            [990, -40, 300, 200, 200, 300, -40, 990],
            [-40, -400, 4, 2, 2, 4, -400, -40],
            [300, 4, 5, 1, 1, 5, 4, 300],
            [200, 2, 1, 0, 0, 1, 2, 200],
            [200, 2, 1, 0, 0, 1, 2, 200],
            [300, 4, 5, 1, 1, 5, 4, 300],
            [-40, -400, 4, 2, 2, 4, -400, -40],
            [990, -40, 300, 200, 200, 300, -40, 990]
        ])
        self.directions = [(1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1)]
        self.x_position = [(1, 1), (1, 6), (6, 1), (6, 6)]
        self.nw_se = [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6)]
        self.ne_sw = [(6, 1), (5, 2), (4, 3), (3, 4), (2, 5), (1, 6)]
        self.risky = [(0, 1), (1, 0), (1, 1), (0, 6), (1, 6), (1, 7), (6, 0), (6, 1), (7, 1), (6, 6), (6, 7), (7, 6)]
        self.c_position = [(0, 1), (0, 6), (1, 0), (1, 7), (6, 0), (6, 7), (7, 1), (7, 6)]
        self.corner_position = [(0, 0), (0, 7), (7, 0), (7, 7)]
        self.corner_side_position = [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7),
                                     (7, 0), (7, 1), (7, 2), (7, 3), (7, 4), (7, 5), (7, 6), (7, 7),
                                     (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0),
                                     (1, 7), (2, 7), (3, 7), (4, 7), (5, 7), (6, 7)]
        self.static_coef = 2
        self.mobility_coef = 50
        self.stability_coef = 8
        self.frontier_coef = 25
        self.difference_coef = 12
        self.MAX_DEPTH = 3

    def go(self, chessboard):
        self.BEST_POINT = -1e7
        self.WORST_POINT = 1e7
        self.candidate_list.clear()

        self.candidate_list = self.generate_moves(chessboard, self.color)

        rest = np.where(chessboard == COLOR_NONE)
        rest = list(zip(rest[0], rest[1]))
        rest_number = len(rest)

        if rest_number <= 8:
            self.MAX_DEPTH = 8
            self.static_coef = 2
            self.mobility_coef = 200
            self.stability_coef = 350
            self.frontier_coef = 90
            self.difference_coef = 0

            self.alpha_beta(chessboard, self.MAX_DEPTH, self.color, True, -1e7, 1e7)
            for i in self.corner_position:
                if i in self.candidate_list:
                    self.candidate_list.append(i)

        elif rest_number <= 13:
            self.static_coef = 1
            self.mobility_coef = 150
            self.stability_coef = 450
            self.frontier_coef = 100
            self.difference_coef = 0

            self.MAX_DEPTH = 4
            self.alpha_beta(chessboard, self.MAX_DEPTH, self.color, True, -1e7, 1e7)
            for i in self.corner_position:
                if i in self.candidate_list:
                    self.candidate_list.append(i)
        else:
            self.MAX_DEPTH = 3
            self.static_coef = 2.5
            self.mobility_coef = 230
            self.stability_coef = 80
            self.frontier_coef = 200
            self.difference_coef = 5

            self.alpha_beta(chessboard, self.MAX_DEPTH, self.color, True, -1e7, 1e7)
            for i in self.corner_position:
                if i in self.candidate_list:
                    self.candidate_list.append(i)

        return self.candidate_list

    def generate_moves(self, chessboard, player: int) -> list:
        idx = np.where(chessboard == COLOR_NONE)
        idx = list(zip(idx[0], idx[1]))
        moves_list = []
        risky_position_list = []
        for p in idx:
            if self.generate_legal_point(p, chessboard, player):
                if p not in self.risky:
                    moves_list.append(p)
                else:
                    risky_position_list.append(p)
        for x in risky_position_list:
            moves_list.append(x)
        return moves_list

    def generate_legal_point(self, point: tuple, chessboard, player: int) -> bool:
        self_color_index = 0  # index 0 marked as the no of self.color
        oppo_color_index = 1  # index 1 marked as the no of oppo.color
        tmp_x, tmp_y = point

        for direc in self.directions:
            if 0 <= (tmp_x + direc[0]) < self.chessboard_size and 0 <= (tmp_y + direc[1]) < self.chessboard_size and \
                    (chessboard[tmp_x + direc[0], tmp_y + direc[1]] == player or chessboard[
                        tmp_x + direc[0], tmp_y + direc[1]] == 0):
                continue
            num_of_each = [0, 0]
            incre_coef = 1
            while self.judge_within_board(tmp_x, tmp_y, incre_coef * direc[0], incre_coef * direc[1]):
                if chessboard[tmp_x + direc[0] * incre_coef, tmp_y + direc[1] * incre_coef] == COLOR_NONE:
                    break
                elif chessboard[tmp_x + direc[0] * incre_coef, tmp_y + direc[1] * incre_coef] == player:
                    if num_of_each[oppo_color_index] == 0:
                        break
                    else:
                        num_of_each[self_color_index] += 1
                        break
                else:
                    incre_coef += 1
                    num_of_each[oppo_color_index] += 1
            if num_of_each[oppo_color_index] != 0 and num_of_each[self_color_index] == 1:
                return True
        return False

    def judge_within_board(self, x: int, y: int, x_incre: int, y_incre: int) -> bool:
        if 0 <= x + x_incre < self.chessboard_size and 0 <= y + y_incre < self.chessboard_size:
            return True
        return False

    def evaluate(self, chessboard) -> float:
        # mobility
        length_of_self_moves = len(self.generate_moves(chessboard, self.color))
        length_of_oppo_moves = len(self.generate_moves(chessboard, -self.color))
        # if there is no way to go, then it must be end game
        if length_of_self_moves == 0 and length_of_oppo_moves == 0:
            evaluation = np.sum(chessboard) * self.color
            if evaluation > 0:
                return 1e6
            elif evaluation < 0:
                return -1e6
            else:
                return 0
        mobility = (length_of_self_moves - length_of_oppo_moves) * self.mobility_coef

        weight_point_cp = self.weight_point.copy()
        if chessboard[0, 0] != COLOR_NONE:
            if chessboard[0, 0] == chessboard[1, 0]:
                weight_point_cp[1, 0] = 400
            if chessboard[0, 0] == chessboard[0, 1]:
                weight_point_cp[0, 1] = 400
            if chessboard[1, 1] == chessboard[0, 0] == chessboard[1, 0] == chessboard[0, 1]:
                weight_point_cp[1, 1] = 100
        if chessboard[7, 0] != COLOR_NONE:
            if chessboard[7, 1] == chessboard[7, 0]:
                weight_point_cp[7, 1] = 400
            if chessboard[6, 0] == chessboard[7, 0]:
                weight_point_cp[6, 0] = 400
            if chessboard[6, 1] == chessboard[7, 1] == chessboard[7, 0] == chessboard[6, 0]:
                weight_point_cp[6, 1] = 100
        if chessboard[0, 7] != COLOR_NONE:
            if chessboard[0, 6] == chessboard[0, 7]:
                weight_point_cp[0, 6] = 400
            if chessboard[1, 7] == chessboard[0, 7]:
                weight_point_cp[1, 7] = 400
            if chessboard[1, 7] == chessboard[0, 7] == chessboard[0, 6] == chessboard[1, 6]:
                weight_point_cp[1, 6] = 100
        if chessboard[7, 7] != COLOR_NONE:
            if chessboard[6, 7] == chessboard[7, 7]:
                weight_point_cp[6, 7] = 400
            if chessboard[7, 6] == chessboard[7, 7]:
                weight_point_cp[7, 6] = 400
            if chessboard[6, 7] == chessboard[7, 7] == chessboard[6, 6] == chessboard[7, 6]:
                weight_point_cp[6, 6] = 100

        # difference
        self_number = 0
        oppo_number = 0

        # frontier
        self_frontier = 0
        oppo_frontier = 0

        # stable
        self_stable = 0
        oppo_stable = 0
        for x in range(self.chessboard_size):
            for y in range(self.chessboard_size):
                if chessboard[x, y] == COLOR_NONE:
                    continue
                elif chessboard[x, y] == self.color:
                    # difference
                    self_number += 1
                    # frontier
                    for direc in self.directions:
                        dx = x + direc[0]
                        dy = y + direc[1]
                        if 0 <= dx < self.chessboard_size and 0 <= dy < self.chessboard_size and chessboard[
                            dx, dy] == 0:
                            oppo_frontier += 1
                            break
                    if (x, y) in self.corner_side_position and self.stability_point(chessboard, x, y, self.color):
                        self_stable += 1
                else:
                    # difference
                    oppo_number += 1
                    # frontier
                    for direc in self.directions:
                        dx = x + direc[0]
                        dy = y + direc[1]
                        if 0 <= dx < self.chessboard_size and 0 <= dy < self.chessboard_size and chessboard[
                            dx, dy] == 0:
                            self_frontier += 1
                            break
                    if (x, y) in self.corner_side_position and self.stability_point(chessboard, x, y, -self.color):
                        oppo_stable += 1
        # frontier
        frontier = self.frontier_coef * (self_frontier - oppo_frontier)

        # static
        static_weight = np.sum(weight_point_cp * chessboard) * self.color * self.static_coef

        # number difference
        number_difference = (-self_number + oppo_number) * self.difference_coef
        # stable
        stable = (self_stable - oppo_stable) * self.stability_coef

        evaluation = mobility + frontier + static_weight + number_difference + stable
        return evaluation

    def make_next_move(self, chessboard, x: int, y: int, player: int):
        chessboard[x, y] = player
        for direc in self.directions:
            ctr = 0
            for i in range(self.chessboard_size):
                dx = x + direc[0] * (i + 1)
                dy = y + direc[1] * (i + 1)
                if dx < 0 or dx >= self.chessboard_size or dy < 0 or dy >= self.chessboard_size:
                    ctr = 0
                    break
                elif chessboard[dx, dy] == player:
                    break
                elif chessboard[dx, dy] == COLOR_NONE:
                    ctr = 0
                    break
                else:
                    ctr += 1
            for j in range(ctr):
                ddx = x + direc[0] * (j + 1)
                ddy = y + direc[1] * (j + 1)
                chessboard[ddx, ddy] = player
        return chessboard

    def alpha_beta(self, chessboard, depth: int, player: int, maximizing: bool, alpha: float, beta: float) -> float:
        move_list = self.generate_moves(chessboard, player)
        if depth == 0:
            return self.evaluate(chessboard)
        if not move_list:
            return self.alpha_beta(chessboard, depth - 1, -player, ~maximizing, alpha, beta)
        if maximizing:
            init_value = self.BEST_POINT
            for move in move_list:
                chessboard_tmp = self.make_next_move(chessboard.copy(), move[0], move[1], player)
                value_tmp = self.alpha_beta(chessboard_tmp, depth - 1, -player, False, alpha, beta)
                if depth == self.MAX_DEPTH:
                    if move in self.c_position:
                        if self.do_lose_corner(chessboard, move):
                            value_tmp -= 2000
                    # special judgement for star position
                    if move in self.x_position:
                        if (move == (1, 1) or move == (6, 6)) and chessboard[0, 0] != self.color and chessboard[
                            7, 7] != self.color:
                            for diag in self.nw_se:
                                if diag != move and chessboard_tmp[diag] != self.color:
                                    # print(move, diag)
                                    value_tmp -= 2800
                                    break
                        if (move == (1, 6) or move == (6, 1)) and chessboard[0, 7] != self.color and chessboard[
                            7, 0] != self.color:
                            for diag in self.ne_sw:
                                if diag != move and chessboard_tmp[diag] != self.color:
                                    value_tmp -= 2800
                                    break

                    print(move, value_tmp)

                    if value_tmp > init_value:
                        init_value = value_tmp
                        self.candidate_list.append(move)
                else:
                    init_value = max(init_value, value_tmp)
                    alpha = max(alpha, init_value)
                if alpha >= beta:
                    break
            return init_value
        else:
            init_value = self.WORST_POINT
            for move in move_list:
                chessboard_tmp = self.make_next_move(chessboard.copy(), move[0], move[1], player)
                value_tmp = self.alpha_beta(chessboard_tmp, depth - 1, -player, True, alpha, beta)
                init_value = min(init_value, value_tmp)
                beta = min(beta, init_value)
                if alpha >= beta:
                    break
            return init_value

    def stability_point(self, chessboard, x: int, y: int, player: int) -> bool:
        four_direction = [(1, 0), (1, 1), (0, 1), (-1, 1)]
        num_stable_directions = 0
        for fd in four_direction:
            has_blank = False
            has_oppo = False
            increment = 1
            while True:
                dx = x + fd[0] * increment
                dy = y + fd[1] * increment
                if 0 <= dx < 8 and 0 <= dy < 8:
                    if chessboard[dx, dy] == player:
                        pass
                    elif chessboard[dx, dy] == -player:
                        has_oppo = True
                    else:
                        has_blank = True
                        break
                    increment += 1
                else:
                    break

            if not has_oppo and not has_blank:
                num_stable_directions += 1
            elif has_oppo and not has_blank:
                increment = -1
                reverse_oppo_flag = False
                while True:
                    dx = x + fd[0] * increment
                    dy = y + fd[1] * increment
                    if 0 <= dx < 8 and 0 <= dy < 8:
                        if chessboard[dx, dy] == -player:
                            reverse_oppo_flag = True
                        elif chessboard[dx, dy] == COLOR_NONE:
                            if not reverse_oppo_flag:
                                return False
                            else:
                                break
                        else:
                            pass
                        increment -= 1
                    else:
                        break
                num_stable_directions += 1
            elif has_oppo and has_blank:
                increment = -1
                oppo_reverse = False
                while True:
                    dx = x + fd[0] * increment
                    dy = y + fd[1] * increment
                    if 0 <= dx < 8 and 0 <= dy < 8:
                        if chessboard[dx, dy] == COLOR_NONE:
                            if not oppo_reverse:
                                return False
                            else:
                                break
                        elif chessboard[dx, dy] == player:
                            pass
                        else:
                            oppo_reverse = True
                        increment -= 1
                    else:
                        break
                num_stable_directions += 1
            else:  # not has_oppo and has_blank
                increment = -1
                while True:
                    dx = x + fd[0] * increment
                    dy = y + fd[1] * increment
                    if 0 <= dx < 8 and 0 <= dy < 8:
                        if chessboard[dx, dy] != player:
                            return False
                        increment -= 1
                    else:
                        break
                num_stable_directions += 1
        if num_stable_directions == 4:
            return True
        else:
            return False

    def do_lose_corner(self, chessboard, move: tuple) -> bool:
        """
        :param chessboard:
        :param move: tuple(x,y)
        :return: bool
        """
        nw = [(0, 1), (1, 0)]
        ne = [(0, 6), (1, 7)]
        sw = [(6, 0), (7, 1)]
        se = [(6, 7), (7, 6)]
        if move in nw:
            if chessboard[0, 0] == self.color:
                return False
            elif chessboard[0, 0] == -self.color:
                dx = 2 * move[0] - 0
                dy = 2 * move[1] - 0
                if chessboard[dx, dy] == -self.color:
                    return False
                else:
                    return True
            else:
                x_incre = move[0] - 0
                y_incre = move[1] - 0
                dx = move[0]
                dy = move[1]
                self_counter = 0
                blank_counter = 0
                while True:
                    dx += x_incre
                    dy += y_incre
                    if 0 <= dx < 8 and 0 <= dy < 8:
                        if chessboard[dx, dy] == self.color:
                            self_counter += 1
                            if blank_counter % 2 == 1:
                                return True
                            else:
                                blank_counter = 0
                        elif chessboard[dx, dy] == -self.color:
                            return True
                        else:
                            blank_counter += 1
                    else:
                        break
                return False

        elif move in ne:
            if chessboard[0, 7] == self.color:
                return False
            elif chessboard[0, 7] == -self.color:
                dx = 2 * move[0] - 0
                dy = 2 * move[1] - 7
                if chessboard[dx, dy] == -self.color:
                    return False
                else:
                    return True
            else:
                x_incre = move[0] - 0
                y_incre = move[1] - 7
                dx = move[0]
                dy = move[1]
                self_counter = 0
                blank_counter = 0
                while True:
                    dx += x_incre
                    dy += y_incre
                    if 0 <= dx < 8 and 0 <= dy < 8:
                        if chessboard[dx, dy] == self.color:
                            self_counter += 1
                            if blank_counter % 2 == 1:
                                return True
                            else:
                                blank_counter = 0
                        elif chessboard[dx, dy] == -self.color:
                            return True
                        else:
                            blank_counter += 1
                    else:
                        break
                return False
        elif move in sw:
            if chessboard[7, 0] == self.color:
                return False
            elif chessboard[7, 0] == -self.color:
                dx = 2 * move[0] - 7
                dy = 2 * move[1] - 0
                if chessboard[dx, dy] == -self.color:
                    return False
                else:
                    return True
            else:
                x_incre = move[0] - 7
                y_incre = move[1] - 0
                dx = move[0]
                dy = move[1]
                self_counter = 0
                blank_counter = 0
                while True:
                    dx += x_incre
                    dy += y_incre
                    if 0 <= dx < 8 and 0 <= dy < 8:
                        if chessboard[dx, dy] == self.color:
                            self_counter += 1
                            if blank_counter % 2 == 1:
                                return True
                            else:
                                blank_counter = 0
                        elif chessboard[dx, dy] == -self.color:
                            return True
                        else:
                            blank_counter += 1
                    else:
                        break
                return False
        else:
            if chessboard[7, 7] == self.color:
                return False
            elif chessboard[7, 7] == -self.color:
                dx = 2 * move[0] - 7
                dy = 2 * move[1] - 7
                if chessboard[dx, dy] == -self.color:
                    return False
                else:
                    return True
            else:
                x_incre = move[0] - 7
                y_incre = move[1] - 7
                dx = move[0]
                dy = move[1]
                self_counter = 0
                blank_counter = 0
                while True:
                    dx += x_incre
                    dy += y_incre
                    if 0 <= dx < 8 and 0 <= dy < 8:
                        if chessboard[dx, dy] == self.color:
                            self_counter += 1
                            if blank_counter % 2 == 1:
                                return True
                            else:
                                blank_counter = 0
                        elif chessboard[dx, dy] == -self.color:
                            return True
                        else:
                            blank_counter += 1
                    else:
                        break
                return False
