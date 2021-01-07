import numpy as np
import traceback
import imp

class ChessCase:
    def __init__(self, chessboard_size):
        """
        params: chessboard should have the same number of balck and white piece
        params: rational_steps: rational locations the next piece should be put
        """
        self.chessboard = np.zeros((chessboard_size, chessboard_size), dtype=np.int)
        self.rational_steps = []

    def add_partial_board(self, board, offset):
        """
        params: board: the partial board in numpy array
        params: offset location of the left top piece of the particial board in the main board
        """
        shape = board.shape
        board[board == 2] = -1
        self.chessboard[
            offset[0] : offset[0] + shape[0], offset[1] : offset[1] + shape[1]
        ] = board

    @staticmethod
    def load_cases_files(filename):
        # filename='testcases.txt'
        case_list = []
        with open(filename) as fileobj:
            content = fileobj.read()
        usefulpart = content.split("###############")[1]
        all_cases = usefulpart.split("===============")
        for case in all_cases:
            new_case = ChessCase(8)
            subboards = case.split("---------------")
            for board in subboards:
                offset, arr_2d = ChessCase.__parse_array(board)
                if 3 in arr_2d:
                    rational_steps = np.argwhere(arr_2d == 3)
                    new_case.set_rational_steps(rational_steps + offset)
                else:
                    new_case.add_partial_board(arr_2d, offset)
            case_list.append(new_case)
        return case_list

    @staticmethod
    def __parse_array(text):
        lines = text.split()
        offset = (int(lines[0]), int(lines[1]))
        arr_list = []
        for line in lines[2:]:
            arr_list.append(np.array([int(char) for char in line]))
        arr_2d = np.stack(arr_list)
        return offset, arr_2d

    # print(offset, lines)

    def get_board(self):
        return self.chessboard

    def get_rational_steps(self):
        """
        -1 first
        """
        return self.rational_steps

    def set_rational_steps(self, steps):
        """
        -1 first
        """
        self.rational_steps = steps


class CodeCheck:
    def __init__(self, script_file_path, chessboard_size):
        self.time_out = 5
        self.script_file_path = script_file_path
        self.chessboard_size = chessboard_size
        self.agent = None
        self.errormsg = ""

    # Call this function and get True or False, self.errormsg has the massage
    def check_code(self):
        # check initialization
        advanced, passNum = self.__check_advance_chessboard()
        if not advanced:
            self.errormsg = (
                "Your code is too weak, fail to pass advance test, you passed only %d test cases"
                % (passNum)
                + self.errormsg
            )
            return False, passNum
        return True, passNum

    def time_out_init(self):
        self.agent = imp.load_source("AI", self.script_file_path).AI(
            self.chessboard_size, 1, self.time_out
        )
        self.agent = imp.load_source("AI", self.script_file_path).AI(
            self.chessboard_size, -1, self.time_out
        )

    def __check_go(self, chessboard,color=-1):
        self.agent = imp.load_source("AI", self.script_file_path).AI(
            self.chessboard_size, color, self.time_out
        )
        try:
            # self.agent.go(np.copy(chessboard))
            self.agent.go(np.copy(chessboard))
        except Exception:
            if len(self.agent.candidate_list) > 1000:
                self.errormsg = (
                    "Error: Time out and candidate list so large. The size of Candidate list should less then 1000"
                    + traceback.format_exc()
                )
            else:
                self.errormsg = traceback.format_exc()
            return False
        return True

    def __check_result(self, chessboard, result, color=-1):
        try:
            if not self.__check_go(chessboard, color=color):
                print("Timeout")
                return False
            if len(self.agent.candidate_list) > 1000:
                self.errormsg = (
                    "Error: Candidate list so large. The size of Candidate list should less then 1000"
                    + traceback.format_exc()
                )
                return False
            for i in self.agent.candidate_list:
                if i not in result:
                    print("User choice {}\nThe answer should be {}".format(self.agent.candidate_list, result.tolist() ))
                    return False
            if result is not None:
                for i in result:
                    if tuple(i) not in self.agent.candidate_list:
                        print("User choice {}\nThe answer should be {}".format(self.agent.candidate_list, result.tolist() ))
                        return False

                # if not self.agent.candidate_list or list(self.agent.candidate_list[-1]) not in result:

        except Exception:
            self.errormsg = traceback.format_exc()
            return False
        return True

    def __check_advance_chessboard(self):
        pass_num = 0
        size = 0
        case_list = ChessCase.load_cases_files("testcases.txt")
        print("======Test for black chess======")
        index = 0
        for case in case_list:
            if not self.__check_result(case.get_board(), case.get_rational_steps(),color=-1):
                print("Haven't pass the {}st case for black chess".format(index+1))
            else:
                pass_num += 1
            size += 1
            index += 1
        case_list = ChessCase.load_cases_files("testcases_white.txt")
        print("======Test for white chess======")
        index = 0
        for case in case_list:
            if not self.__check_result(case.get_board(), case.get_rational_steps(),color=1):
                print("Haven't pass the {}st case for white chess".format(index+1))
            else:
                pass_num += 1
            size += 1
            index += 1
        if pass_num == size:
            return True, pass_num
        else:
            return False, pass_num


if __name__ == "__main__":
    code_checker = CodeCheck("submit.py", 8) ## change the file name to your python file name
    success, passNum = code_checker.check_code()
    if not success:
        print(code_checker.errormsg)
    else:
        print("pass")
