
import random
import numpy as np

class creator():
    def change_format(self, sudo):
        sudo_generate = []
        for row_index, row in enumerate(sudo):
            row = list(map(str, row.tolist()))
            row = [element.replace('0', ' ') for element in row]
            sudo_generate.append(row)
        return sudo_generate

    def create_base_sudo(self):
        sudo = np.zeros((9, 9), dtype=int)
        num = random.randrange(9) + 1
        for row_index in range(9):
            for col_index in range(9):
                sudo_row = sudo[row_index, :]
                sudo_col = sudo[:, col_index]
                row_start = row_index // 3 * 3
                col_start = col_index // 3 * 3
                sudo_block = sudo[row_start: row_start + 3, col_start: col_start + 3]
                while (num in sudo_row) or (num in sudo_col) or (num in sudo_block):
                    num = num % 9 + 1
                sudo[row_index, col_index] = num
                num = num % 9 + 1
        return sudo

    def random_sudo(self):
        sudo = self.create_base_sudo()
        times = 50
        for _ in range(times):
            rand_row_base = random.randrange(3) * 3
            rand_rows = random.sample(range(3), 2)
            row_1 = rand_row_base + rand_rows[0]
            row_2 = rand_row_base + rand_rows[1]
            sudo[[row_1, row_2], :] = sudo[[row_2, row_1], :]
            rand_col_base = random.randrange(3) * 3
            rand_cols = random.sample(range(3), 2)
            col_1 = rand_col_base + rand_cols[0]
            col_2 = rand_col_base + rand_cols[1]
            sudo[:, [col_1, col_2]] = sudo[:, [col_2, col_1]]
        return(sudo)

    def get_sudo_subject(self, level):
        sudo = self.random_sudo()
        subject = sudo.copy()
        max_clear_count = 64
        min_clear_count = 14
        each_level_count = (max_clear_count - min_clear_count) // 5
        level_start = min_clear_count + (level - 1) * each_level_count
        del_nums = random.randrange(level_start, level_start + each_level_count)
        clears = random.sample(range(81), del_nums)
        for clear_index in clears:
            row_index = clear_index // 9
            col_index = clear_index % 9
            subject[row_index, col_index] = 0
        subject = self.change_format(subject)
        return subject
