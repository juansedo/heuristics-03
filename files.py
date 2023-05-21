import os
import itertools
from xlwt import Workbook

class ExcelBook:
    output_path = "./results/"

    def __init__(self, title):
        self.title = title
        self.wb = Workbook()

    def get_sheet_by_name(self, name):
        try:
            for idx in itertools.count():
                sheet = self.wb.get_sheet(idx)
                if sheet.name == name:
                    return sheet
        except IndexError:
            return self.wb.add_sheet(name)

    def add_sheet(self, index, problem_result, Th):
        sheet1 = self.get_sheet_by_name("mtVRP" + str(index))
        paths, distances, total_time = problem_result
        R = len(paths)
        for i in range(0, R):
            size = len(paths[i])
            for j in range(size):
                sheet1.write(i, j, paths[i][j])
            sheet1.write(i, size, round(distances[i], 2))
            sheet1.write(i, size + 1, 1 if distances[i] > Th else 0)

        sheet1.write(R, 0, round(sum(distances), 2))
        sheet1.write(R, 1, round(total_time, 2))
        sheet1.write(R, 2, 1)

    def save(self):
        if not os.path.exists(ExcelBook.output_path):
            os.makedirs(ExcelBook.output_path)
        self.wb.save(ExcelBook.output_path + self.title)
        print(f"{self.title} saved!")
