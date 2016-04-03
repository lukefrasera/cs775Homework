#!/usr/bin/env python
import csv, sys

class CSVInput:
    def __init__(self, filename, first_row_titles=False, num_convert=True, set_true_false_01=True):
        self.titles = []
        self.data = []
        self.boolean_false = ['F', 'f', 'False', 'FALSE', 'false']
        self.boolean_true  = ['T', 't', 'True', 'TRUE', 'true']
        with open(filename, 'rb') as file:
            reader = csv.reader(file, delimiter='\t')
            for i, row in enumerate(reader):
                if i==0 and first_row_titles:
                    self.titles += row
                else:
                    if num_convert:
                        row_list = []
                        for elem in row:
                            try:
                                value = float(elem)
                            except ValueError:
                                try:
                                    value = int(elem)
                                except ValueError:
                                    value = elem
                                    if any(false in value for false in self.boolean_false):
                                      value = 0
                                    elif any(true in value for true in self.boolean_true):
                                      value = 1
                            row_list.append(value)
                    self.data.append(row_list)


def main():
    # Read Data in and convert numbers to numbers
    reader = CSVInput(sys.argv[1], first_row_titles=True)
    print reader.titles
    # print reader.data

if __name__ == '__main__':
    main()
