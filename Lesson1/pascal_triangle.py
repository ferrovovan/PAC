import argparse


parser = argparse.ArgumentParser(description='Blaise Pascal triangle script.')
parser.add_argument('height', type=int, help='Height of triangle')
args = parser.parse_args()

triangle_massiv = []


def add_row():
    global triangle_massiv

    if len(triangle_massiv) < 1:
        triangle_massiv.append( (1,) )
        return

    prev_row: tuple = triangle_massiv[-1]
    this_row: list  = [1]
    for idx in range(len(prev_row) - 1):
        this_row.append(prev_row[idx] + prev_row[idx + 1])
    this_row.append(1)
    
    this_row = tuple(this_row)
    triangle_massiv.append(this_row)


def print_triangle():
    widest_string =  " ".join(
                 list(map(str, triangle_massiv[-1]))
    )
    axis_num = int(len(widest_string) / 2)

    for idx in range(args.height):
        cur_row: tuple = triangle_massiv[idx]
        cur_string =  " ".join(
                 list(map(str, triangle_massiv[idx]))
        )
        cur_lenght_num = len(cur_string) // 2
      
        print(" " * (axis_num - cur_lenght_num) + cur_string)

for _ in range(args.height):
    add_row()

print_triangle()
