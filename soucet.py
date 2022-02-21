# #programování 1

def line(p1, p2):
    A = (p1[1] - p2[1])
    B = (p2[0] - p1[0])
    C = (p1[0]*p2[1] - p2[0]*p1[1])
    return A, B, -C

def intersection(L1, L2):
    D  = L1[0] * L2[1] - L1[1] * L2[0]
    Dx = L1[2] * L2[1] - L1[1] * L2[2]
    Dy = L1[0] * L2[2] - L1[2] * L2[0]
    if D != 0:
        x = Dx / D
        y = Dy / D
        print("ANO")
        return x,y

    else:
        print("NE")
        return False

input = input().split() #save input to array

L1 = line([input[0],input[1]], [input[2],input[3]])
# L2 = line([input[4],input[5]], [input[6],input[7]])
#
# R = intersection(L1, L2)

def line_intersection(x1, y1, x2, y2, x3, y3, x4, y4):
    """find the intersection of line segments A=(x1,y1)/(x2,y2) and
    B=(x3,y3)/(x4,y4). Returns a point or None"""
    denom = ((x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4))
    if denom == 0:
        print("NE")
        return None
    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom
    if (px - x1) * (px - x2) < 0 and (py - y1) * (py - y2) < 0 \
      and (px - x3) * (px - x4) < 0 and (py - y3) * (py - y4) < 0:
        print("ANO")
        return [px, py]
    else:
        print("NE")

input = [1, 1, 1, 1, 1, 1, 1, 1]
line_intersection(input[0], input[1], input[2], input[3], input[4], input[5], input[6], input[7])




