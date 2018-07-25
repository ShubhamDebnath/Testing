import numpy as np

class node:
  def __init__(self, x, y):
    self.g = 0
    self.h = 0
    self.x = x
    self.y = y
    self.parent = None

  def equals(self, target):
    if self.x == target.x and self.y == target.y:
      return True

  def node_print(self, end = '\n'):
    print('({0}, {1})'.format(self.x, self.y), end = end)

def get_children(start):
  children = []
  x, y = start.x, start.y
  for i in range(-1, 2):
    for j in range(-1, 2):
      child = node(x+i, y+j)
      children.append(child)

  return children

def is_valid(point, rows, cols, maze):
  x = point.x
  y = point.y
  if x > rows-1 or y > cols-1:
    # print('point over')
    return False

  if x<0 or y<0:
    # print('point under ', end = '')
    # point.node_print()
    return False

  if maze[x, y] == 1:
    # print('block found')
    return False

  return True

def eucleadian_distance(current, target):
  x1, y1 = current.x, current.y
  x2, y2 = target.x, target.y
  distance = ((x1 - x2)**2 + (y1 - y2)**2) ** 0.5
  return distance

def manhatten_distance(current, target):
  x1, y1 = current.x, current.y
  x2, y2 = target.x, target.y
  distance = (abs(x1 - x2) + abs(y1 - y2))
  return distance

def priority_queue(point, open_set):
  if len(open_set) < 1:
    open_set.add
  for i in range(len(open_set)):
    if open_set[i].f > point.f:
      open_set.insert(i, point)
  return open_set

def stack_trace(point):
  trace = []
  trace.append(point)
  while point.parent:
    trace.append(point.parent)
    point = point.parent

  return trace

def A_star(maze, start, target, rows, cols):
  open_set = set()
  open_set2 = set()
  closed_set = set()
  start.g = 0
  start.h = eucleadian_distance(start, target)
  current = start
  open_set.add((current.x, current.y))
  open_set2.add(current)

  while open_set:
    current = min(open_set2, key = lambda o: o.g + o.h)
    # print('selected min: ', end = '')
    # current.node_print()
    if current.equals(target):
      path = stack_trace(current)
      return path

    open_set.remove((current.x, current.y))
    open_set2.remove(current)
    closed_set.add((current.x, current.y))

    children = get_children(current)
    for child in children:
      if (child.x, child.y) in closed_set or not is_valid(child, rows, cols, maze):
        # print(is_valid(child, rows, cols, maze))
        # print('child in closed set', end = '')
        # child.node_print()
        continue

      if (child.x, child.y) in open_set:
        g = current.g + 1
        if child.g > g:
          child.g = g
          child.parent = current
          # print('child already in open set: ', end = '')
          # child.node_print()

      if (child.x, child.y) not in open_set:
        child.g = current.g + 1
        child.h = eucleadian_distance(child, target)
        child.parent = current
        open_set.add((child.x, child.y))
        open_set2.add(child)
        # print('child not in open set: ', end = '')
        # child.node_print()


if __name__ == '__main__':
  rows = 32
  cols = 24
  maze = np.random.randint(0, high = 2, size = (rows, cols), dtype = 'int')
  # maze = np.zeros((rows, cols))
  start = node(2, 2)
  target = node(20, 20)
  maze[20, 20] = 0
  maze[2, 2] = 0
  open_list = []
  open_list.append(start)
  closed_list = []
  trace = A_star(maze, start, target, rows, cols)
  for node in trace:
    node.node_print()