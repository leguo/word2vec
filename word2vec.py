
import numpy as np
import heapq
from collections import Counter

class VocabItem(object):
  def __init__(self, token, count=0):
    self.token = token
    self.count = count
    self.code = None
    self.path = None
  def __repr__(self):
    return "%s:%d" %(self.token, self.count)

class Vocab(object):
  def __init__(self, f, min_count):
    counter = Counter()
    with open(f, 'r') as f:
      for line in f:
        tokens = line.split()
        counter.update(tokens)

    unk_count = 0
    removal = Counter()
    for (token, c) in counter.iteritems():
      if c < min_count:
        removal[token] = c
        unk_count += 1

    counter -= removal
    counter['<unk>'] = unk_count

    self.items = [VocabItem(token, c) for token, c in counter.most_common()]
    self.vocab_hash = dict([(item.token, i) for i,item in enumerate(self.items)])

    print 'vocab size: %d' % len(self.items)
    print 'top 10 words', self.items[:10]

  def encode_huffman(self):
    V = len(self.items)
    parent = [0] * V
    binary = [0] * V
    cnt_idx = [(x.count,i) for i, x in enumerate(self.items)]
    heapq.heapify(cnt_idx)
    new_idx = V
    while len(cnt_idx) > 1:
      (min1_cnt, min1) = heapq.heappop(cnt_idx)
      (min2_cnt, min2) = heapq.heappop(cnt_idx)
      binary[min2] = 1
      binary.append(0)
      parent[min1] = new_idx
      parent[min2] = new_idx
      parent.append(0)
      heapq.heappush(cnt_idx, (min1_cnt + min2_cnt, new_idx));
      new_idx += 1
    root_cnt, root = cnt_idx.pop()
    assert root == 2 * V - 2

    for i in range(V):
      code = [binary[i]]
      path = []
      parent_node = parent[i]
      while parent_node is not root:
        code.append(binary[parent_node])
        path.append(parent_node)
        parent_node = parent[parent_node]
      path.append(root)

      self.items[i].path = path[::-1]
      self.items[i].code = code[::-1]
      it = self.items[i]
