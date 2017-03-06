
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
    self.word_count = 0
    with open(f, 'r') as f:
      for line in f:
        tokens = line.split()
        counter.update(tokens)
        self.word_count += len(tokens)

    unk_count = 0
    removal = Counter()
    for (token, c) in counter.iteritems():
      if c < min_count:
        removal[token] = c
        unk_count += 1

    counter -= removal
    self.unk = '<unk>'
    counter[self.unk] = unk_count

    self.items = [VocabItem(token, c) for token, c in counter.most_common()]
    self.vocab_hash = dict([(item.token, i) for i,item in enumerate(self.items)])

    self.unk_hash = self.vocab_hash[self.unk]
    self.vocab_size = len(self.items)

    print 'vocab size: %d' % len(self.items)
    print 'top 10 words', self.items[:10]

  def indice(self, sent):
    return [self.vocab_hash.get(x, self.unk_hash) for x in sent]

  def encode_huffman(self):
    parent = [0] * self.vocab_size
    binary = [0] * self.vocab_size
    cnt_idx = [(x.count,i) for i, x in enumerate(self.items)]
    heapq.heapify(cnt_idx)
    new_idx = self.vocab_size
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
    assert root == 2 * self.vocab_size - 2

    for i in range(self.vocab_size):
      code = [binary[i]]
      path = []
      parent_node = parent[i]
      while parent_node is not root:
        code.append(binary[parent_node])
        path.append(parent_node)
        parent_node = parent[parent_node]
      path.append(root)

      self.items[i].path = [x - self.vocab_size for x in path[::-1]]
      self.items[i].code = code[::-1]
      it = self.items[i]

def sigmoid(x):
  return 1.0 / (1.0 + np.exp(-x))

class Word2Vec(object):
  def __init__(self):
    self.train_file = "test"
    
    self.vocab = Vocab(self.train_file, 5)
    self.vocab.encode_huffman()
    
    self.D = 100
    self.alpha = 0.025
    self.window = 5

    V = self.vocab.vocab_size
    self.syn0 = np.random.uniform(-0.5/V, 0.5/V, (V, self.D))
    self.syn1 = np.zeros((V, self.D))

  def _step(self, token, context):
    neu1 = np.mean([self.syn0[x] for x in context], axis=0) # (self.D,)
    dneu1 = np.zeros(self.D)
    loss = 0
    
    for j, label in zip(self.vocab.items[token].path, self.vocab.items[token].code):
      # forward
      z = np.sum(neu1 * self.syn1[j])
      p = sigmoid(z)
      loss_j = - ((1 - label) * np.log(p) + label * np.log(1-p))
      loss += loss_j
    
      # back prop
      dz = p + label - 1
      dneu1 += dz * self.syn1[j]
      dsyn1_j = dz * neu1
      self.syn1[j] -= self.alpha * dsyn1_j
    
    for i in context:
      dsyn0_i = dneu1
      self.syn0[i] -= self.alpha * dsyn0_i
  
    return loss

  def train(self):
    loss_history = np.zeros(1000)
    with open(self.train_file, "r") as f:
      for line in f:
        sent = self.vocab.indice(line.split())
        sent_len = len(sent)
        for pos, token in enumerate(sent):
          current_window = np.random.randint(1, self.window+1)
          left = max(pos - current_window, 0)
          right = min(pos + current_window + 1, sent_len)
          context = sent[left:pos] + sent[pos+1:right]
          loss = self._step(token, context)
          loss_history[pos % 1000] = loss
          if pos % 10000 == 0:
            print "%.2f%%" % (100.0 * pos / self.vocab.word_count), pos, np.mean(loss_history)

  def save(self, fo):
    with open(fo, 'w') as fo:
      fo.write('%d %d\n' % (len(self.syn0), self.D))
      for item, vector in zip(self.vocab.items, self.syn0):
        token = item.token
        vector_str = ' '.join([str(s) for s in vector])
        fo.write('%s %s\n' % (token, vector_str))

if '__main__' == __name__:
  model = Word2Vec()
  model.train()
  model.save("w2v.out")
