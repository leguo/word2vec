
import numpy as np
import heapq
import math
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

  def tokens(self, indice):
    if type(indice) == list:
      return [self.items[x].token for x in indice]
    return self.items[indice]

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

class UnigramTable(object):
  def __init__(self, vocab):

    self.table_size = int(1e8)
    self.table = np.zeros(self.table_size, dtype=np.uint32)
    power = 0.75

    norm = sum([math.pow(x.count, power) for x in vocab.items])
    vocab_size = len(vocab.items)

    token_idx = 0
    cumu_prob = pow(vocab.items[token_idx].count, power) / norm
    
    for i in range(self.table_size):
      self.table[i] = token_idx
      if 1.0 * i / self.table_size > cumu_prob:
        token_idx += 1
        cumu_prob += math.pow(vocab.items[token_idx].count, power) / norm
      if token_idx >= vocab_size:
        token_idx = vocab_size - 1

  def sample(self, count):
    indices = np.random.randint(low=0, high=self.table_size, size=count)
    return [self.table[x] for x in indices]

def sigmoid(x):
  return 1.0 / (1.0 + np.exp(-x))

class Word2Vec(object):
  def __init__(self, train_file="text8", cbow=True, hs=True):
    self.train_file = train_file
    self.cbow = cbow
    self.hs = hs
    self.negative = 5
    
    print "Initializing vocab ..."
    self.vocab = Vocab(self.train_file, 5)
    self.vocab.encode_huffman()
    
    if not hs:
      print "Initializing unigram table ..."
      self.table = UnigramTable(self.vocab)

    self.D = 100
    self.starting_alpha = 0.025
    self.alpha = self.starting_alpha
    self.window = 5

    V = self.vocab.vocab_size
    self.syn0 = np.random.uniform(-0.5/V, 0.5/V, (V, self.D))
    self.syn1 = np.zeros((V, self.D))

  def _step(self, token, context):
    loss = 0
    
    if self.cbow:
      neu1 = np.mean([self.syn0[x] for x in context], axis=0) # (self.D,)
      dneu1 = np.zeros(self.D)
      if self.hs:
        for j, label in zip(self.vocab.items[token].path, self.vocab.items[token].code):
          # forward
          z = np.sum(neu1 * self.syn1[j])
          p = sigmoid(z)
          #loss_j = - ((1 - label) * np.log(p) + label * np.log(1-p))
          #loss += loss_j
          # back prop
          dz = p + label - 1
          dneu1 += dz * self.syn1[j]
          dsyn1_j = dz * neu1
          self.syn1[j] -= self.alpha * dsyn1_j
        
      else:
        neg = self.table.sample(self.negative)
        for j, label in [(token, 1)] + [(x, 0) for x in neg if x != token]:
          # forward
          z = np.sum(neu1 * self.syn1[j])
          p = sigmoid(z)
          # loss_j = - (label*np.log(p) + (1-label)*np.log(1-p))
          # back prop
          dz = p - label
          dneu1 += dz * self.syn1[j]
          dsyn1_j = dz * neu1
          self.syn1[j] -= self.alpha * dsyn1_j

      for i in context:
        dsyn0_i = dneu1
        self.syn0[i] -= self.alpha * dsyn0_i

    # skip-gram
    else:
      for c in context:
        neu1 = self.syn0[c] # (self.D,)
        dneu1 = np.zeros(self.D)

        # hierarchical softmax
        if self.hs:
          for j, label in zip(self.vocab.items[token].path, self.vocab.items[token].code):
            z = np.sum(neu1 * self.syn1[j])
            p = sigmoid(z)
            dz = p + label - 1
            dneu1 += dz * self.syn1[j]
            self.syn1[j] -= self.alpha * (dz * neu1)

        # negative sampling
        else:
          neg = self.table.sample(self.negative)
          for j, label in [(token, 1)] + [(x, 0) for x in neg if x != token]:
            # propagate hidden -> output
            z = np.sum(neu1 * self.syn1[j])
            p = sigmoid(z)
            # loss_j = - (label*np.log(p) + (1-label)*np.log(1-p))
            # back prop
            dz = p - label
            dneu1 += dz * self.syn1[j]
            # learn weights hidden -> output
            self.syn1[j] -= self.alpha * (dz * neu1)

        # learn weights input -> hidden
        self.syn0[c] -= self.alpha * dneu1
  
    return loss

  def train(self):
    loss_history = np.zeros(1000)
    count = 0
    with open(self.train_file, "r") as f:
      for line in f:
        sent = self.vocab.indice(line.split())
        sent_len = len(sent)
        for pos, token in enumerate(sent):
          count += 1
          if count % 10000 == 0:
            self.alpha = max(self.starting_alpha * (1 - float(count) / self.vocab.word_count),
                             self.starting_alpha * 0.0001)
            print "\rAlpha: %f Progress: %d of %d (%.2f%%) Loss:%f" % (
              self.alpha,
              count,
              self.vocab.word_count,
              100.0 * count / self.vocab.word_count,
              np.mean(loss_history)
            )

          current_window = np.random.randint(1, self.window+1)
          left = max(pos - current_window, 0)
          right = min(pos + current_window + 1, sent_len)
          context = sent[left:pos] + sent[pos+1:right]
          loss = self._step(token, context)
          loss_history[count % 1000] = loss

  def save(self, fo):
    with open(fo, 'w') as fo:
      fo.write('%d %d\n' % (len(self.syn0), self.D))
      for item, vector in zip(self.vocab.items, self.syn0):
        token = item.token
        vector_str = ' '.join([str(s) for s in vector])
        fo.write('%s %s\n' % (token, vector_str))

if '__main__' == __name__:
  model = Word2Vec(train_file="text8", cbow=False, hs=False)
  model.train()
  model.save("skipgram-ns.vector")
