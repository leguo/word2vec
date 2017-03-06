import numpy as np
import sys

emb = {}
pos = {}

def load_emb(emb_file):
  with open (emb_file, "r") as f:
    meta = f.readline().split()
    V, D = int(meta[0]), int(meta[1])
    print "V: ", V
    print "D: ", D
  
    i = 0
    for line in f:
      e = line.split()
      assert len(e) == D + 1
      word = e[0]
      emb[word] = np.array([float(x) for x in e[1:]])
      pos[word] = i
      i += 1

if __name__ == "__main__":
  print "Loading embedding:", sys.argv[1]
  load_emb(sys.argv[1])
  while True:
    target = raw_input("Enter word or sentence (EXIT to break):")
    if target == "EXIT":
      sys.exit(0)

    target_pos = pos.get(target, -1)
    print "Word: %s Position in vocabulary: %d" % (target, target_pos)
    if target_pos == -1:
      print "Out of dictionary word!"
      continue

    print "%50s%20s" % ("Word", "Cosine distance")
    print "-" * 72
    target_emb = emb[target]
    dist = []

    t = np.sqrt(target_emb.dot(target_emb))
    for word, e in emb.iteritems():
      if word == target:
        continue
      d = target_emb.dot(e) / t / np.sqrt(e.dot(e))
      dist.append((d, word))
    
    for (d, word) in sorted(dist, reverse=True)[:20]:
      print "%50s%20.6f" % (word, d)
