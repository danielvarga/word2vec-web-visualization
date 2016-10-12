import sys
import numpy as np

import glove


def intersection(wordList, gloveService, limit=10, wordWeights=None):
    if wordWeights is None:
        wordWeights = [ 1.0 for word in wordList ]
    print wordWeights
    weightedWords = zip(wordList, wordWeights)
    filteredWeightedWords = [ (word, weight) for (word, weight) in weightedWords if (word in gloveService.reWords) ]
    # Beware, we overwrite the wordList variable, not the original input anymore!
    wordList = [ word for (word, weight) in filteredWeightedWords ]
    indices = [ gloveService.reWords[word] for (word, weight) in filteredWeightedWords ]
    if len(weightedWords)!=len(filteredWeightedWords):
        sys.stderr.write("%d items dropped out of %d, not in glove dict.\n" % (len(weightedWords)-len(filteredWeightedWords), len(weightedWords)))
    if len(indices)==0:
            return []

    size = 10
    while True:
        inter = None
        for i in indices :
            neiSetForIndex = set(gloveService.annoyIndex.get_nns_by_item(i, size))
            if inter==None:
                inter = neiSetForIndex
            else:
                inter &= neiSetForIndex
            if len(inter)<limit:
                # We've lost the chance to reach the limit, early termination.
                break
        if len(inter)>=limit:
            break
        size = int(1.5*size)

    # The input words are forced to be in the output list.
    inter |= set(indices)

    candidates = np.array([ gloveService.model[gloveService.words[i]] for i in inter ]) # len(inter) x len(features)
    anchors = np.array([ gloveService.model[w] for w in wordList]) # len(wordList) x len(features)

    # Our goal is to find the candidates c with the minimal \sum_{a \in anchors} d(c,a) .
    dimOfSpace = len(gloveService.model[wordList[0]])
    candidatesL2S = np.sum(np.abs(candidates)**2,axis=-1)
    anchorsL2S = np.sum(np.abs(anchors)**2,axis=-1)

    # Repeats len(anchors) times the row that tells the L2S of each candidate:
    candidatesL2SM = np.tile(candidatesL2S, (len(anchors), 1)) # len(wordList) x len(inter)
    # Same, the other way round:
    anchorsL2SM = np.tile(anchorsL2S, (len(candidates), 1)) # len(inter) x len(wordList)

    # We combine these into a formula:
    # d^2(w_i,v_j) = <w_i|w_i> + <v_j|v_j> -2<w_i|v_j>
    # In matrix notation:
    # d^2(w_i,v_j) = candidatesL2SM + anchorsL2SM.T -2 candidates anchors.T
    squaredDistances = candidatesL2SM + anchorsL2SM.T - 2.0*anchors.dot(candidates.T) # len(wordList) x len(inter)
    distances = np.sqrt(squaredDistances+1e-6) # elementwise. +1e-6 is to supress sqrt-of-negative warning.
    summedDistances = np.sum(distances, axis=0) # len(inter) length vector

    zipped = zip(summedDistances.tolist(), [ gloveService.words[i] for i in inter ])
    # The input words are forced to be in the output list, at the top positions:
    inputWords = set(wordList)
    zipped.sort(key=lambda (distance, word) : (word not in inputWords, distance)) # Triple negation: words in inputWords go to the top.
    return [ w for (dist,w) in zipped[:limit] ]


def query(wordList, gloveService):
    wordIndices = [ gloveService.reWords.get(word, -1) for word in wordList ]
    if min(wordIndices)==-1:
            return []
    # wordVecs, unlike newVecs are numpy arrays.
    wordVecs =  [ np.array(gloveService.model[word]) for word in wordList ]
    assert len(wordVecs)==3
    a,b,c = wordVecs
    grid = [(-1,0), (0,-1), (1,1), (0.5,0), (0,0.5), (0.5,0.5)]
    newVecs = []
    for x,y in grid:
        d = a+(b-a)*float(x)+(c-a)*float(y)
        newVecs.append(d.tolist())
    words = []
    for d in newVecs:
        # Never, ever try to put a numpy array here, annoy silently corrupts memory
        # so that it crashes much later elsewhere.
        wordIndex = gloveService.annoyIndex.get_nns_by_vector(d, 1)[0]
        word = gloveService.words[wordIndex]
        words.append(word)
    assert gloveService.projection is not None, "Should set up global projection at startup time with --globalProjection switch."
    reduced = [ gloveService.projection[gloveService.reWords[word]] for word in words ]
    output = []
    for word, point, gridPoint in zip(words, reduced, grid):
        output.append((word, point[0], point[1], gridPoint[0], gridPoint[1]))
    return output


def main():
    gloveFile = sys.argv[1]
    wordList = sys.argv[2:5]
    gloveService = glove.GloveService(gloveFile, buildGlobalProjection=True)
    # output = query(wordList, gloveService)
    output = intersection(wordList, gloveService, limit=10)
    for item in output:
        # print "\t".join(map(str, item))
        print item


if __name__=="__main__":
    main()
