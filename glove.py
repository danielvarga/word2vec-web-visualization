import annoy
import time
import sys
import random
import operator
import numpy as np
import sklearn.decomposition
import json

import multiWord


phase = None
startTime = None


def logg(*ss):
    s = " ".join(map(str,ss))
    sys.stderr.write(s+"\n")


def start(s):
    global startTime
    global phase
    phase = s
    logg(phase+".")
    startTime = time.clock()


def end(s=None):
    global startTime
    global phase
    if s is not None:
        phase = s
    endTime = time.clock()
    logg(phase,"finished in",endTime-startTime,"seconds.")


def read(f):
    m = {}
    words = []
    d = None
    errorCount = 0
    for l in f:
        a = l.strip("\n").split(" ")
        try:
            w = a[0].decode("utf-8")
        except:
            errorCount +=1
            continue
        words.append(w)
        v = map(float, a[1:])
        if d is not None:
            assert d==len(v)
        d = len(v)
        m[w] = v
    if errorCount>0:
        logg("glove datafile had unicode problems",errorCount,"lines dropped.")
    return m, words, d


def buildAnnoyIndex(model, words, dim):
    n = len(words)
    start("Building annoy index")
    annoyIndex = annoy.AnnoyIndex(dim)
    for i,w in enumerate(words):
        v = model[w]
        annoyIndex.add_item(i, v)
        if i>0 and (i-1)*10/n != i*10/n:
            logg("Annoy index building at", i*100/n, "percent.")
    end()
    start("Finalizing annoy index")
    annoyIndex.build(-1)
    end()
    return annoyIndex


def queryAnnoyIndex(model, annoyIndex, words, dim, n):
    badSign = 0
    start("Querying annoy index")
    for i,w in enumerate(words):
        v = model[w]
        neis = annoyIndex.get_nns_by_item(i,3)
        if i not in neis :
            badSign += 1
        print w, " ".join( words[i] for i in neis )
        if i>0 and (i-1)*10/n != i*10/n:
               logg("Annoy index querying at", i*100/n, "percent.")
    end()
    logg("Minimal quality evaluation:", float(badSign)/n)


def setupGloveService(gloveFile):
    start("Reading glove datafile")
    model, words, dim = read(file(gloveFile))
    end()
    annoyIndex = buildAnnoyIndex(model, words, dim)
    return model, words, annoyIndex, dim


class GloveService:
    # If loadStateFromSaveFile=True, then gloveFile should not be a filename.
    # Rather, a path to a .json and .ann annoy file, but without the extensions.
    # That is, GloveService("x/y", loadStateFromSaveFile=True) looks for "x/y.ann" and "x/y.json".
    # Such a pair of files can be created from a glove file using the testSerialization() function.
    def __init__(self, gloveFile, buildGlobalProjection=False, loadStateFromSaveFile=False):
        if loadStateFromSaveFile:
            self.load(gloveFile)
        else:
            self.model, self.words, self.annoyIndex, self.dim = setupGloveService(gloveFile)
        self.reWords = dict( (word,i) for (i,word) in enumerate(self.words) )
        self.projection = None
        if buildGlobalProjection:
            self.preProject()

    def save(self, baseFilename):
        start("Saving json")
        with open(baseFilename+'.json', 'w') as f:
            json.dump((self.model, self.words, self.dim), f)
        end()
        start("Saving ann")
        self.annoyIndex.save(baseFilename+'.ann')
        end()

    def load(self, baseFilename):
        start("Loading json")
        with open(baseFilename+'.json') as f:
            self.model, self.words, self.dim = json.load(f)
        end()
        start("Loading ann")
        self.annoyIndex = annoy.AnnoyIndex(self.dim)
        self.annoyIndex.load(baseFilename+'.ann')
        end()

    def preProject(self):
        matrix = np.array([ self.model[word] for word in self.words ])
        start("Calculating SVD for the full dataset")
        svd = sklearn.decomposition.TruncatedSVD(n_components=2, random_state=42)
        reduced = svd.fit_transform(matrix)
        end()
        self.projection = reduced

    def locallyProject(self, localWords):
        matrix = np.array([ self.model[localWord] for localWord in localWords ])
        svd = sklearn.decomposition.TruncatedSVD(n_components=2, random_state=42)
        reduced = svd.fit_transform(matrix)
        return reduced

    def embedCloud(self, localWords, useGlobalProjection=False):
        if len(localWords)==0:
            return np.empty((0,0))
        if useGlobalProjection:
            assert self.projection is not None, "Should set up global projection at startup time with --globalProjection switch."
            reduced = np.array([ self.projection[self.reWords[word]] for word in localWords ])
        else:
            reduced = self.locallyProject(localWords)
        return reduced

    # static method
    def sample(self, neis, limit):
        random.seed(1234)
        neisIndexed = sorted(random.sample(list(enumerate(neis)), limit))
        return map(operator.itemgetter(1), neisIndexed)

    def findNeighbors(self, word, limit=100, serendipity=0.0):
        wordIndex = self.reWords.get(word, -1)
        if wordIndex==-1:
            return []
        insideLimit = int(limit*(1.0+serendipity))
        neis = self.annoyIndex.get_nns_by_item(wordIndex, insideLimit)

        if insideLimit>limit:
            neis = self.sample(neis, limit)

        localWords = [ self.words[nei] for nei in neis ]
        return localWords

    def queryPure(self, wordOrWords, limit=100):
        justAWord = isinstance(wordOrWords, basestring)
        if not justAWord and len(wordOrWords)==0:
            return []
        if justAWord:
            word = wordOrWords
            localWords = self.findNeighbors(word, limit)
        else:
            words = wordOrWords
            localWords = multiWord.intersection(words, self, limit=limit)
        return localWords

    def query(self, wordOrWords, limit=100, useGlobalProjection=False):
        localWords = self.queryPure(wordOrWords, limit=limit)
        reduced = self.embedCloud(localWords, useGlobalProjection)
        return localWords, reduced

    def queryJson(self, wordOrWords, negativeWordOrWords=None, limit=100, useGlobalProjection=False):
        localWords, reduced = self.query(wordOrWords, limit, useGlobalProjection=useGlobalProjection)
        return cloudToJson(localWords, reduced)


def cloudToJson(localWords, reduced):
    assert len(localWords)==len(reduced)
    results = []
    for word, vec in zip(localWords, reduced):
        results.append( [ word, vec[0], vec[1] ] )
    return json.dumps({ "objects": results }, indent=4)


def testSerialization():
    gloveFile, savePath = sys.argv[1:]
    gloveService = GloveService(gloveFile)
    print gloveService.query("apple", limit=10)
    gloveService.save(savePath)
    gloveService.load(savePath)
    print gloveService.query("apple", limit=10)
    gloveService2 = GloveService(savePath, loadStateFromSaveFile=True)
    print gloveService2.query("apple", limit=10)


def testGlove(gloveService, words):
    js = gloveService.queryJson(words, limit=30, useGlobalProjection=False)
    print json.dumps(js, indent=4)


def main():
    gloveFile = sys.argv[1]
    words = sys.argv[2:]
    loadStateFromSaveFile = not gloveFile.endswith(".txt")
    gloveService = GloveService(gloveFile, loadStateFromSaveFile=loadStateFromSaveFile)

    testGlove(gloveService, words)


if __name__=="__main__":
    # testSerialization() ; sys.exit()
    main()
