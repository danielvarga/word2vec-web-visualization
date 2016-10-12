## Webapp for visualizing word2vec/GloVe-based word clouds

### Installation

```
sudo pip install annoy numpy scipy scikit-learn
```

### Using the service

Click the below links, and click on some blue circles:
- English version: http://localhost:8080/vis/
- Hungarian version: http://localhost:8090/vis/

You can start from any given word:
- http://localhost:8080/vis/?q=token

There's an `lp=1` argument which overrides the server-side global SVD projection feature.
Instead of working with a global 2D embedding created at startup (see `--globalProjection`),
we create a 2D embedding just for the neighboring words:
- http://localhost:8080/vis/?q=token&lp=1&



### Glove data

The service needs data to work. For convenience, I've created a small dataset, you can grab it by running

```
wget http://oam2.us.prezi.com/~daniel/glove.840B.50k.300d.txt
```

from the backend directory, and skip the rest of this section.

The original Glove corpora can be downloaded from here: [`http://nlp.stanford.edu/projects/glove/`](http://nlp.stanford.edu/projects/glove/) , like this:

```wget http://www-nlp.stanford.edu/data/glove.42B.300d.txt.gz```

...or if you feel brave, you can go for the much larger

```wget http://www-nlp.stanford.edu/data/glove.840B.300d.txt.gz```

To speed up startup time and save memory, it's recommended to use only the first N lines
(the N most common words) of the corpus:

```
gunzip -c  glove.42B.300d.txt.gz | head -50000 >  glove.42B.50k.300d.txt
gunzip -c glove.840B.300d.txt.gz | head -50000 > glove.840B.50k.300d.txt
```

Ask takdavid@gmail.com for the Hungarian dataset.


### Running

```
Arguments:
  -h, --help           show this help message and exit
  --glove GLOVE        filename for glove data. If omitted, data loading is
                       skipped.
  --port PORT          port of service.
  --globalProjection   use a single SVD for the whole dataset instead of
                       building it from local data. Can be overridden with
                       /glove/?q=query&globalProjection=0
```

Example:
```
python server.py --glove glove.840B.50k.300d.txt --port 8080 --globalProjection
```

After starting the server, please wait until the ```Service has started.``` logline appears.
