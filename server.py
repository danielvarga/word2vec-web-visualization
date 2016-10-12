import sys
import BaseHTTPServer
import SimpleHTTPServer
import urlparse
import json
import argparse

import glove


def getQuery(queryParsed):
    try:
        return queryParsed['q'][0].decode("utf-8")
    except:
        return None

def matches(command, pathList):
    return any( command==path or command.startswith(path+"/") for path in pathList )

def getFlag(queryParsed, arg):
    return arg in queryParsed and queryParsed[arg][0]=='1'


class MyRequestHandler(SimpleHTTPServer.SimpleHTTPRequestHandler):

    # content is already serialized!
    def sendContent(self, content, format="json", status=200):
        self.send_response(status)
        self.send_header('Content-Type', 'text/'+format+'; charset=utf-8')
        self.end_headers()

        self.wfile.write(content)
        self.wfile.close()

    def do_GET(self):
      try:
        parsedParams = urlparse.urlparse(self.path)
        queryParsed = urlparse.parse_qs(parsedParams.query)
        query = getQuery(queryParsed)

        status = 200

        command = parsedParams.path.strip("/")
        if command=="glove":
            assert g_glove is not None, "glove dataset not loaded, please use the --glove argument"
            limit = int( queryParsed.get('limit', [100])[0])

            # By default, we use global projection if and only if this was asked for at startup time.
            useGlobalProjection = g_glove.projection is not None
            if 'globalProjection' in queryParsed:
                askedForGlobalProjection = queryParsed['globalProjection'][0]=='1'
                assert not( askedForGlobalProjection and not useGlobalProjection ), "Global projection has not been set up."
                useGlobalProjection = askedForGlobalProjection

            keywords = query.split(" ")

            if len(keywords)>1:
                wordOrWords = keywords
            else:
                wordOrWords = keywords[0]

            jsonResult = g_glove.queryJson(wordOrWords,
                            limit=limit, useGlobalProjection=useGlobalProjection)

        elif matches(command, ("vis",)):
            return SimpleHTTPServer.SimpleHTTPRequestHandler.do_GET(self)

        else:
            self.sendContent("unknown service: "+command, status=400)
            return

        self.sendContent(jsonResult, status=status)
      except:
        sys.stderr.write("Exception catched and re-raised for request "+str(parsedParams)+" aka "+str(self.path)+"\n")
        raise

    # Allow XSS:
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        SimpleHTTPServer.SimpleHTTPRequestHandler.end_headers(self)

g_glove = None

def main():
    global g_glove
    parser = argparse.ArgumentParser(description='Backend service for word2vec visualizations.')
    parser.add_argument('--glove', type=str, help='''filename of word embedding data file or serialized GloveService.
If the filename ends with .txt, it's interpreted as word embedding data file, otherwise two files are looked for,
filename.ann and filename.json. This argument is mandatory.''')
    parser.add_argument('--port',  type=int, default=8080, help='port of service.')
    parser.add_argument('--globalProjection', action='store_true',
            help='''use a single SVD for the whole dataset instead of always building it from local data.
Can be overridden with /glove/?q=query&globalProjection=0''')
    args = parser.parse_args()

    assert args.glove is not None, "Please provide a --glove argument."
    loadStateFromSaveFile = not args.glove.endswith(".txt")
    g_glove = glove.GloveService(args.glove, buildGlobalProjection=args.globalProjection, loadStateFromSaveFile=loadStateFromSaveFile)

    server_address = ('0.0.0.0', args.port)

    server = BaseHTTPServer.HTTPServer(server_address, MyRequestHandler)
    sys.stderr.write("Service has started.\n")
    server.serve_forever()

if __name__ == "__main__":
    main()
