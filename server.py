import sys
import http.server
import urllib.parse
import json
import argparse
import traceback

import glove


def getQuery(queryParsed):
    try:
        return queryParsed['q'][0]
    except Exception:
        return None

def matches(command, pathList):
    return any(command == path or command.startswith(path + "/") for path in pathList)

def getFlag(queryParsed, arg):
    return arg in queryParsed and queryParsed[arg][0] == '1'

class MyRequestHandler(http.server.SimpleHTTPRequestHandler):

    def sendContent(self, content, format="json", status=200):
        self.send_response(status)
        self.send_header('Content-Type', 'text/' + format + '; charset=utf-8')
        self.end_headers()

        if isinstance(content, str):
            content = content.encode("utf-8")
        self.wfile.write(content)

    def do_GET(self):
        try:
            parsedParams = urllib.parse.urlparse(self.path)
            queryParsed = urllib.parse.parse_qs(parsedParams.query)
            query = getQuery(queryParsed)

            status = 200
            command = parsedParams.path.strip("/")

            if command == "glove":
                assert g_glove is not None, "glove dataset not loaded, please use the --glove argument"
                limit = int(queryParsed.get('limit', [100])[0])

                useGlobalProjection = g_glove.projection is not None
                if 'globalProjection' in queryParsed:
                    askedForGlobalProjection = queryParsed['globalProjection'][0] == '1'
                    assert not (askedForGlobalProjection and not useGlobalProjection), "Global projection has not been set up."
                    useGlobalProjection = askedForGlobalProjection

                keywords = query.split(" ") if query else []
                wordOrWords = keywords if len(keywords) > 1 else keywords[0] if keywords else ""

                jsonResult = g_glove.queryJson(wordOrWords, limit=limit, useGlobalProjection=useGlobalProjection)
                self.sendContent(jsonResult, status=status)

            elif matches(command, ("vis",)):
                return super().do_GET()

            else:
                self.sendContent("unknown service: " + command, status=400)

        except Exception:
            sys.stderr.write("Exception caught for request {} aka {}\n".format(str(parsedParams), str(self.path)))
            traceback.print_exc()
            self.sendContent("Internal server error", status=500)

    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        super().end_headers()

g_glove = None

def main():
    global g_glove
    parser = argparse.ArgumentParser(description='Backend service for word2vec visualizations.')
    parser.add_argument('--glove', type=str, required=True, help='''Filename of word embedding data file or serialized GloveService.
If the filename ends with .txt, it's interpreted as a word embedding data file. Otherwise, .ann and .json files are expected.''')
    parser.add_argument('--port', type=int, default=8080, help='Port of service.')
    parser.add_argument('--globalProjection', action='store_true',
                        help='Use a single SVD for the dataset instead of local projections. Can be overridden per request.')

    args = parser.parse_args()
    loadStateFromSaveFile = not args.glove.endswith(".txt")

    g_glove = glove.GloveService(
        args.glove,
        buildGlobalProjection=args.globalProjection,
        loadStateFromSaveFile=loadStateFromSaveFile
    )

    server_address = ('0.0.0.0', args.port)
    server = http.server.HTTPServer(server_address, MyRequestHandler)
    sys.stderr.write("Service has started on port {}\n".format(args.port))
    server.serve_forever()

if __name__ == "__main__":
    main()
