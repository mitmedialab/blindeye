import http.server
import socketserver
import sys
sys.path.insert(1, '../tf-pose-estimation')
from run_webcam import *

class MyHttpRequestHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.path = './templates/main.html'
        return http.server.SimpleHTTPRequestHandler.do_GET(self)

handler_object = MyHttpRequestHandler

PORT = 8000
my_server = socketserver.TCPServer(("", PORT), handler_object)


my_server.serve_forever()