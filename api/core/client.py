import grpc

from ..proto import face_pb2, face_pb2_grpc


class Client(object):
    def __init__(self, ip_port='0.0.0.0:50051', options=None):
        self.ip_port = ip_port
        self.options = options
        self.channel = grpc.insecure_channel(ip_port, options=options)
        self.stub = face_pb2_grpc.APIStub(self.channel)

    def Exec(self, req):
        reply = self.stub.SearchFace(req)
        return reply
