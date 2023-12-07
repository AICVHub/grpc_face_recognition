import time
import grpc
from concurrent import futures
import logging
import cv2
import glob
import os
import sys
import numpy as np

sys.path.insert(0, os.getcwd())
sys.path.insert(0, 'ai_bridge')

from ..proto import face_pb2, face_pb2_grpc
from ..utils.tools import img_binary_to_cv, img_cv_to_binary

# from py_imo_core import api as pic
# from configs import auth

logging.basicConfig(
    format='%(levelname)s %(asctime)s [%(filename)s:%(lineno)d] %(message)s',
    level=logging.DEBUG
)


class FaceServicer(face_pb2_grpc.APIServicer):
    def __init__(self, features_dir="data/faces/*.npy", thres=0.8):
        feature_paths = sorted(glob.glob(features_dir))
        features, face_ids = [], []
        for feature_path in feature_paths:
            face_id = int(os.path.split(feature_path)[-1].split('.')[0])
            face_ids.append(face_id)
            feature = np.load(feature_path)
            features.append(feature / np.linalg.norm(feature))

        self.features = np.asarray(features)
        self.face_ids = face_ids

        self.thres = thres

    def SearchFace(self, request, context):
        logging.info("Got request!".format())
        feature = np.asarray(request.feats)
        feature = feature / np.linalg.norm(feature)
        print(len(feature))

        # 人脸比对，得到face_id
        scores = np.dot(self.features, feature.T)
        face_id = self.face_ids[np.argmax(scores)] if np.max(scores) >= self.thres else ''
        print("face rec result: id:{}, score:{}".format(face_id, np.max(scores)))

        reply = self._pase_reply(face_id)
        return reply

    def _pase_reply(self, face_id):
        reply = face_pb2.SearchFaceReply(
            student_id=str(face_id)
        )
        return reply


class Server(object):
    def __init__(self, algo_kwargs, ip_port='0.0.0.0:50051', max_workers=10, options=None):
        self.ip_port = ip_port
        self.max_workers = max_workers
        self.options = options
        self.face_server = FaceServicer(**algo_kwargs)

    def run(self):
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=self.max_workers), options=self.options)
        face_pb2_grpc.add_APIServicer_to_server(self.face_server, server)
        server.add_insecure_port(self.ip_port)
        server.start()
        logging.info('listening on %s...' % self.ip_port)
        server.wait_for_termination()
