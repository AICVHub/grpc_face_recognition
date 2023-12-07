import logging
import time
import cv2
import numpy as np
import glob
import os
import sys

sys.path.insert(0, os.getcwd())
logging.basicConfig(
    format='%(levelname)s %(asctime)s [%(filename)s:%(lineno)d] %(message)s',
    level=logging.DEBUG
)

from api.core.client import Client
from api.proto import face_pb2


def main():
    MAX_MESSAGE_LENGTH = 1024 * 1024 * 1024
    client = Client(
        ip_port='0.0.0.0:50050',
        options=[
            ('grpc.max_send_message_length', MAX_MESSAGE_LENGTH),
            ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH),
        ]
    )

    feature = np.load("data/faces/10001.npy")
    req = face_pb2.ExecReq(
        feature=[f for f in feature],
    )
    reply = client.Exec(req)

    # parse gRPC result
    face_id = reply.face_id

    print(face_id)


if __name__ == '__main__':
    main()
