import os
import sys;

sys.path.insert(0, os.getcwd())

from api.core.server import Server


def main():
    MAX_MESSAGE_LENGTH = 1024 * 1024 * 1024

    algo_kwargs = dict(
        features_dir="/home/aimall/data2/projects/kangtong/grpc_face_rec/data/faces/*.npy"
    )
    server = Server(
        algo_kwargs,
        ip_port='0.0.0.0:50053',
        options=[
            ('grpc.max_send_message_length', MAX_MESSAGE_LENGTH),
            ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH),
        ]
    )
    server.run()


if __name__ == '__main__':
    main()
