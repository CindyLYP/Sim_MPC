import threading
import time

from Communication.RPCComm import Peer
from Client.DataProviders import PreprocessClient, AlignClient
from Utils.Log import Logger
from Crypto import Random
from Crypto.Cipher import AES

iv = Random.new().read(AES.block_size)

def test_align_data():
    print("\n======================== Test aligning a Dataset ========================\n")
    ip_dict = {
        0: "127.0.0.1:19001",
        1: "127.0.0.1:19002",
        2: "127.0.0.1:19003",
        3: "127.0.0.1:19004"
    }
    channel0 = Peer(0, "[::]:19001", 10, ip_dict, 3, logger=Logger(prefix="Channel0:"))
    channel1 = Peer(1, "[::]:19002", 10, ip_dict, 3, logger=Logger(prefix="Channel1:"))
    channel2 = Peer(2, "[::]:19003", 10, ip_dict, 3, logger=Logger(prefix="Channel2:"))
    channel3 = Peer(3, "[::]:19004", 10, ip_dict, 3, logger=Logger(prefix="Channel3:"))
    align_client = AlignClient(channel0, [1, 2, 3], logger=Logger(prefix="align client:"))
    data_client1 = PreprocessClient(channel1,filepath='TestDataset/data',filename='d1.csv',prim_key=0,
                                    iv=iv, key='d1d1d1d1d1d1d1d1', align_id=0,other_data_clients=[2, 3],
                                    logger=Logger(prefix="Data client 1:"))

    data_client2 = PreprocessClient(channel2, filepath='TestDataset/data', filename='d2.csv', prim_key=0,
                                    iv=iv, key='2f342dfwvge412qw', align_id=0, other_data_clients=[1, 3],
                                    logger=Logger(prefix="Data client 2:"))

    data_client3 = PreprocessClient(channel3, filepath='TestDataset/data', filename='d3.csv', prim_key=0,
                                    iv=iv, key='data_client3d9d9', align_id=0, other_data_clients=[1, 2],
                                    logger=Logger(prefix="Data client 3:"))

    data_client1_th = threading.Thread(target=data_client1.start_align)
    data_client2_th = threading.Thread(target=data_client2.start_align)
    data_client3_th = threading.Thread(target=data_client3.start_align)
    align_client_th = threading.Thread(target=align_client.start_align)

    data_client1_th.start()
    data_client2_th.start()
    data_client3_th.start()
    align_client_th.start()

    data_client1_th.join()
    data_client2_th.join()
    data_client3_th.join()
    align_client_th.join()

    print("==========================  finished ==========================")


test_align_data()