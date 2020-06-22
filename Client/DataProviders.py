import numpy as np
import pandas as pd
import threading
import pathlib
import pickle
import traceback
from Client.Client import BaseClient, ClientException
from Communication.Message import MessageType, ComputationMessage
from Communication.Channel import BaseChannel
from Client.Data import DataLoader
from Client.Learning.Losses import LossFunc, MSELoss
from Client.Learning.Metrics import onehot_accuracy
from Utils.Log import Logger
from Crypto import Random
from Crypto.Hash import SHA
from Crypto.Cipher import AES
from Crypto.Cipher import PKCS1_v1_5 as Cipher_pkcs1_v1_5
from Crypto.Signature import PKCS1_v1_5 as Signature_pkcs1_v1_5
from Crypto.PublicKey import RSA
import base64


class DataClient(BaseClient):
    def __init__(self, channel: BaseChannel, data_loader: DataLoader, test_data_loader: DataLoader,
                 server_id: int, triplets_id: int, other_data_clients: list, logger: Logger=None):
        # random generate some to data
        super(DataClient, self).__init__(channel, logger)
        self.data_loader = data_loader
        self.test_data_loader = test_data_loader
        self.batch_data = None
        self.server_id = server_id
        self.triplets_id = triplets_id
        self.other_data_clients = other_data_clients

        # Configs
        self.configs = None

        self.batch_size = None
        self.para = None
        self.other_paras = [None for _ in range(channel.n_clients)]
        self.learning_rate = None

        self.test_mode = False
        self.prediction_mode =False
        self.error = False

        """
        Lock for matrix multiplication triplet
        Note: since we only have one buffer to 
        """
        self.triplet_lock = threading.Lock()

        # 变量储存器，用于Secret Sharing矩阵乘法
        self.current_triplets = [None for _ in range(channel.n_clients)]

        self.shared_own_mat = [None for _ in range(channel.n_clients)]
        self.shared_other_mat = [None for _ in range(channel.n_clients)]
        self.recovered_own_value = [None for _ in range(channel.n_clients)]
        self.recovered_other_value = [None for _ in range(channel.n_clients)]
        self.shared_out_AB = [None for _ in range(channel.n_clients)]
        self.shared_out_BA = [None for _ in range(channel.n_clients)]
        self.own_out = None

        self.logger.log("Client initialized")

    def __calculate_first_hidden_layer(self, other_id):
        """
        This function is to interactively calculate matrix product with dataclient 'otherid'
        It is following the SMC protocol, the multiplication triplet is provided by a third service.
        :param other_id:
        :return:
        """

        # 提供数据作为矩阵乘法中的乘数
        def set_triplet_AB():
            self.send_check_msg(self.triplets_id,
                          ComputationMessage(MessageType.SET_TRIPLET, (1, other_id, self.batch_data.shape,
                                                                       self.para.shape)))

        # 提供参数作为矩阵乘法中的的被乘数
        def set_triplet_BA():
            self.send_check_msg(self.triplets_id,
                          ComputationMessage(MessageType.SET_TRIPLET, (2, other_id, self.other_paras[other_id].shape,
                                                                       (self.batch_data.shape[0], self.other_paras[other_id].shape[0]))))

        def get_triples():
            msg = self.receive_check_msg(self.triplets_id, MessageType.TRIPLE_ARRAY, other_id)
            self.current_triplets[msg.data[0]] = msg.data[1:]

        def share_data():
            self.shared_own_mat[other_id] = self.batch_data * np.random.uniform(0, 1, self.batch_data.shape)
            self.send_check_msg(other_id, ComputationMessage(MessageType.MUL_DATA_SHARE, self.batch_data - self.shared_own_mat[other_id]))

        def share_para():
            self.shared_own_mat[other_id] = self.other_paras[other_id] * \
                                            np.random.uniform(0, 1, self.other_paras[other_id].shape)
            self.send_check_msg(other_id, ComputationMessage(MessageType.MUL_DATA_SHARE, self.other_paras[other_id] - self.shared_own_mat[other_id]))

        def get_other_share():
            other_share = self.receive_check_msg(other_id, MessageType.MUL_DATA_SHARE)
            self.shared_other_mat[other_id] = other_share.data

        def recover_own_value():
            self.send_check_msg(other_id, ComputationMessage(MessageType.MUL_OwnVal_SHARE,
                                                       self.shared_own_mat[other_id] - self.current_triplets[other_id][0]))

        def get_other_value_share():
            msg = self.receive_check_msg(other_id, MessageType.MUL_OwnVal_SHARE)
            self.recovered_other_value[other_id] = self.shared_other_mat[other_id] - self.current_triplets[other_id][1] + msg.data

        def recover_other_value():
            self.send_check_msg(other_id, ComputationMessage(MessageType.MUL_OtherVal_SHARE,
                                                       self.shared_other_mat[other_id] - self.current_triplets[other_id][1]))

        def get_own_value_share():
            msg = self.receive_check_msg(other_id, MessageType.MUL_OtherVal_SHARE)
            self.recovered_own_value[other_id] = self.shared_own_mat[other_id] - self.current_triplets[other_id][0] + msg.data

        def get_shared_out_AB():
            self.shared_out_AB[other_id] = - np.matmul(self.recovered_own_value[other_id],
                                                       self.recovered_other_value[other_id])
            self.shared_out_AB[other_id] += np.matmul(self.shared_own_mat[other_id], self.recovered_other_value[other_id]) + \
                                            np.matmul(self.recovered_own_value[other_id], self.shared_other_mat[other_id]) + self.current_triplets[other_id][2]

        def get_shared_out_BA():
            self.shared_out_BA[other_id] = np.matmul(self.recovered_other_value[other_id], self.shared_own_mat[other_id]) + \
                                           np.matmul(self.shared_other_mat[other_id], self.recovered_own_value[other_id]) + self.current_triplets[other_id][2]

        # Calculate X_own * Theta_other
        def calc_AB():
            # Atomic operation to acquire triplet
            self.triplet_lock.acquire()
            set_triplet_AB()
            get_triples()
            self.triplet_lock.release()
            # release lock
            share_data()
            get_other_share()
            recover_own_value()
            get_other_value_share()
            recover_other_value()
            get_own_value_share()
            get_shared_out_AB()

        # Calculate Theta_own * X_other
        def calc_BA():
            # Atomic operation to acquire triplet
            self.triplet_lock.acquire()
            set_triplet_BA()
            get_triples()
            self.triplet_lock.release()
            # release lock
            share_para()
            get_other_share()
            recover_own_value()
            get_other_value_share()
            recover_other_value()
            get_own_value_share()
            get_shared_out_BA()
        try:
            if other_id < self.client_id:
                calc_AB()
                calc_BA()
            else:
                calc_BA()
                calc_AB()
        except ClientException as e:
            self.logger.logE("Client Exception encountered, stop calculating.")
            self.error = True
        except Exception as e:
            self.logger.logE("Python Exception encountered , stop calculating.")
            self.error = True
        finally:
            return

    def __calc_out_share(self):
        calc_threads = []
        for client in self.other_data_clients:
            calc_threads.append(threading.Thread(target=self.__calculate_first_hidden_layer, args=(client,)))
            calc_threads[-1].start()

        # While do secret-sharing matrix multiplication with other clients, do matrix multiplication on
        # local data and parameter.

        self.own_out = np.matmul(self.batch_data, self.para)

        for calc_thread in calc_threads:
            calc_thread.join()

    def __send_updates_to(self, client_id: int, update: np.ndarray):
        try:
            self.send_check_msg(client_id, ComputationMessage(MessageType.CLIENT_PARA_UPDATE, update))
        except:
            self.logger.logE("Error encountered while sending parameter updates to other data clients")
            self.error = True

    def __recv_updates_from(self, client_id: int):
        try:
            update_msg = self.receive_check_msg(client_id, MessageType.CLIENT_PARA_UPDATE)
            self.other_paras[client_id] -= self.learning_rate * update_msg.data
        except:
            self.logger.logE("Error encountered while receiving parameter updates from other data clients")
            self.error = True

    def __parameter_update(self):
        updates = self.receive_check_msg(self.server_id, MessageType.CLIENT_OUT_GRAD).data
        own_para_grad = self.batch_data.transpose() @ updates
        portion = np.random.uniform(0, 1, len(self.other_data_clients) + 1)
        portion /= np.sum(portion)
        self.para -= self.learning_rate * own_para_grad * portion[0]
        send_update_threads = []
        for i, data_client in enumerate(self.other_data_clients):
            send_update_threads.append(threading.Thread(
                target=self.__send_updates_to, args=(data_client, own_para_grad * portion[i + 1])
            ))
            send_update_threads[-1].start()

        recv_update_threads = []
        for data_client in self.other_data_clients:
            recv_update_threads.append(threading.Thread(
                target=self.__recv_updates_from, args=(data_client,)
            ))
            recv_update_threads[-1].start()
        for thread in send_update_threads + recv_update_threads:
            thread.join()

    def __train_one_round(self):
        """

        :return: `True` if no error occurred during this training round. `False` otherwise
        """
        try:
            """
            Waiting for server's next-round message
            """
            start_msg = self.receive_check_msg(self.server_id,
                                               [MessageType.NEXT_TRAIN_ROUND, MessageType.TRAINING_STOP])
            """
            If server's message is stop message, stop
            otherwise, if server's next-round message's data is "Test", switch to test mode
            """
            if start_msg.header == MessageType.TRAINING_STOP:
                self.logger.log("Received server's training stop message, stop training")
                return False
            else:
                self.test_mode = False
                if start_msg.data == "Test":
                    self.logger.log("Test Round:")
                    self.test_mode = True

        except:
            self.logger.logE("Error encountered while receiving server's start message")
            return False

        """
        Load batch data. If client is in test_mode, load from test data loader. 
        Otherwise, load from train data loader
        """
        if not self.test_mode:
            self.batch_data = self.data_loader.get_batch(self.batch_size)
        else:
            self.batch_data = self.test_data_loader.get_batch(self.test_batch_size)
        """
        Interactively calculate first layer's output. Each data client gets a share of output o_i
        o_1 + o_2 + ... + o_n (elements-wise add) is the actual first layer's output
        """
        self.__calc_out_share()
        if self.error:
            self.logger.logE("Error encountered while calculating shared outputs")
            return False
        """
        Send the first layer's share to server
        """
        try:
            self.send_check_msg(self.server_id,
                                ComputationMessage(MessageType.MUL_OUT_SHARE,
                                                   (self.own_out, self.shared_out_AB, self.shared_out_BA)))
        except:
            self.logger.logE("Error encountered while sending output shares to server")
            return False
        """
        If not in the test_mode, interactively calculate the gradients w.r.t. to data client's share of parameters
        """
        if not self.test_mode:
            try:
                self.__parameter_update()
            except:
                self.logger.logE("Error encountered while updateing parameters")
                return False
            if self.error:
                self.logger.logE("Error encountered while updateing parameters")
                return False

        return True

    def set_config(self, config:dict):
        self.configs = config
        client_dims = config["client_dims"]
        out_dim = config["out_dim"]
        self.batch_size = config["batch_size"]
        self.test_batch_size = config.get("test_batch_size")
        self.learning_rate = config["learning_rate"]
        self.data_loader.sync_data(config["sync_info"])
        self.test_data_loader.sync_data(config["sync_info"])
        for other_id in client_dims:
            self.other_paras[other_id] = \
                np.random.normal(0,
                                 1 / (len(self.other_data_clients) * client_dims[other_id]),
                                 [client_dims[other_id], out_dim])

    def load_parameters(self, directory):
        self.para = np.load(pathlib.Path(directory).joinpath("own_param.npy"))
        self.other_paras = pickle.load(pathlib.Path(directory).joinpath("other_paras.pkl"))

    def save(self, directory):
        np.save(pathlib.Path(directory).joinpath("own_param.npy"), self.para)
        pickle.dump(self.other_paras, pathlib.Path(directory).joinpath("other_paras.pkl"))

    def start_train(self, wait_for_server: float=100):
        """
        :param wait_for_server:
        :return:
        """
        """
        Receive config message from server, then initialize some parameters
        After this, send CLIENT_READY message to server
        """
        self.logger.log("Client started, waiting for server config message with time out %.2f" % wait_for_server)
        try:
            msg = self.receive_check_msg(self.server_id, MessageType.TRAIN_CONFIG, time_out=wait_for_server)
            self.set_config(msg.data)

            self.para = self.other_paras[self.client_id]
            self.send_check_msg(self.server_id, ComputationMessage(MessageType.CLIENT_READY, None))
        except ClientException:
            self.logger.logE("Train not started")
            return
        except Exception as e:
            self.logger.logE("Python Exception encountered, stop.")
            self.logger.logE("Train not started")
            return

        self.logger.log("Received train conifg message: %s" % msg.data)

        n_rounds = 0
        while True:
            train_res = self.__train_one_round()
            n_rounds += 1
            self.logger.log("Train round %d finished" % n_rounds)
            """
            After one train round over, send CLIENT_ROUND_OVER message to server
            """
            try:
                self.send_check_msg(self.server_id, ComputationMessage(MessageType.CLIENT_ROUND_OVER, train_res))
            except:
                self.logger.logE("Error encountered while sending round over message to server")
                break
            if not train_res:
                self.logger.logE("Error encountered while training one round. Stop.")
                break


class LabelClient(BaseClient):
    def __init__(self, channel: BaseChannel, label_loader: DataLoader, test_label_loader: DataLoader,
                 server_id: int, loss_func=None, metric_func=None, logger:Logger=None):
        super(LabelClient, self).__init__(channel, logger)
        self.label_loader = label_loader
        self.test_label_loader = test_label_loader
        self.batch_size = None
        self.test_batch_size = None
        self.server_id = server_id
        if loss_func is None:
            self.loss_func = MSELoss()
        else:
            self.loss_func = loss_func
        if metric_func is None:
            self.metric_func = onehot_accuracy
        else:
            self.metric_func = metric_func

        self.test_mode = False
        self.error = False

        # cached labels
        self.batch_labels = None
        self.compute_grad_thread = None


        self.logger.log("Client initialized")

    def __compute_pred_grad(self):
        preds = self.receive_check_msg(self.server_id, MessageType.PRED_LABEL).data
        loss = self.loss_func.forward(self.batch_labels, preds)

        self.logger.log("Current batch loss: %.4f, accuracy: %.4f" % (loss, self.metric_func(self.batch_labels, preds)))
        grad = self.loss_func.backward()
        self.send_check_msg(self.server_id, ComputationMessage(MessageType.PRED_GRAD, (grad, loss)))

    def __train_one_round(self):
        try:
            start_msg = self.receive_check_msg(self.server_id,
                                               [MessageType.NEXT_TRAIN_ROUND, MessageType.TRAINING_STOP])
            if start_msg.header == MessageType.TRAINING_STOP:
                self.logger.log("Received server's training stop message, stop training")
                return False
            else:
                self.test_mode = False
                if start_msg.data in ["Test", "Predict"]:
                    self.logger.log("Test Round:")
                    self.test_mode = True
        except:
            self.logger.logE("Error encountered while receiving server's start message")
            return False
        try:
            if not self.test_mode:
                self.batch_labels = self.label_loader.get_batch(self.batch_size)
            else:
                self.batch_labels = self.test_label_loader.get_batch(self.test_batch_size)
        except:
            self.logger.logE("Error encountered while loading batch labels")
            return False

        try:
            self.__compute_pred_grad()
        except:
            self.logger.logE("Error encountered while computing prediction gradients")
            return False
        return True

    def set_config(self, config: dict):
        self.batch_size = config["batch_size"]
        self.test_batch_size = config["test_batch_size"]
        self.label_loader.sync_data(config["sync_info"])
        self.test_label_loader.sync_data(config["sync_info"])

    def start_train(self, wait_for_server:float=100):
        """
        :param wait_for_server:
        :return:
        """
        self.logger.log("Client started, waiting for server config message with time out %.2f" % wait_for_server)
        """
        Receive config message from server, then initialize some parameters
        After this, send CLIENT_READY message to server
        """
        try:
            msg = self.receive_check_msg(self.server_id, MessageType.TRAIN_CONFIG, time_out=wait_for_server)
            self.set_config(msg.data)
            self.send_check_msg(self.server_id, ComputationMessage(MessageType.CLIENT_READY, None))
        except ClientException:
            self.logger.logE("Train not started")
            return
        except Exception as e:
            self.logger.logE("Python Exception encountered, stop\n")
            self.logger.logE("Train not started")
            return

        self.logger.log("Received train config message: %s" % msg.data)

        n_rounds = 0
        while True:
            train_res = self.__train_one_round()
            n_rounds += 1
            self.logger.log("Train round %d finished" % n_rounds)
            try:
                self.send_check_msg(self.server_id, ComputationMessage(MessageType.CLIENT_ROUND_OVER, train_res))
            except:
                self.logger.logE("Error encountered while sending round over message to server")
                break
            if not train_res:
                self.logger.logE("Error encountered while training one round. Stop.")
                break


class PreprocessClient(BaseClient):
    def __init__(self, channel: BaseChannel, filepath, filename, prim_key: int, iv, key,
                 align_id: int, other_data_clients: list, logger: Logger = None):
        super(PreprocessClient, self).__init__(channel, logger)
        self.filepath = filepath
        self.filename = filename
        self.data = None
        self.id = None
        self.prim_key = prim_key
        self.iv = iv
        self.align_id = align_id
        self.other_clients = other_data_clients
        self.private_pem = None
        self.public_pem = None
        self.public_pem_list = dict()
        self.aes_key = key
        self.aes_key_list = list()
        self.random_generator = Random.new().read

    def __padding(self, s):
        while len(s) % 16 !=0:
            s += '\0'
        return s

    def __generate_rsa_keys(self):
        rsa = RSA.generate(1024, self.random_generator)
        self.private_pem = rsa.exportKey()
        self.public_pem = rsa.publickey().exportKey()

    def __send_public_key(self):
        for client in self.other_clients:
            res = self.send_check_msg(client, ComputationMessage(MessageType.RSA_KEY, self.public_pem))

    def __receive_public_keys(self):
        for client in self.other_clients:
            msg = self.receive_check_msg(client, MessageType.RSA_KEY)
            self.public_pem_list[client] = msg.data

    def __send_aes_key(self):
        for client in self.other_clients:
            rsa_key = RSA.importKey(self.public_pem_list[client])
            cipher = Cipher_pkcs1_v1_5.new(rsa_key)
            cipher_text = base64.b64encode(cipher.encrypt(self.aes_key.encode('utf-8')))
            res = self.send_check_msg(client, ComputationMessage(MessageType.AES_KEY, cipher_text))

    def __receive_aes_key(self):
        for client in self.other_clients:
            msg = self.receive_check_msg(client,MessageType.AES_KEY)
            rsa_key = RSA.importKey(self.private_pem)
            cipher = Cipher_pkcs1_v1_5.new(rsa_key)
            aes_key = cipher.decrypt(base64.b64decode(msg.data), self.random_generator)
            self.aes_key_list.append(aes_key.decode('utf-8'))

    def __generate_aes_key(self):
        self.__generate_rsa_keys()
        self.__send_public_key()
        self.__receive_public_keys()
        self.__send_aes_key()
        self.__receive_aes_key()
        for key in self.aes_key_list:
            if len(self.aes_key) > len(key):
                self.aes_key = "".join([chr(ord(x) ^ ord(y)) for (x, y) in zip(self.aes_key[:len(key)], key)])
            else:
                self.aes_key = "".join([chr(ord(x) ^ ord(y)) for (x, y) in zip(self.aes_key, key[:len(self.aes_key)])])

    def __load_and_crypto_data(self):
        self.data = pd.read_csv(self.filepath+'/'+self.filename, header=None)
        self.data[self.prim_key] = self.data[self.prim_key].astype(str)
        self.id = self.data[self.prim_key].values.tolist()
        enc_id = list()
        for id in self.id:
            cipher = AES.new(self.aes_key, AES.MODE_CBC,self.iv)
            enc_id.append(cipher.encrypt(self.__padding(id)))
        self.id = enc_id

    def start_align(self):
        self.__generate_aes_key()
        self.__load_and_crypto_data()
        res = self.send_check_msg(self.align_id, ComputationMessage(MessageType.ALIGN_SEND, self.id))
        msg = self.receive_check_msg(self.align_id, MessageType.ALIGN_REC)
        aligned_id = msg.data
        dec_id = list()
        for id in aligned_id:
            cipher = AES.new(self.aes_key, AES.MODE_CBC,self.iv)
            dec_id.append(cipher.decrypt(id).decode('utf-8').replace('\0',''))
        aligned_id = dec_id
        aligned_data = self.data[self.data[self.prim_key].isin(aligned_id)]
        aligned_data.to_csv(self.filepath+'/aligned_'+self.filename, header=None, index=None)


class AlignClient(BaseClient):
    def __init__(self, channel: BaseChannel, data_clients: list, logger: Logger=None):
        super(AlignClient, self).__init__(channel, logger)
        self.data_clients = data_clients
        self.id_lists = list()
        self.aligned_id = None

    def __receive_crypto_ids(self):
        for client in self.data_clients:
            msg = self.receive_check_msg(client, MessageType.ALIGN_SEND)
            self.id_lists.append(msg.data)

    def __send_aligned_id(self):
        for client in self.data_clients:
            res = self.send_check_msg(client, ComputationMessage(MessageType.ALIGN_REC, self.aligned_id))

    def start_align(self):
        self.__receive_crypto_ids()
        ald_id = set(self.id_lists[0])
        for id in self.id_lists[1:]:
            ald_id = ald_id & set(id)
        self.aligned_id = list(ald_id)
        self.__send_aligned_id()