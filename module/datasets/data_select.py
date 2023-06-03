import os
from utlis.logger import get_root_logger
from module.datasets.D1_processing_data import Pty1Rating
from module.datasets.D2_processing_data import Pty2Rating
from module.datasets.D3_processing_data import Pty3Rating
from module.datasets.D4_processing_data import Pty4Rating
from module.datasets.D5_processing_data import Pty5Rating
from module.datasets.D6_processing_data import Pty6Rating


class DataSelect:

    def __init__(self,
                 data_name='d1',
                 data_root="D:/QYZ/Code/random_latent_factor/dataset/",
                 logger=get_root_logger()):
        assert isinstance(data_name, str) and data_name in ['d1', 'd2', 'd3', 'd4', 'd5', 'd6']
        assert isinstance(data_root, str) and os.path.exists(data_root)

        self.data_root = data_root
        self.user_rating_data_path = f"D{data_name[-1]}-movie-ratings.xlsx"
        self.user_trust_data_path = f"D{data_name[-1]}-trust-data.xlsx"
        self.logger = logger

    def pty1rating(self):
        return Pty1Rating(root=self.data_root,
                          user_rating_data_path=self.user_trust_data_path,
                          user_trust_data_path=self.user_trust_data_path,
                          logger=self.logger)

    def pty2rating(self):
        return Pty2Rating(root=self.data_root,
                          user_rating_data_path=self.user_trust_data_path,
                          user_trust_data_path=self.user_trust_data_path,
                          logger=self.logger)

    def pty3rating(self):
        return Pty3Rating(root=self.data_root,
                          user_rating_data_path=self.user_trust_data_path,
                          user_trust_data_path=self.user_trust_data_path,
                          logger=self.logger)

    def pty4rating(self):
        return Pty4Rating(root=self.data_root,
                          user_rating_data_path=self.user_trust_data_path,
                          user_trust_data_path=self.user_trust_data_path,
                          logger=self.logger)

    def pty5rating(self):
        return Pty5Rating(root=self.data_root,
                          user_rating_data_path=self.user_trust_data_path,
                          user_trust_data_path=self.user_trust_data_path,
                          logger=self.logger)

    def pty6rating(self):
        return Pty6Rating(root=self.data_root,
                          user_rating_data_path=self.user_trust_data_path,
                          user_trust_data_path=self.user_trust_data_path,
                          logger=self.logger)
