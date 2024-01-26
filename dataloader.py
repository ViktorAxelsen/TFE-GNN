import dgl
from dgl.data import DGLDataset

from config import Config


config = Config()


class MixTrafficFlowDataset4DGL(DGLDataset):
    def __init__(self, payload_path, header_path):
        self.payload_path = payload_path
        self.header_path = header_path
        super(MixTrafficFlowDataset4DGL, self).__init__(name="MixTrafficFlowDataset4DGL")

    def process(self):
        self.payload_data, self.label = dgl.load_graphs(self.payload_path)
        self.header_data, self.label = dgl.load_graphs(self.header_path)
        self.label = self.label["glabel"]
        assert len(self.payload_data) == len(self.header_data), "Error {} != {}".format(len(self.payload_data), len(self.header_data))


    def __getitem__(self, index):
        start_ind = config.FLOW_PAD_TRUNC_LENGTH * index
        end_ind = start_ind + config.FLOW_PAD_TRUNC_LENGTH
        return self.header_data[start_ind: end_ind], self.payload_data[start_ind: end_ind], self.label[index]

    def __len__(self):
        return int(len(self.payload_data) / config.FLOW_PAD_TRUNC_LENGTH)


if __name__ == '__main__':
    pass
