import time

import numpy as np
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as P
from mindspore.ops import composite as C
from mindspore.ops import functional as F
from mindspore.ops import operations as op
from mindspore import Tensor, context, Parameter
from mindspore.common.initializer import TruncatedNormal
from pathlib import Path
from dataset import ms_map, dataloader
from utils.tools import DataProcessing as DP
from utils.tools import ConfigS3DIS as cfg
class SharedMLP(nn.Cell):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        transpose=False,
        pad_mode='same',
        bn=False,
        activation_fn=None,
        bias=True
    ):
        super(SharedMLP, self).__init__()

        conv_fn = nn.Conv2dTranspose if transpose else nn.Conv2d

        self.conv = conv_fn(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            pad_mode='valid',
            has_bias=bias,
            weight_init=TruncatedNormal(sigma=1e-3)
        )
        self.batch_norm = nn.BatchNorm2d(out_channels, eps=1e-6, momentum=0.99) if bn else None
        self.activation_fn = activation_fn

    def construct(self, input):
        r"""
            construct method
            Parameters
            ----------
            input: ms.Tensor, shape (B, d_in, N, K)
            Returns
            -------
            ms.Tensor, shape (B, d_out, N, K)
        """
        x = self.conv(input)
        if self.batch_norm:
            x = self.batch_norm(x)
        if self.activation_fn:
            x = self.activation_fn(x)
        return x

class LocalSpatialEncoding(nn.Cell):
    def __init__(self, in_channel=10, out_channel=1, use_pos_encoding=True):
        super(LocalSpatialEncoding, self).__init__()

        self.mlp = SharedMLP(in_channel, out_channel, bn=True, activation_fn=nn.LeakyReLU(0.2))
        # self.mlp = SharedMLP(10, d, bn=True, activation_fn=nn.LeakyReLU(0.2))
        self.d = out_channel
        self.use_pos_encoding = use_pos_encoding

    def construct(self, coords, features, neighbor_idx):
        r"""
            construct method
            Parameters
            ----------
            coords: ms.Tensor, shape (B, N, 3)
                coordinates of the point cloud
            features: ms.Tensor, shape (B, d, N, 1)
                features of the point cloud
            neighbor_idx: ms.Tensor, shape (B, N, K)
            Returns
            -------
            ms.Tensor, shape (B, 2*d, N, K)
        """

        idx = neighbor_idx  # (4,40960,16)

        cat = P.Concat(-3)
        if self.use_pos_encoding:
            # finding neighboring points
            extended_idx = P.Tile()(idx.expand_dims(1), (1, 3, 1, 1))  # (4,40960,16) -> (4,1,40960,16) -> (4,3,40960,16)
            xyz_tile = P.Tile()(coords.transpose(0, 2, 1).expand_dims(-1), (1, 1, 1, idx.shape[-1]))  # (4,3,40960) -> (4,3,40960,16)
            neighbor_xyz = P.GatherD()(xyz_tile, 2, extended_idx)  # shape (4, 3, 40960, 16)
            relative_xyz = xyz_tile - neighbor_xyz  # relative_xyz

            relative_dist = P.Sqrt()(P.ReduceSum(keep_dims=True)(P.Square()(relative_xyz), -3))

            # relative point position encoding
            f_xyz = cat((
                relative_dist,  # (4,1,40960,16)
                relative_xyz,  # (4,3,40960,16)
                xyz_tile,  # (4,3,40960,16)
                neighbor_xyz,  # (4,3,40960,16)
            ))  # (4,10,40960,16)

            # ==========> tensorflow 源码
            #  f_xyz = self.relative_pos_encoding(xyz, neigh_idx)
            #  def relative_pos_encoding(self, xyz, neigh_idx):
            #      ...
            #      relative_feature = tf.concat([relative_dis, relative_xyz, xyz_tile, neighbor_xyz], axis=-1)
            #      return relative_feature
            # ==========> tensorflow 源码
        else:
            f_xyz = coords
        f_xyz = self.mlp(f_xyz)  # (4,10,40960,16) -> (4,8,40960,16)

        f_tile = P.Tile()(features, (1, 1, 1, idx.shape[-1]))  # (4, 8, 40960, 1) -> (4,8,40960,16)
        extended_idx_for_feat = P.Tile()(idx.expand_dims(1), (1, f_xyz.shape[1], 1, 1))
        f_neighbours = P.GatherD()(f_tile, 2, extended_idx_for_feat)  # (4,8,40960,16) -> (4,8,40960,16)

        f_concat = cat([f_xyz, f_neighbours])  # (4,8,40960,16) & (4,8,40960,16) -> (4,16,40960,16)

        if self.use_pos_encoding:
            return f_xyz, f_concat
        else:
            return f_concat

class AttentivePooling(nn.Cell):
    def __init__(self, in_channels, out_channels):
        super(AttentivePooling, self).__init__()

        self.score_fn = nn.SequentialCell([
            nn.Dense(in_channels, in_channels, has_bias=False),
            nn.Softmax(-2)
        ])
        self.mlp = SharedMLP(in_channels, out_channels, bn=True, activation_fn=nn.LeakyReLU(0.2))

    def construct(self, x):
        r"""
            construct method
            Parameters
            ----------
            x: ms.Tensor, shape (B, d_in, N, K)
            Returns
            -------
            ms.Tensor, shape (B, d_out, N, 1)
        """
        # computing attention scores
        scores = self.score_fn(x.transpose(0,2,3,1)).transpose(0,3,1,2)

        # sum over the neighbors
        features = scores * x
        features = P.ReduceSum(keep_dims=True)(features, -1) # shape (B, d_in, N, 1)

        return self.mlp(features)

class LocalFeatureAggregation(nn.Cell):
    def __init__(self, d_in, d_out):
        super(LocalFeatureAggregation, self).__init__()

        self.mlp1 = SharedMLP(d_in, d_out // 2, bn=True, activation_fn=nn.LeakyReLU(0.2))
        self.mlp2 = SharedMLP(d_out, 2 * d_out, bn=True)
        self.shortcut = SharedMLP(d_in, 2 * d_out, bn=True)

        self.lse1 = LocalSpatialEncoding(in_channel=10, out_channel=d_out // 2, use_pos_encoding=True)
        # self.lse2 = LocalSpatialEncoding(in_channel=10, out_channel=d_out // 2)
        self.lse2 = LocalSpatialEncoding(in_channel=d_out // 2, out_channel=d_out // 2, use_pos_encoding=False)
        # self.lse2 = LocalSpatialEncoding(d_out // 2)

        self.pool1 = AttentivePooling(d_out, d_out // 2)
        self.pool2 = AttentivePooling(d_out, d_out)

        self.lrelu = nn.LeakyReLU(0.2)

    def construct(self, coords, features, neighbor_idx):
        r"""
            construct method
            Parameters
            ----------
            coords: ms.Tensor, shape (B, N, 3)
                coordinates of the point cloud
            features: ms.Tensor, shape (B, d, N, 1)
                features of the point cloud
            neighbor_idx: ms.Tensor, shape (B, N, 16)
                knn neighbor idx
            Returns
            -------
            ms.Tensor, shape (B, 2*d_out, N, 1)
        """

        # x = self.mlp1(features)  # (4, 8, 40960, 1)
        #
        # x = self.lse1(coords, x, neighbor_idx)  # (4, 16, 40960, 16)
        # x = self.pool1(x)  # (4, 8, 40960, 1)
        #
        # x = self.lse2(coords, x, neighbor_idx)  # coords: (4, 40960, 3); x: (4, 8, 40960, 1)  neighbor_idx:(4, 40960, 16)
        # x = self.pool2(x)
        #
        # return self.lrelu(self.mlp2(x) + self.shortcut(features))

        f_pc = self.mlp1(features)  # (4, 8, 40960, 1)

        f_xyz, f_concat = self.lse1(coords, f_pc, neighbor_idx)  # (4, 8, 40960, 16) (4, 16, 40960, 16)
        f_pc_agg = self.pool1(f_concat)  # (4, 8, 40960, 1)

        f_concat = self.lse2(f_xyz, f_pc_agg, neighbor_idx)  # coords: (4, 40960, 3); x: (4, 8, 40960, 1)  neighbor_idx:(4, 40960, 16)
        f_pc_agg = self.pool2(f_concat)

        return self.lrelu(self.mlp2(f_pc_agg) + self.shortcut(features))


class RandLANet(nn.Cell):
    def __init__(self, d_in, num_classes):
        super(RandLANet, self).__init__()

        self.fc_start = nn.Dense(d_in, 8)
        self.bn_start = nn.SequentialCell([
            nn.BatchNorm2d(8, eps=1e-6, momentum=0.99),
            nn.LeakyReLU(0.2)
        ])

        # encoding layers
        self.encoder = nn.CellList([
            LocalFeatureAggregation(8, 16),
            LocalFeatureAggregation(32, 64),
            LocalFeatureAggregation(128, 128),
            LocalFeatureAggregation(256, 256),
            LocalFeatureAggregation(512, 512)
        ])

        self.mlp = SharedMLP(1024, 1024, bn=True, activation_fn=nn.LeakyReLU(0.2))
        supervise_kwargs = dict(
            bn=True,
            activation_fn=None
        )
        self.supervise = nn.CellList([
            SharedMLP(1024, 13, **supervise_kwargs),
            SharedMLP(512, 13, **supervise_kwargs),
            SharedMLP(256, 13, **supervise_kwargs),
            SharedMLP(128, 13, **supervise_kwargs),
            SharedMLP(32, 13, **supervise_kwargs),
            SharedMLP(32, 13, **supervise_kwargs),
        ])

        Se_kwargs = dict(
            bn=True,
            activation_fn=None
        )
        self.Se = nn.CellList([
            SharedMLP(512, 10, **Se_kwargs),
            SharedMLP(256, 10, **Se_kwargs),
            SharedMLP(128, 10, **Se_kwargs),
            SharedMLP(32, 10, **Se_kwargs),
        ])
        # decoding layers
        decoder_kwargs = dict(
            transpose=True,
            bn=True,
            activation_fn=nn.LeakyReLU(0.2)
        )
        self.decoder = nn.CellList([
            SharedMLP(1536, 512, **decoder_kwargs),
            SharedMLP(768, 256, **decoder_kwargs),
            SharedMLP(384, 128, **decoder_kwargs),
            SharedMLP(160, 32, **decoder_kwargs),
            SharedMLP(64, 32, **decoder_kwargs),
        ])

        # final semantic prediction
        self.fc_end = nn.SequentialCell([
            SharedMLP(32, 64, bn=True, activation_fn=nn.LeakyReLU(0.2)),
            SharedMLP(64, 32, bn=True, activation_fn=nn.LeakyReLU(0.2)),
            nn.Dropout(),
            SharedMLP(32, num_classes)
        ])

    def construct(self, xyz, feature, neighbor_idx, sub_idx, interp_idx, labels):
        r"""
            construct method
            Parameters
            ----------
            xyz: list of ms.Tensor, shape (num_layer, B, N_layer, 3), each layer xyz
            feature: ms.Tensor, shape (B, N, d), input feature [xyz ; feature]
            neighbor_idx: list of ms.Tensor, shape (num_layer, B, N_layer, 16), each layer knn neighbor idx
            sub_idx: list of ms.Tensor, shape (num_layer, B, N_layer, 16), each layer pooling idx
            interp_idx: list of ms.Tensor, shape (num_layer, B, N_layer, 1), each layer interp idx
            Returns
            -------
            ms.Tensor, shape (B, num_classes, N)
                segmentation scores for each point
        """

        feature = self.fc_start(feature).swapaxes(-2, -1).expand_dims(-1)
        feature = self.bn_start(feature)  # shape (B, 8, N, 1)

        #
        print(labels)
        onehot = nn.OneHot(depth=13)
        # print(labels, labels.shape, type(labels))
        multihot_labels = [P.cast(onehot(labels), ms.int32)]
        # print(multihot_labels)
        # <<<<<<<<<< ENCODER

        f_stack = []
        for i in range(5):
            # at iteration i, feature.shape = (B, d_layer, N_layer, 1)
            f_encoder_i = self.encoder[i](xyz[i], feature, neighbor_idx[i])
            f_sampled_i = self.random_sample(f_encoder_i, sub_idx[i])
            feature = f_sampled_i
            #label sample
            # print(i, multihot_labels[i])
            tmp_multihot_label = P.reduce_max(self.gather_neighbour(multihot_labels[i], neighbor_idx[i]), 2)
            # print(1, tmp_multihot_label.shape)
            tmp_multihot_label = P.reduce_max(self.gather_neighbour(tmp_multihot_label, neighbor_idx[i]), 2)

            # print(2, tmp_multihot_label.shape)
            tmp_multihot_label = tmp_multihot_label.expand_dims(2)
            tmp_multihot_label = tmp_multihot_label.transpose(0, 3, 1, 2)
            tmp_multihot_label = P.Squeeze(2)(self.random_sample(tmp_multihot_label, sub_idx[i]).transpose(0, 2, 3, 1))
            # print(3, tmp_multihot_label)
            if i == 0:
                f_stack.append(f_encoder_i)
            f_stack.append(f_sampled_i)
            multihot_labels.append(tmp_multihot_label)
        # # >>>>>>>>>> ENCODER
        feature = self.mlp(f_stack[-1])  # [B, d, N, 1]

        # <<<<<<<<<< DECODER
        supervised_features = []
        tmp_features = self.supervise[0](feature) #[B, d, 13, 1]
        supervised_features.append(P.Squeeze(2)(tmp_features.transpose(0, 2, 3, 1)))
        self_enhance_loss = None
        num_loss = 0
        f_decoder_list = []
        for j in range(5):
            f_interp_i = self.random_sample(feature, interp_idx[-j - 1])  # [B, d, n, 1]
            # f_interp_i = self.nearest_interpolation(feature, interp_idx[-j - 1])
            cat = P.Concat(1)
            f_decoder_i = self.decoder[j](cat((f_stack[-j - 2], f_interp_i)))
            feature = f_decoder_i
            f_decoder_list.append(f_decoder_i)
            if j < 4:
                #se
                se_features = self.Se[j](f_decoder_i)
                target_se_features = P.cast(P.GreaterEqual()(se_features, 0.), ms.int32)
                if self_enhance_loss is None:
                    self_enhance_loss = P.reduce_mean(P.SigmoidCrossEntropyWithLogits()(target_se_features.astype(ms.float32), se_features.astype(ms.float32)))
                    num_loss += 1
                else:
                    self_enhance_loss += P.reduce_mean(P.SigmoidCrossEntropyWithLogits()(target_se_features.astype(ms.float32), se_features.astype(ms.float32)))
                    num_loss += 1
            tmp_features = self.supervise[j+1](feature)
            # print(2333, tmp_features.shape)
            supervised_features.append(P.Squeeze(2)(tmp_features.transpose(0, 2, 3, 1)))


        # >>>>>>>>>> DECODER

        scores = self.fc_end(f_decoder_list[-1])  # [B, num_classes, N, 1]
        self_enhance_loss /= num_loss
        h_loss = P.reduce_mean(P.SigmoidCrossEntropyWithLogits()(multihot_labels[5], supervised_features[0]))
        h_loss += P.reduce_mean(P.SigmoidCrossEntropyWithLogits()(multihot_labels[4], supervised_features[1]))
        h_loss += P.reduce_mean(P.SigmoidCrossEntropyWithLogits()(multihot_labels[3], supervised_features[2]))
        h_loss += P.reduce_mean(P.SigmoidCrossEntropyWithLogits()(multihot_labels[2], supervised_features[3]))
        h_loss /= 4.
        return scores.squeeze(-1), h_loss, self_enhance_loss

    @staticmethod
    def random_sample(feature, pool_idx):
        """
        :param feature: [B, d, N, 1] input features matrix
        :param pool_idx: [B, N', max_num] N' < N, N' is the selected position after pooling
        :return: pool_features = [B, d, N', 1] pooled features matrix
        """

        b, d = feature.shape[:2]
        n_ = pool_idx.shape[1]
        # [B, N', max_num] --> [B, d, N', max_num]
        # pool_idx = P.repeat_elements(pool_idx.expand_dims(1), feature.shape[1], 1)
        pool_idx = P.Tile()(pool_idx.expand_dims(1), (1, feature.shape[1], 1, 1))
        # [B, d, N', max_num] --> [B, d, N'*max_num]
        pool_idx = pool_idx.reshape((b, d, -1))
        pool_features = P.GatherD()(feature.squeeze(-1), -1, pool_idx)
        pool_features = pool_features.reshape((b, d, n_, -1))
        pool_features = P.ReduceMax(keep_dims=True)(pool_features, -1) # [B, d, N', 1]
        return pool_features

    @staticmethod
    def gather_neighbour(pc, neighbor_idx):
        # gather the coordinates or features of neighboring points
        # return: (4, 40960, 16, 13)
        # pc: (4, 40960, 13)
        idx = neighbor_idx
        extended_idx = P.Tile()(idx.expand_dims(1), (1, 13, 1, 1)) #(4, 13, 40960, 16)

        # (4, 13, 40960, 16)
        xyz_tile = P.Tile()(pc.transpose(0, 2, 1).expand_dims(-1), (1, 1, 1, idx.shape[-1]))
        # print(xyz_tile.shape, extended_idx.shape)
        features = P.GatherD()(xyz_tile, 2, extended_idx)
        features = features.transpose(0, 2, 3, 1)
        features = features.astype(ms.float32)
        return features


    @staticmethod
    def nearest_interpolation(feature, interp_idx):
        """
        :param feature: [B, N, 1, d] input features matrix
        :param interp_idx: [B, up_num_points, 1] nearest neighbour index
        :return: [B, up_num_points, 1, d] interpolated features matrix
        """
        feature = P.squeeze(feature, axis=2)
        batch_size = P.shape(interp_idx)[0]
        up_num_points = P.shape(interp_idx)[1]
        interp_idx = P.reshape(interp_idx, (batch_size, up_num_points))
        interpolated_features = P.GatherD()(feature, interp_idx)
        interpolated_features = P.expand_dims(interpolated_features, axis=2)
        return interpolated_features

class RandLAWithLoss(nn.Cell):
    """RadnLA-net with loss"""
    def __init__(self, network, weights, num_classes):
        super(RandLAWithLoss, self).__init__()
        self.se_loss = Parameter(Tensor(0.0, ms.float32), name='se_loss', requires_grad=True)
        self.CE_loss = Parameter(Tensor(0.0, ms.float32), name='CE_loss', requires_grad=True)
        self.h_loss = Parameter(Tensor(0.0, ms.float32), name='h_loss', requires_grad=True)
        self.network = network
        self.weights = weights
        self.num_classes = num_classes
        self.onehot = nn.OneHot(depth=num_classes)
        self.loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=False)

    def construct(self, feature, labels, input_inds, cloud_inds, p0, p1, p2, p3, p4, n0, n1, n2, n3, n4, pl0, pl1, pl2,
                  pl3, pl4, u0, u1, u2, u3, u4):
        xyz = [p0, p1, p2, p3, p4]
        neighbor_idx = [n0, n1, n2, n3, n4]
        sub_idx = [pl0, pl1, pl2, pl3, pl4]
        interp_idx = [u0, u1, u2, u3, u4]
        logits, h_loss, se_loss = self.network(xyz, feature, neighbor_idx, sub_idx, interp_idx, labels)
        logit = logits.swapaxes(-2, -1).reshape((-1, self.num_classes))  # [b*n, 13]
        labels = labels.reshape((-1,))  # [b, n] --> [b*n]
        one_hot_labels = self.onehot(labels)  # [b*n, 13]
        # self.weights = weights.expand_dims(0) # [13,] --> [1, 13]
        weights = self.weights * one_hot_labels  # [b*n, 13]
        weights = P.ReduceSum()(weights, 1)  # [b*n]
        unweighted_loss = self.loss_fn(logit, one_hot_labels)  # [b*n]
        weighted_loss = unweighted_loss * weights  # [b*n]
        CE_loss = weighted_loss.mean()  # [1]

        # return CE_loss, logits
        print(CE_loss, h_loss, se_loss)
        self.CE_loss = CE_loss
        self.h_loss = h_loss
        self.se_loss = se_loss
        return CE_loss + h_loss + se_loss

class TrainingWrapper(nn.Cell):
    """Training wrapper."""
    def __init__(self, network, optimizer, sens=1.0):
        super(TrainingWrapper, self).__init__(auto_prefix=False)
        self.network = network
        self.network.set_grad()
        self.weights = optimizer.parameters
        self.optimizer = optimizer
        self.grad = C.GradOperation(get_by_list=True, sens_param=True)
        self.sens = sens

    def construct(self, *args):
        """Build a forward graph"""
        weights = self.weights
        loss, logits = self.network(*args)
        sens = op.Fill()(op.DType()(loss), op.Shape()(loss), self.sens)
        grads = self.grad(self.network, weights)(*args, sens)
        return F.depend(loss, self.optimizer(grads)), logits


def get_param_groups(network):
    """Param groups for optimizer."""
    decay_params = []
    no_decay_params = []
    for x in network.trainable_params():
        parameter_name = x.name
        if parameter_name.endswith('.bias'):
            # all bias not using weight decay
            no_decay_params.append(x)
        elif parameter_name.endswith('.gamma'):
            # bn weight bias not using weight decay, be carefully for now x not include BN
            no_decay_params.append(x)
        elif parameter_name.endswith('.beta'):
            # bn weight bias not using weight decay, be carefully for now x not include BN
            no_decay_params.append(x)
        else:
            decay_params.append(x)

    return [{'params': no_decay_params, 'weight_decay': 0.0}, {'params': decay_params}]


if __name__ == '__main__':
    import time



    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU", device_id=0)
    context.set_context(max_call_depth=2048)

    print('Computing weights...')

    n_samples = Tensor(cfg.class_weights, ms.float32)
    ratio_samples = n_samples / n_samples.sum()
    weights = 1 / (ratio_samples + 0.02)
    weights.expand_dims(axis=0)

    print('Done')

    d_in = 6
    model = RandLANet(d_in, cfg.num_classes)
    # network = RandLAWithLoss(model, weights, cfg.num_classes)

    dir = Path('/home/ubuntu/hdd2/lkl/dataset/s3dis/input_0.040')
    print('generating data loader....')
    train_ds, val_ds, dataset = dataloader(dir, val_area='Area_5', num_parallel_workers=8, shuffle=False)
    train_loader = train_ds.batch(batch_size=1,
                                  per_batch_map=ms_map,
                                  input_columns=["xyz", "colors", "labels", "q_idx", "c_idx"],
                                  output_columns=["features", "labels", "input_inds", "cloud_inds",
                                                  "p0", "p1", "p2", "p3", "p4",
                                                  "n0", "n1", "n2", "n3", "n4",
                                                  "pl0", "pl1", "pl2", "pl3", "pl4",
                                                  "u0", "u1", "u2", "u3", "u4"],
                                  num_parallel_workers=8,
                                  drop_remainder=True)
    train_loader = train_loader.create_dict_iterator()
    for i, data in enumerate(train_loader):
        # data prepare
        features = data['features']
        labels = data['labels']
        print(labels)
        xyz = [data['p0'], data['p1'], data['p2'], data['p3'], data['p4']]
        neigh_idx = [data['n0'], data['n1'], data['n2'], data['n3'], data['n4']]
        sub_idx = [data['pl0'], data['pl1'], data['pl2'], data['pl3'], data['pl4']]
        interp_idx = [data['u0'], data['u1'], data['u2'], data['u3'], data['u4']]

        # predict
        pred, b, cx = model(xyz, features, neigh_idx, sub_idx, interp_idx, labels)
        # print('step:', i, ' pred.shape:', pred.shape)