
import warnings
from functools import partial
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from skimage import color

"""
    from: https://github.com/shuchenweng/CT2/blob/main/segm/model/utils.py
"""
_SOURCE_DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))       # last dir
_RESOURCE_DIR = os.path.join(_SOURCE_DIR, 'resources')

def lab_to_rgb(img):
    assert img.dtype == np.float32

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        return (255 * np.clip(color.lab2rgb(img), 0, 1)).astype(np.uint8)

def get_resource_path(path):
   return os.path.join(_RESOURCE_DIR, path)

class ABGamut:
    RESOURCE_POINTS = get_resource_path('ab-gamut.npy')
    RESOURCE_PRIOR = get_resource_path('q-prior.npy')

    DTYPE = np.float32
    EXPECTED_SIZE = 313

    def __init__(self):
        self.points = np.load(self.RESOURCE_POINTS, allow_pickle=True).astype(self.DTYPE)
        self.prior = np.load(self.RESOURCE_PRIOR, allow_pickle=True).astype(self.DTYPE)
        assert self.points.shape == (self.EXPECTED_SIZE, 2)
        assert self.prior.shape == (self.EXPECTED_SIZE,)


class CIELAB:
    L_MEAN = 50

    AB_BINSIZE = 10
    AB_RANGE = [-110 - AB_BINSIZE // 2, 110 + AB_BINSIZE // 2, AB_BINSIZE]
    AB_DTYPE = np.float32

    Q_DTYPE = np.int64

    RGB_RESOLUTION = 101
    RGB_RANGE = [0, 1, RGB_RESOLUTION]
    RGB_DTYPE = np.float64

    def __init__(self, gamut=None):
        self.gamut = gamut if gamut is not None else ABGamut()

        a, b, self.ab = self._get_ab()

        self.ab_gamut_mask = self._get_ab_gamut_mask(
            a, b, self.ab, self.gamut)

        self.ab_to_q = self._get_ab_to_q(self.ab_gamut_mask)
        self.q_to_ab = self._get_q_to_ab(self.ab, self.ab_gamut_mask)

    @classmethod
    def _get_ab(cls):
        a = np.arange(*cls.AB_RANGE, dtype=cls.AB_DTYPE)
        b = np.arange(*cls.AB_RANGE, dtype=cls.AB_DTYPE)

        b_, a_ = np.meshgrid(a, b)
        ab = np.dstack((a_, b_))

        return a, b, ab

    @classmethod
    def _get_ab_gamut_mask(cls, a, b, ab, gamut):
        ab_gamut_mask = np.full(ab.shape[:-1], False, dtype=bool)

        a = np.digitize(gamut.points[:, 0], a) - 1
        b = np.digitize(gamut.points[:, 1], b) - 1

        for a_, b_ in zip(a, b):
            ab_gamut_mask[a_, b_] = True

        return ab_gamut_mask

    @classmethod
    def _get_ab_to_q(cls, ab_gamut_mask):
        ab_to_q = np.full(ab_gamut_mask.shape, -1, dtype=cls.Q_DTYPE)

        ab_to_q[ab_gamut_mask] = np.arange(np.count_nonzero(ab_gamut_mask))

        return ab_to_q

    @classmethod
    def _get_q_to_ab(cls, ab, ab_gamut_mask):
        return ab[ab_gamut_mask] + cls.AB_BINSIZE / 2

    @classmethod
    def _plot_ab_matrix(cls, mat, pixel_borders=False, ax=None, title=None):
        if ax is None:
            _, ax = plt.subplots()

        imshow = partial(ax.imshow,
                         np.flip(mat, axis=0),
                         extent=[*cls.AB_RANGE[:2]] * 2)

        if len(mat.shape) < 3 or mat.shape[2] == 1:
            im = imshow(cmap='jet')

            fig = plt.gcf()
            fig.colorbar(im, cax=fig.add_axes())
        else:
            imshow()

        # set title
        if title is not None:
            ax.set_title(title)

        # set axes labels
        ax.set_xlabel("$b$")
        ax.set_ylabel("$a$")

        # minor ticks
        tick_min_minor = cls.AB_RANGE[0]
        tick_max_minor = cls.AB_RANGE[1]

        if pixel_borders:

            ax.set_xticks(
                np.linspace(tick_min_minor, tick_max_minor, mat.shape[1] + 1),
                minor=True)

            ax.set_yticks(
                np.linspace(tick_min_minor, tick_max_minor, mat.shape[0] + 1),
                minor=True)

            ax.grid(which='minor',
                    color='w',
                    linestyle='-',
                    linewidth=2)

        # major ticks
        tick_min_major = tick_min_minor + cls.AB_BINSIZE // 2
        tick_max_major = tick_max_minor - cls.AB_BINSIZE // 2

        ax.set_xticks(np.linspace(tick_min_major, tick_max_major, 5))
        ax.set_yticks(np.linspace(tick_min_major, tick_max_major, 5))

        # some of this will be obscured by the minor ticks due to a five year
        # old matplotlib bug...
        ax.grid(which='major',
                color='k',
                linestyle=':',
                dashes=(1, 4))

        # tick marks
        for ax_ in ax.xaxis, ax.yaxis:
            ax_.set_ticks_position('both')

        ax.tick_params(axis='both', which='major', direction='in')
        ax.tick_params(axis='both', which='minor', length=0)

        # limits
        lim_min = tick_min_major - cls.AB_BINSIZE
        lim_max = tick_max_major + cls.AB_BINSIZE

        ax.set_xlim([lim_min, lim_max])
        ax.set_ylim([lim_min, lim_max])

        # invert y-axis
        ax.invert_yaxis()

    def bin_ab(self, ab):
        ab_discrete = ((ab + 110) / self.AB_RANGE[2]).astype(int)

        a, b = np.hsplit(ab_discrete.reshape(-1, 2), 2)
        q = self.ab_to_q[a, b].reshape(*ab.shape[:2])

        return q

    def plot_ab_gamut(self, l=50, ax=None):
        assert l >= 50 and l <= 100

        # construct Lab color space slice for given L
        l_ = np.full(self.ab.shape[:2], l, dtype=self.ab.dtype)
        color_space_lab = np.dstack((l_, self.ab))

        # convert to RGB
        color_space_rgb = lab_to_rgb(color_space_lab)

        # mask out of gamut colors
        color_space_rgb[~self.ab_gamut_mask, :] = 255

        # display color space
        self._plot_ab_matrix(color_space_rgb,
                             pixel_borders=True,
                             ax=ax,
                             title=r"$RGB(a, b \mid L = {})$".format(l))

    def plot_empirical_distribution(self, dataset, ax=None, verbose=False):
        # accumulate ab values
        ab_acc = np.zeros([self.AB_RANGE[1] - self.AB_RANGE[0]] * 2)

        for i in range(len(dataset)):
            img = dataset[i]

            if verbose:
                fmt = "\rprocessing image {}/{}"

                print(fmt.format(i + 1, len(dataset)),
                      end=('\n' if i == len(dataset) - 1 else ''),
                      flush=True)

            img = np.moveaxis(img, 0, -1)
            ab_rounded = np.round(img[:, :, 1:].reshape(-1, 2)).astype(int)
            ab_offset = ab_rounded - self.AB_RANGE[0]

            np.add.at(ab_acc, tuple(np.split(ab_offset, 2, axis=1)), 1)

        # convert to log scale
        ab_acc[ab_acc == 0] = np.nan

        ab_acc_log = np.log10(ab_acc) - np.log10(len(dataset))

        # display distribution
        self._plot_ab_matrix(ab_acc_log, ax=ax, title=r"$log(P(a, b))$")




class AnnealedMeanDecodeQ:
    def __init__(self, cielab, T, device='cuda'):
        self.q_to_ab = torch.from_numpy(cielab.q_to_ab).to(device)

        self.T = T

    def __call__(self, q, is_actual=False, appli=False):
        if self.T == 0:
            # makeing this a special case is somewhat ugly but I have found
            # no way to make this a special case of the branch below (in
            # NumPy that would be trivial)
            ab = self._unbin(self._mode(q))
        else:
            if is_actual is False:
                q = self._annealed_softmax(q, appli=appli)

            a = self._annealed_mean(q, 0)
            b = self._annealed_mean(q, 1)
            ab = torch.cat((a, b), dim=1)

        return ab.type(q.dtype)

    def _mode(self, q):
        return q.max(dim=1, keepdim=True)[1]

    def _unbin(self, q):
        _, _, h, w = q.shape

        ab = torch.stack([
            self.q_to_ab.index_select(
                0, q_.flatten()
            ).reshape(h, w, 2).permute(2, 0, 1)

            for q_ in q
        ])

        return ab

    def _annealed_softmax(self, q, appli=False, change_mask=None):
        q_softmax = torch.nn.functional.softmax(q/self.T,dim=1)   # [bs, 313, 256, 256]

        return q_softmax

    def _annealed_mean(self, q, d):
        am = torch.tensordot(q, self.q_to_ab[:, d], dims=((1,), (0,)))

        return am.unsqueeze(dim=1)