import numpy as np


class CroBinaryMeter():
    def __init__(self, meter_name='', classes=['DIS', 'SIM']):
        self.meter_name = meter_name
        self.classes = classes
        self.reset()
        return

    def reset(self):
        '''
            0: dissimilar
            1: similar

            [i,j] the i-th class is predicted as the j-th class.
        '''
        self.hit_matrix = np.zeros((len(self.classes), len(self.classes)))
        return

    def update(self, pred, label):
        if len(pred) == 0:
            return
        for p, l in zip(pred, label):
            self.hit_matrix[int(l), int(p)] += 1
            p, l
        return

    def get_matrix(self):
        return self.hit_matrix / self.hit_matrix.sum(1).reshape(-1, 1)

    def __str__(self):
        return self.report()

    def get_recall(self, idx):
        bottom = self.hit_matrix[idx].sum()
        top = float(self.hit_matrix[idx, idx])
        return top / bottom if bottom != 0 else 0

    def get_precision(self, idx):
        bottom = self.hit_matrix[:, idx].sum()
        top = float(self.hit_matrix[idx, idx])
        return top / bottom if bottom != 0 else 0

    def get_f1score(self, idx):
        r = self.get_recall(idx)
        p = self.get_precision(idx)
        if (p + r) == 0:
            return 0
        return 2 * p * r / (p + r)

    def get_str_hit(self):
        str = '\nHit Matrix:\n'
        for i in range(len(self.classes)):
            str += f'[ {self.classes[i]:5s}:'
            for j in range(len(self.classes)):
                str += f'  {self.hit_matrix[i, j]:6.0f}'
            str += f'\t({self.hit_matrix[i].sum():6.0f} in all.)]\n'
        return str

    def get_str_conf(self):
        conf = self.get_matrix()
        str = '\nConfusion Matrix\n'
        for i in range(len(self.classes)):
            str += f'[ {self.classes[i]:5s}:'
            for j in range(len(self.classes)):
                str += f'\t{conf[i, j]:6.1%}'
            str += f'\t({self.hit_matrix[i].sum():6.0f} in all.)]\n'
        return str

    def get_str_f1score(self, idx):
        return f'F1-score of {self.classes[idx]}: {self.get_f1score(idx):3.1%}'

    def report(self, hit=True, caption=''):
        str = f'\n=========== {self.meter_name}: {caption} ============\n'
        str += f'======================== {self.get_f1score(0):2.1%} Dis F1 =======================\n'
        str += self.get_str_hit() if hit else ''
        # str += self.get_str_conf()
        str += '\n'
        for i, c in enumerate(self.classes):
            str += f'[ {c:5s}:\tPR: {self.get_precision(i):5.1%},\tRR: {self.get_recall(i):5.1%},\t F1: {self.get_f1score(i):5.1%}]\n'

        return str + '^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n'
