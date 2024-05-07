"""
`V3X` was my best-performing model over time, based on an
ensemble of gradient-boosted trees.
It never made a killing, but it was super consistent and robust.
This is just an isolated example of the actual model for posterity.
To actually see all the data wrangling and engineering that makes this work,
check out the Jupyter notebook on the repo:
https://github.com/gianlucatruda/numerai
"""

from xgboost import XGBRegressor
from sklearn.linear_model import LassoCV
from sklearn.decomposition import PCA
from sklearn.base import BaseEstimator


class EraEnsemble(BaseEstimator):
    def __init__(self,
                 n_subs=10,
                 pca_frac=None,
                 subalg=XGBRegressor,
                 mainalg=LassoCV,
                 ):
        self.n_subs = n_subs
        self.submodels = []
        self.sub_preds = []
        self.pca_frac = pca_frac
        self.transforms = []
        self.subalg = subalg
        self.mainalg = mainalg

    def get_params(self, *args, **kwargs):
        return {
            'n_subs': self.n_subs,
            'pca_frac': self.pca_frac,
            'subalg': self.subalg,
            'mainalg': self.mainalg,
        }

    def fit(self, df, y, validation=df_val):
        # Partition the "eras" of data
        n_eras = df.era.nunique()
        min_era = df.era.min()
        max_era = df.era.max()
        STEP = n_eras//self.n_subs

        # Loop over era ranges
        for i in range(min_era, max_era, STEP):
            _data = df[df.era.between(i, i+STEP)]
            _target = _data['target']
            _data = _data[FEAT_COLS]
            if self.pca_frac < 1.0:
                _pca = PCA(n_components=self.pca_frac).fit(_data)
                _data = _pca.transform(_data)
                self.transforms.append(_pca)
            if self.subalg == XGBRegressor:
                submodel = self.subalg(verbosity=0).fit(_data, _target)
            else:
                submodel = self.subalg().fit(_data, _target)
            self.submodels.append(submodel)
            if self.pca_frac < 1.0:
                _preds = submodel.predict(
                    _pca.transform(df[FEAT_COLS])
                )
            else:
                _preds = submodel.predict(df[FEAT_COLS])
            self.sub_preds.append(_preds)
        _X = np.array(self.sub_preds).T
        self.mainmodel = self.mainalg().fit(_X, y)

        return self

    def predict(self, df):
        _preds = []
        for i, sm in enumerate(self.submodels):
            if self.pca_frac < 1.0:
                _data = self.transforms[i].transform(df[FEAT_COLS])
            else:
                _data = df[FEAT_COLS]
            _preds.append(sm.predict(_data))
        _X = np.array(_preds).T

        return self.mainmodel.predict(_X)
